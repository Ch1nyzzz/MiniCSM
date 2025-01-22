import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    get_constant_schedule_with_warmup
)
from peft import PeftModel, PeftConfig
from sklearn.model_selection import train_test_split
import wandb

# 设置随机种子
torch.manual_seed(41)

# 定义参数
class Args:
    def __init__(self):
        self.new_model_name = "mse-nokd"
        self.teacher_model_name_or_path = "ppaudel/ctd-flant5-xxl"  # 教师模型的微调权重 (T5-XXL)
        self.base_model_name_or_path = "philschmid/flan-t5-xxl-sharded-fp16"  # 教师基础模型
        self.student_model_name_or_path = "google/flan-t5-base"  # 学生模型 (T5-Small)
        self.batch_size = 16
        self.temp = 3.0
        self.eval_batch_size = 16
        self.num_train_epochs = 2
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.max_seq_length = 512
        self.save_steps = 1000
        self.eval_steps = 1000
        self.logging_steps = 100
        self.seed = 41
        self.kd_ratio = 0.0 # 知识蒸馏损失比例
        self.intermediate_layer_distil_weight = 0.1 # 中间层蒸馏损失权重
        self.attention_distil_weight = 0.0000# 注意力蒸馏损失权重
        self.do_train = True
        self.do_eval = True
        self.cache_dir = "../ec593/models"
        self.checkpoint_dir = "checkpoint-epoch-9"

args = Args()

wandb.init(project=args.new_model_name, name=args.new_model_name)

device = torch.device("cuda")

# 加载学生模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.student_model_name_or_path,
    cache_dir=args.cache_dir
)

# 加载教师模型
config = PeftConfig.from_pretrained(
    args.teacher_model_name_or_path,
    cache_dir=args.cache_dir
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    args.base_model_name_or_path,
    torch_dtype=torch.float32,
    device_map="auto",
    cache_dir=args.cache_dir,
    load_in_8bit=True
)
teacher_model = PeftModel.from_pretrained(
    base_model,
    args.teacher_model_name_or_path,
    cache_dir=args.cache_dir
)
teacher_model.eval()

# 加载学生模型
student_model = AutoModelForSeq2SeqLM.from_pretrained(
    args.student_model_name_or_path,
    torch_dtype=torch.float32,
    device_map="auto",
    cache_dir=args.cache_dir,
)

# 获取教师与学生的配置以提取hidden_size
teacher_config = teacher_model.base_model.config
student_config = student_model.config

teacher_hidden_size = teacher_config.d_model
student_hidden_size = student_config.d_model

# 为隐藏层表示构建投影层，将teacher维度映射到student维度
encoder_proj = nn.Linear(teacher_hidden_size, student_hidden_size).to(device)
decoder_proj = nn.Linear(teacher_hidden_size, student_hidden_size).to(device)

df = pd.read_csv("dataset/dataset/perspectrum_trainsamples_25000.csv")

class ClimateDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        stance = str(row['output'])
        input_text = str(row['text']).replace("'", "")
        return {'input_text': input_text, 'stance': stance}

def collate_fn(batch):
    inputs = [item['input_text'] for item in batch]
    stances = [item['stance'] for item in batch]
    targets = [stance.lower() for stance in stances]

    model_inputs = tokenizer(
        inputs,
        max_length=args.max_seq_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        targets,
        max_length=5,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    model_batch = {
        'input_ids': model_inputs['input_ids'].to(device),
        'attention_mask': model_inputs['attention_mask'].to(device),
    }

    no_model_batch = {
        'labels': labels.to(device),
    }

    return model_batch, no_model_batch

train_df, eval_df = train_test_split(df, test_size=0.1, random_state=args.seed)

train_dataset = ClimateDataset(train_df, tokenizer, args.max_seq_length)
eval_dataset = ClimateDataset(eval_df, tokenizer, args.max_seq_length)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=args.eval_batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# 假设学生8层，教师24层
# 我们对每个学生层i，对齐教师的layer_map[i]
# 等间距选择教师层：每隔24/8 = 3层取一层
layernorm = nn.LayerNorm(student_hidden_size).to(device)

def distill_intermediate_representations(args, teacher_outputs, student_outputs, no_model_batch):
    """
    只对齐教师模型和学生模型的「最后三层」隐藏状态，
    并使用 MSE 作为蒸馏损失。
    """
    # 获取所有层的输出
    teacher_encoder_hs = teacher_outputs.encoder_hidden_states  # list, 每层 shape = [batch, seq_len, teacher_hidden_size]
    student_encoder_hs = student_outputs.encoder_hidden_states
    teacher_decoder_hs = teacher_outputs.decoder_hidden_states
    student_decoder_hs = student_outputs.decoder_hidden_states

    # 找到最后三层的索引
    # 例如，教师 encoder 共有 25 个 hidden_states (含 embedding 0 层)，最后 3 层索引就是 [-3, -2, -1]
    teacher_enc_len = len(teacher_encoder_hs)
    student_enc_len = len(student_encoder_hs)
    last3_teacher_enc_idx = [teacher_enc_len - 3, teacher_enc_len - 2, teacher_enc_len - 1]
    last3_student_enc_idx = [student_enc_len - 3, student_enc_len - 2, student_enc_len - 1]

    # 对 encoder 的最后三层做投影 + MSE
    encoder_loss = 0.0
    for t_idx, s_idx in zip(last3_teacher_enc_idx, last3_student_enc_idx):
        t_enc = teacher_encoder_hs[t_idx]      # [batch, seq_len, teacher_hidden_size]
        s_enc = student_encoder_hs[s_idx]      # [batch, seq_len, student_hidden_size]

        # 将教师 hidden state 用线性层投影到学生维度
        t_enc_proj = encoder_proj(t_enc)       # [batch, seq_len, student_hidden_size]

        # 可选：对齐前先做 LayerNorm，减小数值尺度差异
        t_enc_proj = layernorm(t_enc_proj)
        s_enc = layernorm(s_enc)

        # 计算 MSE
        encoder_loss += F.mse_loss(s_enc, t_enc_proj, reduction='mean')

    # 最后求平均
    encoder_loss = encoder_loss / 3.0

    # 同理，对 decoder 的最后三层做投影 + MSE
    teacher_dec_len = len(teacher_decoder_hs)
    student_dec_len = len(student_decoder_hs)
    last3_teacher_dec_idx = [teacher_dec_len - 3, teacher_dec_len - 2, teacher_dec_len - 1]
    last3_student_dec_idx = [student_dec_len - 3, student_dec_len - 2, student_dec_len - 1]

    decoder_loss = 0.0
    for t_idx, s_idx in zip(last3_teacher_dec_idx, last3_student_dec_idx):
        t_dec = teacher_decoder_hs[t_idx]
        s_dec = student_decoder_hs[s_idx]

        t_dec_proj = decoder_proj(t_dec)
        t_dec_proj = layernorm(t_dec_proj)
        s_dec = layernorm(s_dec)

        decoder_loss += F.mse_loss(s_dec, t_dec_proj, reduction='mean')

    decoder_loss = decoder_loss / 3.0

    # 返回 encoder 和 decoder 的综合蒸馏损失
    return (encoder_loss + decoder_loss) / 2


def get_distil_loss(args, tokenizer, student_model, teacher_model, model_batch, no_model_batch, student_outputs):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch,
                                        labels=no_model_batch["labels"],
                                        use_cache=False,
                                        output_hidden_states=True,
                                        output_attentions=True)
        
        teacher_logits = teacher_outputs.logits

    student_logits = student_outputs.logits

    # 基础KD：logits层面蒸馏
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / args.temp, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits / args.temp, dim=-1, dtype=torch.float32)

    epsilon = 1e-10
    student_probs = torch.clamp(student_probs, epsilon, 1.0)
    teacher_probs = torch.clamp(teacher_probs, epsilon, 1.0)

    kl_loss = F.kl_div(
        teacher_probs.log(),
        student_probs,
        reduction="none"
    )

    mask = (no_model_batch["labels"] != -100).int()
    kl_loss = kl_loss * mask.unsqueeze(-1).float()
    distil_loss = kl_loss.sum() / mask.sum()

    # 中间层表示蒸馏损失（使用映射）
    inter_loss = distill_intermediate_representations(args, teacher_outputs, student_outputs, no_model_batch) * args.intermediate_layer_distil_weight

    # 注意力权重蒸馏损失（使用层映射）
    return distil_loss, inter_loss


optimizer = AdamW(list(student_model.parameters()) + 
                  list(encoder_proj.parameters()) +
                  list(decoder_proj.parameters()), 
                  lr=args.learning_rate, 
                  weight_decay=args.weight_decay)

total_steps = len(train_dataloader) * args.num_train_epochs
scheduler = get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps)
)

epoch = 0
print(f"模型和训练状态未加载，从第 {epoch} 个epoch开始。")

# 训练循环
for epoch in range(epoch, args.num_train_epochs):
    student_model.train()
    encoder_proj.train()
    decoder_proj.train()
    epoch_loss = 0.0
    for step, (model_batch, no_model_batch) in enumerate(tqdm(train_dataloader, desc=f"训练 Epoch {epoch+1}")):
        
        optimizer.zero_grad()
        student_outputs = student_model(
            **model_batch,
            labels=no_model_batch["labels"], 
            output_hidden_states=True, 
            output_attentions=True
        )

        student_logits = student_outputs.logits
        
        distil_loss, inter_loss = get_distil_loss(args, tokenizer, student_model, teacher_model, model_batch, no_model_batch, student_outputs)

        # 原有CE loss
        loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_func(
            student_logits.float().view(-1, student_logits.shape[-1]), 
            no_model_batch["labels"].view(-1)
        )
        #print(f"ce_loss: {ce_loss}")
        # print(f"distil_loss: {distil_loss}")
        # print(f"inter_loss: {inter_loss}")
        # print(f"att_loss: {att_loss}")
        # 综合loss: CE + KD + intermediate + attention
        # loss =  (1 - args.kd_ratio) * ce_loss + args.kd_ratio * distil_loss * args.temp**2 + inter_loss
        loss =  (1 - args.kd_ratio) * ce_loss 
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        if (step + 1) % args.logging_steps == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {epoch_loss / (step + 1):.4f}")
            wandb.log({"loss": epoch_loss / (step + 1)})

    # 每个 epoch 后进行评估
    student_model.eval()
    encoder_proj.eval()
    decoder_proj.eval()

    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for model_batch, no_model_batch in tqdm(eval_dataloader, desc="评估中"):
            outputs = student_model(**model_batch, labels=no_model_batch["labels"])
            eval_loss += outputs.loss.item()

            # 生成预测
            generated_ids = student_model.generate(
                input_ids=model_batch['input_ids'],
                attention_mask=model_batch['attention_mask'],
                max_length=5,
                num_beams=1,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels = no_model_batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(preds, decoded_labels):
                if pred.strip().lower() == label.strip().lower():
                    correct += 1
                total += 1

    avg_eval_loss = eval_loss / len(eval_dataloader)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    wandb.log({"eval_loss": avg_eval_loss, "eval_accuracy": accuracy})

    output_dir = f"{args.new_model_name + str(epoch)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': avg_eval_loss,
    }
    torch.save(checkpoint, 'checkpoint.pth')

wandb.finish()
