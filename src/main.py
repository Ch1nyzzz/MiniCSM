import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import ClimateDataset, collate_fn
from distill_utils import DistillUtils
from train import train
from config import Args
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    args = Args()
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_name_or_path).to("cuda")
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_name_or_path).to("cuda")

    df = pd.read_csv("dataset/perspectrum_trainsamples_25000.csv")
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=args.seed)

    train_dataset = ClimateDataset(train_df, tokenizer, args.max_seq_length)
    eval_dataset = ClimateDataset(eval_df, tokenizer, args.max_seq_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length)
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length)
    )

    teacher_hidden_size = teacher_model.config.d_model
    student_hidden_size = student_model.config.d_model
    distill_utils = DistillUtils(teacher_hidden_size, student_hidden_size, "cuda")

    optimizer = AdamW(
        list(student_model.parameters()) +
        list(distill_utils.encoder_proj.parameters()) +
        list(distill_utils.decoder_proj.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * len(train_dataloader) * args.num_train_epochs)
    )

    train(args, student_model, teacher_model, train_dataloader, eval_dataloader, distill_utils, optimizer, scheduler)
