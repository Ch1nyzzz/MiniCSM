import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillUtils:
    def __init__(self, teacher_hidden_size, student_hidden_size, device):
        self.encoder_proj = nn.Linear(teacher_hidden_size, student_hidden_size).to(device)
        self.decoder_proj = nn.Linear(teacher_hidden_size, student_hidden_size).to(device)
        self.layernorm = nn.LayerNorm(student_hidden_size).to(device)

    def distill_intermediate_representations(
        self, teacher_outputs, student_outputs
    ):
        teacher_encoder_hs = teacher_outputs.encoder_hidden_states
        student_encoder_hs = student_outputs.encoder_hidden_states
        teacher_decoder_hs = teacher_outputs.decoder_hidden_states
        student_decoder_hs = student_outputs.decoder_hidden_states

        # Encoder对齐
        encoder_loss = 0.0
        for t_enc, s_enc in zip(
            teacher_encoder_hs[-3:], student_encoder_hs[-3:]
        ):
            t_enc_proj = self.layernorm(self.encoder_proj(t_enc))
            s_enc = self.layernorm(s_enc)
            encoder_loss += F.mse_loss(s_enc, t_enc_proj, reduction='mean')

        # Decoder对齐
        decoder_loss = 0.0
        for t_dec, s_dec in zip(
            teacher_decoder_hs[-3:], student_decoder_hs[-3:]
        ):
            t_dec_proj = self.layernorm(self.decoder_proj(t_dec))
            s_dec = self.layernorm(s_dec)
            decoder_loss += F.mse_loss(s_dec, t_dec_proj, reduction='mean')

        return (encoder_loss + decoder_loss) / 2

    def get_distil_loss(
        self, teacher_outputs, student_outputs, labels, temp, kd_ratio, inter_loss_weight
    ):
        teacher_logits = teacher_outputs.logits
        student_logits = student_outputs.logits

        # Logits层蒸馏
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_probs = F.softmax(student_logits / temp, dim=-1)
        kl_loss = F.kl_div(
            student_probs.log(), teacher_probs, reduction="batchmean"
        )

        # 中间层对齐蒸馏
        inter_loss = self.distill_intermediate_representations(
            teacher_outputs, student_outputs
        )

        return kd_ratio * kl_loss * temp ** 2 + inter_loss_weight * inter_loss
