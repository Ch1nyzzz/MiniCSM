import pandas as pd
import torch
from torch.utils.data import Dataset

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

def collate_fn(batch, tokenizer, max_seq_length):
    inputs = [item['input_text'] for item in batch]
    stances = [item['stance'] for item in batch]
    targets = [stance.lower() for stance in stances]

    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
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

    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels,
    }
