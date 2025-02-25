import torch
from torch.utils.data import Dataset

class PreTokenizedDataset(Dataset):
    def __init__(self, texts, emotion_columns, tokenizer, max_length=128):
        # Pre-tokenize the entire list of texts once
        self.texts = texts
        self.labels = emotion_columns
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )

    def __getitem__(self, idx):
        # Retrieve pre-tokenized data for a given index
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
