from torch.utils.data import Dataset
import torch

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, emotion_columns, tokenizer, max_length):
        """
        Args:
            texts (list): List of text samples.
            emotion_columns (list): List of lists, where each inner list contains binary values for emotions.
            tokenizer: BERT tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.texts = texts
        self.labels = emotion_columns
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }