import torch
import torch.nn as nn
#from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from typing import List, Dict
from sklearn.metrics import f1_score
from config import EmotionConfig
from utils import EmotionPostprocessor
from dataset import GoEmotionsDataset
import numpy as np

class BertForEmotionClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_LABELS)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = self.dropout(outputs.pooler_output)
        # return self.classifier(pooled_output)
        hidden_state = outputs[0]  # Last hidden state, shape: (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # Use the first token ([CLS]) representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EmotionDetector:
    def __init__(self, config=EmotionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = self._initialize_model()
        self.label_names = self._load_label_names()

    def _initialize_model(self) -> nn.Module:
        model = BertForEmotionClassification(self.config).to(self.device)
        return model
    
    def _load_label_names(self) -> List[str]:
        return [
        'joy', 'excitement', 'gratitude', 'approval',
        'anger', 'disappointment', 'disgust', 'sadness',
        'confusion', 'surprise'
    ]

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False  # Exclude token_type_ids
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text: str) -> List[Dict]:
        # Preprocess
        inputs = self.preprocess(text)
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs)
        
        # Post-process
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return self._format_output(probs)

    def _format_output(self, probs: np.ndarray) -> List[Dict]:
        results = []
        for i, prob in enumerate(probs):
            # if prob > self.config.MEDIUM_ACTIVATION:
            emotion_name = self.label_names[i]
            plutchik_mapping = EmotionPostprocessor.map_to_plutchik([emotion_name])
            results.append({
                "emotion": emotion_name,
                "score": float(prob),
                "activation": EmotionPostprocessor.get_activation_level(prob),
                "category": plutchik_mapping[emotion_name]  # Access the dictionary correctly
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def train(self, train_loader, val_loader):
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        loss_fn = nn.BCEWithLogitsLoss()

        # Training loop
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = loss_fn(outputs, labels.float())
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation step
            val_loss, val_f1 = self.evaluate(val_loader, loss_fn)
            print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    def evaluate(self, val_loader, loss_fn):
        self.model.eval()
        total_loss = 0
        total_f1 = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = loss_fn(outputs, labels.float())
                total_loss += loss.item()
                
                # Convert logits to binary predictions
                preds = torch.sigmoid(outputs) > 0.5
                f1 = f1_score(labels.cpu(), preds.cpu(), average='micro')
                total_f1 += f1
        
        return total_loss / len(val_loader), total_f1 / len(val_loader)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    @classmethod
    def load_model(cls, path: str):
        checkpoint = torch.load(path, weights_only=False)
        detector = cls(config=checkpoint['config'])
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        return detector