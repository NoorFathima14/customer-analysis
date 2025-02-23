from detector import EmotionDetector
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pretokenized_dataset import PreTokenizedDataset
from transformers import DistilBertModel, DistilBertTokenizer

# Load GoEmotions dataset
dataset = load_dataset("go_emotions", "raw")

# Extract texts and emotion columns from the 'train' split
train_texts = dataset["train"]["text"]
print("Fetched dataset.")

SUBSET_FRACTION = 0.0001  # 0.01% of dataset (adjust as needed)
TEST_SIZE = 0.2            # 20% validation split
RANDOM_STATE = 42

# Emotion columns 
emotion_columns = [
    "joy",
    "excitement",
    "gratitude",
    "approval",  
    "anger",
    "disappointment",
    "disgust",
    "sadness",  
    "confusion",
    "surprise",  
]

print("Processing train labels")

total_samples = len(dataset["train"])
subset_size = int(total_samples * SUBSET_FRACTION)

# Handle edge case for tiny datasets
if subset_size < 1:
    print(f"Warning: Subset size {subset_size} < 1. Using minimum 1 sample")
    subset_size = 1

# Create subset of labels and texts using SAME subset size
train_labels = [
    [dataset["train"][emotion][i] for emotion in emotion_columns]
    for i in range(subset_size) 
]

print("Processed train labels")

train_texts = train_texts[:subset_size]

# Split the train data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,  # 20% validation
)

print(f"Processed {subset_size} samples")
print(f"Training set: {len(train_texts)} samples")
print(f"Validation set: {len(val_texts)} samples")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LENGTH = 128

# Create pre-tokenized datasets
train_dataset = PreTokenizedDataset(train_texts, train_labels, tokenizer, max_length=MAX_LENGTH)
val_dataset = PreTokenizedDataset(val_texts, val_labels, tokenizer, max_length=MAX_LENGTH)

# Create DataLoaders from the pre-tokenized datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print("Pre-tokenization complete. DataLoaders are ready.")

# Initialize detector
detector = EmotionDetector()

# Train
detector.train(train_loader, val_loader)

# Save model
detector.save_model("emotion_model.pth")

# Load model
detector = EmotionDetector.load_model("emotion_model.pth")

# Predict
text = "I'm absolutely thrilled with this product! Best purchase ever!"
results = detector.predict(text)

print("Detected Emotions:")
sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
for result in sorted_results:
    print(
        f"{result['emotion'].capitalize()} ({result['activation']} activation): {result['score']:.2f}"
    )
