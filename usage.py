from detector import EmotionDetector
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load GoEmotions dataset
dataset = load_dataset("go_emotions", "raw")

# Extract texts and emotion columns from the 'train' split
train_texts = dataset["train"]["text"]
print("Fetched dataset.")

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

# Create labels for the train set
train_labels = [
    [dataset["train"][emotion][i] for emotion in emotion_columns]
    for i in range(int(len(dataset["train"]) * 0.0001))
]

print("Processed train labels")

subset_size = int(0.0001 * len(train_texts))  # Adjust the fraction as needed
train_texts = train_texts[:subset_size]

# Split the train data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=0.2,
    random_state=42,  # 20% validation
)

# Initialize detector
detector = EmotionDetector()

# Train
detector.train(train_texts, train_labels, val_texts, val_labels)

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
