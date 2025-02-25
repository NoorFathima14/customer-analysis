import os
import pickle

def process_train_labels(dataset, emotion_columns, subset_size, cache_file="train_labels.pkl"):
    """
    Process and cache the training labels from the dataset.

    Args:
        dataset: The Hugging Face dataset object.
        emotion_columns: List of emotion column names to extract.
        subset_size: Number of samples to process.
        cache_file: Filename to cache the processed labels.

    Returns:
        A list of lists containing labels for each sample.
    """
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading cached train labels from {cache_file}")
        with open(cache_file, "rb") as f:
            train_labels = pickle.load(f)
        return train_labels

    print("Processing train labels...")
    # Use vectorized slicing and zip to extract labels quickly
    train_labels = list(zip(*[dataset["train"][emotion][:subset_size] for emotion in emotion_columns]))
    train_labels = [list(labels) for labels in train_labels]

    # Cache the processed labels for future use
    with open(cache_file, "wb") as f:
        pickle.dump(train_labels, f)

    return train_labels