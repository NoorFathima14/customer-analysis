import numpy as np
import re
import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

class ReviewPreprocessor:
    """Handles review text preprocessing (cleaning, stopword removal, lemmatization)."""
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.stopwords.update(["the", "and", "it", "is", "to", "for", "of", "this", "that", "in", "a", "an", "on", "with"])
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        """Cleans and tokenizes text"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stopwords]
        return " ".join(words)

class MainTopicModel:
    """Main topic detection using BERTopic with Seeded Topic Labeling (Anchor Words)."""

    def __init__(self, embedding_model, anchor_file="anchor_topicwords.json"):
        self.embedding_model = embedding_model
        self.topic_model = BERTopic(embedding_model=self.embedding_model, calculate_probabilities=True, n_gram_range=(1, 3))

        # Define anchor words for each main topic
        self.anchor_words = self.load_anchor_words(anchor_file)
        self.anchor_embeddings = {label: self.embedding_model.encode(words) for label, words in self.anchor_words.items()}

    def load_anchor_words(self, file_path):
        """Loads anchor words from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Anchor words file '{file_path}' not found!")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def train_topic_model(self, processed_reviews):
        """Train BERTopic model on the dataset"""
        self.topics, self.probs = self.topic_model.fit_transform(processed_reviews)
        self.topic_model.reduce_topics(processed_reviews, nr_topics=15)
        assigned_topics, _ = self.topic_model.transform(processed_reviews)
        self.topic_model.reduce_outliers(processed_reviews, assigned_topics, strategy="c-TF-IDF", threshold=0.5)
        print("Topic model trained.")

    def label_topic_with_seeds(self, topic_id):
        """Assigns labels to topics using predefined anchor words"""
        rep = self.topic_model.get_topic(topic_id)
        if rep is None or len(rep) == 0:
            return "Others"
        rep_words = [word for word, score in rep]
        rep_embeds = self.embedding_model.encode(rep_words)
        scores = {label: util.cos_sim(rep_embeds, anchors).mean().item() for label, anchors in self.anchor_embeddings.items()}
        return max(scores, key=scores.get)

    def assign_seeded_labels(self):
        """Assigns topic labels using anchor words"""
        topic_info = self.topic_model.get_topic_info()
        self.seeded_labels = {topic_id: ("Others" if topic_id == -1 else self.label_topic_with_seeds(topic_id)) for topic_id in topic_info["Topic"]}
        self.topic_model.set_topic_labels(self.seeded_labels)
        print("Seeded Topic Labels:", self.seeded_labels)

    def predict_review_labels(self, review, threshold=0.2):
        """Predicts multiple main topics for a given review"""
        review_embed = self.embedding_model.encode(review)
        scores = {label: util.cos_sim(review_embed, anchors).mean().item() for label, anchors in self.anchor_embeddings.items()}
        assigned = {label: score for label, score in scores.items() if score >= threshold}
        if not assigned:
            assigned["Others"] = 1.0
        return dict(sorted(assigned.items(), key=lambda x: x[1], reverse=True))

class SubtopicModel:
    """Subtopic classification within each main topic using anchor words."""

    def __init__(self, embedding_model, subtopic_file="subtopics.json"):
        self.embedding_model = embedding_model
        self.subtopic_anchor_words = self.load_subtopics(subtopic_file)
        self.subtopic_anchor_embeddings = {
            main: {sub: self.embedding_model.encode(words) for sub, words in subs.items()} for main, subs in self.subtopic_anchor_words.items()
        }

    def load_subtopics(self, file_path):
        """Loads subtopics from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Subtopics file '{file_path}' not found!")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_review_subtopics(self, review, main_labels, threshold=0.1):
        """Predicts the best subtopic for each main topic"""
        review_embed = self.embedding_model.encode(review)
        subtopic_predictions = {}
        for main_topic in main_labels:
            if main_topic not in self.subtopic_anchor_embeddings:
                continue
            best_subtopic, best_score = None, 0.0
            for subtopic, anchors in self.subtopic_anchor_embeddings[main_topic].items():
                avg_sim = util.cos_sim(review_embed, anchors).mean().item()
                if avg_sim > best_score:
                    best_score, best_subtopic = avg_sim, subtopic
            if best_subtopic and best_score >= threshold:
                subtopic_predictions[main_topic] = (best_subtopic, best_score)
        return subtopic_predictions

class TopicModel:
    """Main execution class combining both Main Topics & Subtopics."""

    def __init__(self, synthetic_file="synthetic_delivery_reviews.txt", dataset_category="raw_review_Amazon_Fashion", model_path="bertopic_model.pkl"):
        self.synthetic_file = synthetic_file
        self.dataset_category = dataset_category
        self.preprocessor = ReviewPreprocessor()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.main_topic_model = MainTopicModel(self.embedding_model)
        self.subtopic_model = SubtopicModel(self.embedding_model)
        self.model_path = model_path

    def load_datasets(self, size=7000):
        """Loads synthetic dataset and multiple Amazon review categories."""

        # List of categories to include
        categories = ["Amazon_Fashion"]
        DATASET_SIZE=round(size/len(categories))
        SYNDATA_SIZE=6000
        # Load synthetic reviews
        with open(self.synthetic_file, "r", encoding="utf-8") as f:
            synthetic_reviews = [line.strip() for line in f.readlines() if line.strip()]
        synthetic_reviews = synthetic_reviews[:SYNDATA_SIZE]  # Limit to 6,000 reviews

        amazon_reviews = []
        for category in categories:
            dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}",
                                  split="full", streaming=True, trust_remote_code=True)
            category_reviews = list(dataset.shuffle(buffer_size=5000, seed=42).take(DATASET_SIZE))
            amazon_reviews.extend(category_reviews)  # Append category reviews

        print(f"Loaded {len(amazon_reviews)} Amazon reviews across {len(categories)} categories.")

        synthetic_dicts = [{"text": review} for review in synthetic_reviews] # Convert to dictionary format for consistency
        combined_reviews = synthetic_dicts + amazon_reviews

        self.review_texts = list(dict.fromkeys([review["text"] for review in combined_reviews if "text" in review])) #remove duplicates

        # Preprocess all reviews
        self.processed_reviews = [self.preprocessor.preprocess(text) for text in self.review_texts]

        print(f"Total unique preprocessed reviews: {len(self.processed_reviews)}")

    def train(self):
        """Trains topic model on processed dataset."""
        self.main_topic_model.train_topic_model(self.processed_reviews)
        self.main_topic_model.assign_seeded_labels()

    def predict(self, review):
        """Predicts main and subtopics for given reviews and returns structured output."""
        main_labels = self.main_topic_model.predict_review_labels(review, threshold=0.2)

        topics_output = {"main": [], "subtopics": {}}

        for label, main_score in main_labels.items():
            topics_output["main"].append(label)

            sub_pred = self.subtopic_model.predict_review_subtopics(review, {label: main_score}, threshold=0.1)

            if label in sub_pred:
                subtopic, sub_score = sub_pred[label]
                if label not in topics_output["subtopics"]:
                    topics_output["subtopics"][label] = []
                topics_output["subtopics"][label].append(subtopic)

        return {"topics": topics_output}

    def save_model(self):
        """Saves the trained BERTopic model and subtopic embeddings."""
        save_data = {
            "bertopic_model": self.main_topic_model.topic_model,  # Ensure BERTopic is stored properly
            "subtopic_embeddings": self.subtopic_model.subtopic_anchor_embeddings
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(save_data, f)  # Save the dictionary, not just BERTopic

        print(f"Model saved to {self.model_path}")

    @classmethod
    def load_model(cls, model_path="bertopic_model.pkl"):
        """Loads the saved BERTopic model and subtopic embeddings, and returns an instance of TopicModel."""
        with open(model_path, "rb") as f:
            loaded_data = pickle.load(f)

        # Create a new TopicModel instance
        instance = cls(model_path)

        # If the pickle contains just BERTopic, wrap it in a dictionary
        if isinstance(loaded_data, BERTopic):
            instance.main_topic_model.topic_model = loaded_data
            print("Warning: Old model format detected. Subtopic embeddings not loaded.")
        elif isinstance(loaded_data, dict):
            # Properly restore both BERTopic and subtopics
            instance.main_topic_model.topic_model = loaded_data["bertopic_model"]
            instance.subtopic_model.subtopic_anchor_embeddings = loaded_data.get("subtopic_embeddings", {})
        else:
            raise ValueError("Invalid model format: expected BERTopic or dictionary.")

        print(f"Model loaded from {model_path}")
        return instance


    def topic_analysis(self):
        """Loads data, trains the topic model, and saves it."""
        print("Loading datasets...")
        self.load_datasets()

        print("Training topic model...")
        self.train()

        print("Saving trained model...")
        self.save_model()

        print("Topic analysis complete! Model saved successfully.")

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
