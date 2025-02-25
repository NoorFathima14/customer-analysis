import numpy as np
import re
import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

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
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.topic_model = BERTopic(embedding_model=self.embedding_model, calculate_probabilities=True, n_gram_range=(1, 3))
        
        # Define anchor words for each main topic
        self.anchor_words = {
            "Delivery": ["delivery", "shipping", "arrived", "late", "package", "courier", "ship"],
            "Quality": ["quality", "durable", "sturdy", "wellmade", "flimsy", "defective"],
            "Fit and Size": ["fit", "size", "measurement", "true", "loose", "tight"],
            "Price": ["price", "cost", "expensive", "cheap", "value"],
            "Customer Service": ["service", "support", "helpful", "friendly", "responsive"],
            "Clothes": ["clothes", "fashion", "design", "style", "wear"],
            "Others": ["other", "miscellaneous"]
        }
        self.anchor_embeddings = {label: self.embedding_model.encode(words) for label, words in self.anchor_words.items()}

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
        return dict(sorted(assigned.items(), key=lambda x: x[1], reverse=True))

class SubtopicModel:
    """Subtopic classification within each main topic using anchor words."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.subtopic_anchor_words = {
            "Delivery": {
                "Speedy Delivery": ["fast", "prompt", "quick", "on time", "rapid", "swift"],
                "Free Delivery": ["free", "no cost", "complimentary"],
                "Slow Delivery": ["slow", "delayed", "late", "took too long", "long wait"]
            },
            "Quality": {
                "High Quality": ["excellent", "premium", "durable", "superior"],
                "Poor Quality": ["flimsy", "defective", "cheap", "poor"],
                "Consistent Quality": ["consistent", "reliable", "steady"]
            },
            "Fit and Size": {
                "Perfect Fit": ["perfect", "ideal", "true to size"],
                "Too Small": ["small", "tight", "cramped"],
                "Too Big": ["big", "loose", "baggy"]
            },
            "Price": {
                "Affordable": ["cheap", "affordable", "good value"],
                "Expensive": ["expensive", "high priced", "overpriced"]
            },
            "Customer Service": {
                "Helpful Service": ["helpful", "friendly", "responsive"],
                "Poor Service": ["unresponsive", "rude", "disappointing"]
            },
            "Clothes": {
                "Stylish": ["stylish", "trendy", "fashionable"],
                "Comfortable": ["comfortable", "soft", "cozy"],
                "Durable": ["durable", "long lasting", "sturdy"]
            },
            "Others": {
                "Miscellaneous": ["miscellaneous", "other", "varied"]
            }
        }
        self.subtopic_anchor_embeddings = {
            main: {sub: self.embedding_model.encode(words) for sub, words in subs.items()} for main, subs in self.subtopic_anchor_words.items()
        }

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

    def __init__(self, synthetic_file="synthetic_delivery_reviews.txt", dataset_category="raw_review_Amazon_Fashion"):
        self.synthetic_file = synthetic_file
        self.dataset_category = dataset_category
        self.preprocessor = ReviewPreprocessor()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.main_topic_model = MainTopicModel(self.embedding_model)
        self.subtopic_model = SubtopicModel(self.embedding_model)

    def load_datasets(self):
        """Loads synthetic and Amazon Fashion datasets, then preprocesses them."""
        with open(self.synthetic_file, "r", encoding="utf-8") as f:
            synthetic_reviews = [line.strip() for line in f.readlines() if line.strip()]
        synthetic_reviews = synthetic_reviews[:6000]
        amazon_fashion = load_dataset("McAuley-Lab/Amazon-Reviews-2023", self.dataset_category, split="full", streaming=True)
        amazon_fashion = list(amazon_fashion.shuffle(buffer_size=14000, seed=42).take(7000))

        synthetic_dicts = [{"text": review} for review in synthetic_reviews]
        combined_reviews = synthetic_dicts + amazon_fashion
        self.review_texts = list(dict.fromkeys([review["text"] for review in combined_reviews if "text" in review]))
        self.processed_reviews = [self.preprocessor.preprocess(text) for text in self.review_texts]

    def train(self):
        """Trains topic model on processed dataset."""
        self.main_topic_model.train_topic_model(self.processed_reviews)
        self.main_topic_model.assign_seeded_labels()

    def predict(self, reviews):
        """Predicts main and subtopics for given reviews."""
        for review in reviews:
            main_labels = self.main_topic_model.predict_review_labels(review, threshold=0.2)
            print(f"Review: {review}")
            for label, main_score in main_labels.items():
                sub_pred = self.subtopic_model.predict_review_subtopics(review, {label: main_score}, threshold=0.1)
                if label in sub_pred:
                    subtopic, sub_score = sub_pred[label]
                    print(f"- Main Topic: {label} (Confidence: {main_score:.2f})")
                    print(f"    -> Subtopic: {subtopic} (Confidence: {sub_score:.2f})")
                else:
                    print(f"- Main Topic: {label} (Confidence: {main_score:.2f})")
            print("-" * 50)

# Execute
model = TopicModel()
model = TopicModel()
model.load_datasets()
model.train()

new_reviews = [
    "The jeans fit perfectly! Love the quality.",
    "I was disappointed with how late my package arrived.",
    "The customer service was extremely helpful and friendly.",
]
model.predict(new_reviews)
