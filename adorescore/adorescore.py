import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AdorescoreCalculator:
    """Computes an Adorescore based on emotion detection, activation, and topic relevance."""

    def __init__(self,log_file="adorescore_log.csv"):
        self.log_file=log_file
        # Base emotion weight mapping 
        self.base_emotion_weights = {
            "joy": 1, "excitement": 1, "gratitude": 1, "approval": 1, 
            "anger": -1, "disappointment": -1, "disgust": -1, "sadness": -1, 
            "confusion": 0, "surprise": 0 
        }

        # Activation multipliers
        self.activation_weights = {
            "low": 0.75,      
            "medium": 1.0,   
            "high": 1.25     
        }

    def filter_significant_emotions(self, emotions, top_n=3, intensity_threshold=0.05):
        """
        Filters the top N most significant emotions based on intensity.
        Ignores emotions below the intensity threshold.
        """
        filtered = [e for e in emotions if e["intensity"] >= intensity_threshold]
        sorted_emotions = sorted(filtered, key=lambda x: x["intensity"], reverse=True)
        return sorted_emotions[:top_n]  

    def compute_adorescore(self, emotions, topics):
        """Computes Adorescore using only the most significant emotions."""
        
        significant_emotions = self.filter_significant_emotions(emotions)
        if not significant_emotions:  
            return 0, {topic: 0 for topic in topics}

        total_weight = sum(topics.values()) or 1  # Avoid division by zero
        topic_breakdown = {}

        # Normalize emotion intensities
        total_emotion_intensity = sum(e["intensity"] for e in significant_emotions) or 1

        for topic, confidence in topics.items():
            topic_score = 0

            for emotion in significant_emotions:
                name, intensity, activation = emotion["emotion"], emotion["intensity"], emotion["activation"]
                weight = self.base_emotion_weights.get(name, 0)
                activation_multiplier = self.activation_weights.get(activation, 1.0)

                # Normalize intensity and compute weighted emotion impact
                normalized_intensity = intensity / total_emotion_intensity
                emotion_score = weight * normalized_intensity * activation_multiplier * 100

                adjusted_confidence = 0.5 + 0.5 * confidence  # Boost values closer to 1
                topic_score += emotion_score * adjusted_confidence

            # Cap topic score within -100 to 100
            topic_breakdown[topic] = max(min(topic_score, 100), -100)

        # Compute weighted sum for final Adorescore
        adorescore = sum(topic_breakdown[t] * (topics[t] / total_weight) for t in topic_breakdown)
        adorescore = max(min(adorescore, 100), -100)
        self.log_adorescore(datetime.now(), adorescore)
        return adorescore, topic_breakdown

    def log_adorescore(self, timestamp, adorescore):
        """Logs Adorescore with timestamp for trend analysis."""
        df = pd.DataFrame([[timestamp, adorescore]], columns=["timestamp", "adorescore"])
        df.to_csv(self.log_file, mode='a', header=not pd.io.common.file_exists(self.log_file), index=False)
    
    def compute_trends(self, window='7D'):
        """Computes Adorescore trends over time."""
        try:
            df = pd.read_csv(self.log_file, parse_dates=["timestamp"])
            if df.empty:
                print("No data available for trends.")
                return pd.DataFrame()  # to avoid NoneType error
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  
            df.dropna(subset=["timestamp"], inplace=True) 
            df.set_index("timestamp", inplace=True)

            df["adorescore"] = pd.to_numeric(df["adorescore"], errors="coerce")

            df["rolling_avg"] = df["adorescore"].rolling(window="7D", min_periods=1).mean()
            
            return df

        except Exception as e:
            print(f"Error computing trends: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if an error occurs

    
    def plot_trends(self, window='7D'):
        """Plots Adorescore trends over time."""
        df = self.compute_trends(window)
        if df.empty:
            print("No data to plot.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["adorescore"], label="Daily Adorescore", alpha=0.5, marker="o")
        plt.plot(df.index, df["rolling_avg"], label=f"{window} Rolling Avg", linewidth=2, linestyle="--")
        
        plt.xlabel("Date")
        plt.ylabel("Adorescore")
        plt.title("Adorescore Sentiment Trends Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_review(self, review, emotions, topics):
        """
        Predicts emotions, topics, and computes Adorescore for a given review.

        :param review: The input review text
        :param emotion_detector: Pretrained emotion detection model
        :param topic_model: Pretrained topic analysis model
        :return: Adorescore, topic breakdown, detected emotions
        """

        adorescore, topic_scores = self.compute_adorescore(emotions, topics)

        result = {
          "adorescore": {
              "overall": round(adorescore),
              "breakdown": {topic: round(score) for topic, score in topic_scores.items()}
          }
        }

        return result

