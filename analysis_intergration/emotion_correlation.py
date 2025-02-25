import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ThemeEmotionCorrelation:
    """Analyzes correlation between emotions and topics to detect patterns."""
    
    def __init__(self, emotions=None, topics=None, topic_subtopics=None):
        """
        Initializes the emotion-topic correlation data.
        :param emotions: List of detected emotions with intensity.
        :param topics: Dictionary of main topics with relevance scores.
        :param topic_subtopics: Dictionary mapping subtopics to main topics.
        """
        self.emotion_topic_data = []  # Store emotion-topic relationships
        
        # Populate data if provided
        if emotions and topics:
            self.update_correlation(emotions, topics, topic_subtopics)

    def update_correlation(self, emotions, topics, topic_subtopics):
        """
        Updates the emotion-topic mapping with new data.
        """
        new_data = []
        for topic, topic_confidence in topics.items():
            subtopic, _ = topic_subtopics.get(topic, (None, 0))  # Get subtopic if available
            for emotion in emotions:
                new_data.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "emotion": emotion["emotion"],
                    "intensity": emotion["intensity"],
                    "activation": emotion["activation"],
                    "weighted_intensity": emotion["intensity"] * topic_confidence
                })
        
        # Extend existing data instead of overwriting
        self.emotion_topic_data.extend(new_data)
    
    def compute_correlation_metrics(self):
        """Computes average emotional impact per topic."""
        df = pd.DataFrame(self.emotion_topic_data)
        if df.empty:
            print("No data available for correlation analysis.")
            return None

        # Group by topic and emotion, then compute mean intensity
        correlation_df = df.groupby(["topic", "emotion"]).agg(
            avg_intensity=("weighted_intensity", "mean"),
            count=("emotion", "count")
        ).reset_index()

        return correlation_df.sort_values(by=["topic", "avg_intensity"], ascending=[True, False])
    
    def generate_summary(self):
        """Generates a summary of emotion-topic relationships."""
        correlation_df = self.compute_correlation_metrics()
        if correlation_df is None:
            return
        
        print("\n=== Theme-Emotion Correlation Summary ===")
        for topic in correlation_df["topic"].unique():
            print(f"\nTopic: {topic}")
            topic_data = correlation_df[correlation_df["topic"] == topic]
            for _, row in topic_data.iterrows():
                print(f"  - {row['emotion'].capitalize()}: Avg Intensity {row['avg_intensity']:.2f} ({int(row['count'])} occurrences)")
    
    def plot_correlation_heatmap(self):
        """Generates a heatmap of emotion-topic correlation."""
        correlation_df = self.compute_correlation_metrics()
        if correlation_df is None:
            return
        
        pivot_table = correlation_df.pivot(index="topic", columns="emotion", values="avg_intensity").fillna(0)
        plt.figure(figsize=(12, 6))
        plt.title("Emotion-Topic Correlation Heatmap")
        plt.xlabel("Emotions")
        plt.ylabel("Topics")
        plt.imshow(pivot_table, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Avg Intensity")
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        plt.show()


