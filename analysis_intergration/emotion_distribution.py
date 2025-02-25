import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EmotionDistribution:
    """Analyzes per-topic emotional distribution for better insights."""
    
    def __init__(self, emotions=None, topics=None, topic_subtopics=None):
        """
        Initializes the emotion-topic distribution.
        :param emotions: List of detected emotions with intensity.
        :param topics: Dictionary of main topics with relevance scores.
        :param topic_subtopics: Dictionary mapping subtopics to main topics.
        """
        self.emotion_topic_data = []  # Stores emotion-topic relationships
        self.emotion_distribution = None
        # Populate data if inputs are provided
        if emotions and topics:
            self.update_distribution(emotions, topics, topic_subtopics)

    def update_distribution(self, emotions, topics, topic_subtopics):
        """
        Updates the dataset with emotions mapped to topics and subtopics.
        """
        new_data = []
        for topic, topic_confidence in topics.items():
            subtopic, _ = topic_subtopics.get(topic, (None, 0))  # Get subtopic if available
            for emotion in emotions:
                new_data.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "emotion": emotion["emotion"],
                    "intensity": emotion["intensity"] * topic_confidence  # Weight emotion by topic confidence
                })
        
        # Extend the existing data instead of overwriting
        self.emotion_topic_data.extend(new_data)
    
    def compute_emotion_distribution(self):
        """Computes per-topic emotion proportions."""
        df = pd.DataFrame(self.emotion_topic_data)
        if df.empty:
            print("No data available for emotion distribution analysis.")
            return None
        
        # Normalize emotion intensities within each topic
        emotion_distribution = df.groupby(["topic", "emotion"]).agg(total_intensity=("intensity", "sum")).reset_index()
        topic_totals = emotion_distribution.groupby("topic")["total_intensity"].transform("sum")
        emotion_distribution["proportion"] = emotion_distribution["total_intensity"] / topic_totals
        
        self.emotion_distribution = (
                emotion_distribution.sort_values(by=["topic", "proportion"], ascending=[True, False])
            )
        return self.emotion_distribution

    def generate_summary(self):
        """Summarizes per-topic emotional distributions."""
        if self.emotion_distribution is None or self.emotion_distribution.empty:
            self.compute_emotion_distribution()  # Compute it if not already available
            if self.emotion_distribution is None or self.emotion_distribution.empty:  # If still None, exit
                print("No emotional distribution data available.")
                return
        
        print("\n=== Per-Topic Emotional Distribution Summary ===")
        for topic in self.emotion_distribution["topic"].unique():
            print(f"\nTopic: {topic}")
            topic_data = self.emotion_distribution[self.emotion_distribution["topic"] == topic]
            for _, row in topic_data.iterrows():
                print(f"  - {row['emotion'].capitalize()}: {row['proportion']:.2%}")
    
    def plot_emotion_distribution(self):
        """Generates a stacked bar chart of emotion proportions per topic."""
        self.compute_emotion_distribution()
        if self.emotion_distribution is None:
            return
        print(self.emotion_distribution)
        pivot_table = self.emotion_distribution.pivot(index="topic", columns="emotion", values="proportion").fillna(0)
        pivot_table.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
        plt.title("Per-Topic Emotional Distribution")
        plt.xlabel("Topics")
        plt.ylabel("Proportion of Emotions")
        plt.legend(title="Emotions", bbox_to_anchor=(1, 1))
        plt.show()