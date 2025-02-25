import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_intergration.emotion_correlation import ThemeEmotionCorrelation
from analysis_intergration.emotion_distribution import EmotionDistribution


class AggregatedInsights:
    """
    Provides aggregated insights across topics, emotions, and sentiment trends.
    """
    
    def __init__(self, theme_emotion_analyzer, emotion_distribution_analyzer, adorescore_calculator):
        self.theme_emotion_analyzer = theme_emotion_analyzer
        self.emotion_distribution_analyzer = emotion_distribution_analyzer
        self.adorescore_calculator = adorescore_calculator
    
    def generate_summary(self):
        """Generates a comprehensive summary of insights across all dimensions."""
        print("\n=== Aggregated Insights ===")
        
        # Theme-Emotion Correlation Summary
        print("\n--- Theme-Emotion Correlation ---")
        self.theme_emotion_analyzer.generate_summary()
        
        # Per-Topic Emotional Distribution Summary
        print("\n--- Per-Topic Emotional Distribution ---")
        self.emotion_distribution_analyzer.generate_summary()
        
        # Adorescore Trend Analysis
        print("\n--- Sentiment Trends Over Time ---")
        trend_data = self.adorescore_calculator.compute_trends()
        avg_adorescore = trend_data['adorescore'].mean()
        trend_direction = "increasing" if avg_adorescore > 0 else "decreasing"
        print(f"Average Adorescore: {avg_adorescore:.2f} ({trend_direction} trend)")
        
        # Actionable Insights
        print("\n--- Key Takeaways ---")
        high_neg_emotions = self.theme_emotion_analyzer.compute_correlation_metrics()
        negative_topics = high_neg_emotions[high_neg_emotions['avg_intensity'] < 0].groupby("topic").mean()
        
        if not negative_topics.empty:
            print("\nTopics with highest negative sentiment:")
            print(negative_topics[['avg_intensity']].sort_values(by='avg_intensity'))
            print("\nConsider addressing these areas to improve sentiment.")
        else:
            print("\nNo major negative sentiment detected across topics.")
    
    def plot_insights(self):
        """Plots emotion-topic correlation and sentiment trends over time."""
        print("\nGenerating visual insights...")
        
        # Plot emotion-topic heatmap
        self.theme_emotion_analyzer.plot_correlation_heatmap()
        
        # Plot per-topic emotional distribution
        self.emotion_distribution_analyzer.plot_emotion_distribution()
        
        # Plot Adorescore trends
        self.adorescore_calculator.plot_trends()

# Example Usage:
sample_review = "The delivery was very late! the product is fit is nice."

emotions = emotion_detector.predict(sample_review)[:3]
topics = topic_model.main_topic_model.predict_review_labels(sample_review, threshold=0.2)
subtopics = topic_model.subtopic_model.predict_review_subtopics(sample_review, topics, threshold=0.1)
correlation_analyzer = ThemeEmotionCorrelation(emotions, topics, subtopics)
emotion_distribution = EmotionDistribution(emotions, topics, subtopics)

insights = AggregatedInsights(correlation_analyzer, emotion_distribution, adorescore_calculator)
insights.generate_summary()
insights.plot_insights()
