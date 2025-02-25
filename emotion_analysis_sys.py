from emotion_detection.detector import EmotionDetector
from topic_analysis.topic_model import TopicModel
from adorescore.adorescore import AdorescoreCalculator
from analysis_intergration.emotion_correlation import ThemeEmotionCorrelation
from analysis_intergration.emotion_distribution import EmotionDistribution
from analysis_intergration.aggregated_insights import AggregatedInsights

# Load model
detector = EmotionDetector.load_model("emotion_model.pth")

# Predict
review = "The product was awesome, but the delivery was terrible"
emotion_results = detector.predict(review)[:3]

topic_model = TopicModel.load_model()
topic_results = topic_model.predict(review)

topics = topic_model.main_topic_model.predict_review_labels(review, threshold=0.2)
subtopics = topic_model.subtopic_model.predict_review_subtopics(review, topics, threshold=0.1)

adorescore_calculator = AdorescoreCalculator()
adore_results = adorescore_calculator.analyze_review(review, emotion_results, topics)

def format_emotions(emotion_list):
    """
    Formats a list of detected emotions into primary and secondary emotions.

    :param emotion_list: List of emotion dictionaries with 'emotion', 'intensity', and 'activation'.
    :return: Dictionary with 'primary' and 'secondary' emotions formatted.
    """
    if not emotion_list:
        return {"primary": None, "secondary": []}  

    # Extract primary emotion (highest intensity)
    primary_emotion = {
        "emotion": emotion_list[0]["emotion"].capitalize(),
        "activation": emotion_list[0]["activation"],
        "intensity": round(emotion_list[0]["intensity"], 2)
    }

    # Extract secondary emotion (if exsists)
    secondary_emotions = [
        {
            "emotion": emo["emotion"].capitalize(),
            "activation": emo["activation"],
            "intensity": round(emo["intensity"], 2)
        }
        for emo in emotion_list[1:]  # Skip first emotion
    ]

    return {"primary": primary_emotion, "secondary": secondary_emotions}

def format_output(emotion_list, topic_list, adorescore_list):
    return {
        "emotions": format_emotions(emotion_list),
        "topics": topic_list,
        "adorescore": adorescore_list
    }

print(format_output(emotion_results,topic_results,adore_results))

correlation_analyzer = ThemeEmotionCorrelation(emotion_results, topics, subtopics)
emotion_distribution = EmotionDistribution(emotion_results, topics, subtopics)
insights = AggregatedInsights(correlation_analyzer, emotion_distribution, adorescore_calculator)

insights.generate_summary()
insights.plot_insights()


