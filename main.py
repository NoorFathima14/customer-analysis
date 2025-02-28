import sys
import emotion_detection.config

# Override the 'config' module reference
sys.modules["config"] = emotion_detection.config

from emotion_detection.detector import EmotionDetector
from topic_analysis.topic_model import TopicModel
from adorescore.adorescore import AdorescoreCalculator
from translate import Translator

from fastapi import FastAPI
from pydantic import BaseModel

# model loading

detector = EmotionDetector.load_model("emotion_model.pth")
topic_model = TopicModel.load_model()
adorescore_calculator = AdorescoreCalculator()

app = FastAPI()

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


@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Analysis Model API"}

class Review(BaseModel):    
    text: str
    language: str

@app.post("/predict")
async def predict(request: Review):
    language = request.language
    text = request.text
    translator = Translator()

    if language != "english":
        text = await translator.translate(prompt=text, source_lang=language)
        print(text)
    
    emotion_results = detector.predict(text)[:3]
    topic_results = topic_model.predict(text)

    topics = topic_model.main_topic_model.predict_review_labels(text, threshold=0.2)
    subtopics = topic_model.subtopic_model.predict_review_subtopics(text, topics, threshold=0.1)

    adore_results = adorescore_calculator.analyze_review(text, emotion_results, topics)

    output = format_output(emotion_results, topic_results, adore_results)
    
    return output

