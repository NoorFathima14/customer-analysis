from typing import List, Dict
from config import EmotionConfig

class EmotionPostprocessor:
    @staticmethod
    def get_activation_level(score: float) -> str:
        if score >= EmotionConfig.HIGH_ACTIVATION:
            return "High"
        elif score >= EmotionConfig.MEDIUM_ACTIVATION:
            return "Medium"
        return "Low"

    @staticmethod
    def map_to_plutchik(emotions: List[str]) -> Dict[str, str]:
        # Map GoEmotions labels to Plutchik's wheel
        PLUTCHIK_MAPPING = {
            'admiration': 'trust',
            'amusement': 'joy',
            'anger': 'anger',
            'annoyance': 'anger',
            'approval': 'trust',
            'caring': 'trust',
            'confusion': 'surprise',
            'curiosity': 'anticipation',
            'desire': 'anticipation',
            'disappointment': 'sadness',
            'disapproval': 'disgust',
            'disgust': 'disgust',
            'embarrassment': 'fear',
            'excitement': 'joy',
            'fear': 'fear',
            'gratitude': 'trust',
            'grief': 'sadness',
            'joy': 'joy',
            'love': 'joy+trust',
            'nervousness': 'fear',
            'optimism': 'anticipation',
            'pride': 'joy',
            'realization': 'surprise',
            'relief': 'joy',
            'remorse': 'sadness',
            'sadness': 'sadness',
            'surprise': 'surprise',
        }
        return {e: PLUTCHIK_MAPPING.get(e.lower(), e) for e in emotions}