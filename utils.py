from typing import List, Dict
from config import EmotionConfig

class EmotionPostprocessor:
    @staticmethod
    def get_activation_level(emotion: str) -> str:
      emotion = emotion.lower()
      high_activation = {'ecstasy', 'vigilance','admiration','terror','amazement','grief','loathing','rage'}
      medium_activation = {'joy', 'anticipation', 'trust', 'fear', 'suprise', 'sadness','disgust','anger'}
      low_activation = {'serenity', 'interest','acceptance','apprehension','distraction','pensiveness','boredom','annoyance'}

      if emotion in high_activation:
          return 'High'
      elif emotion in medium_activation:
          return 'Medium'
      elif emotion in low_activation:
          return 'Low'
      else:
          return 'Low'

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