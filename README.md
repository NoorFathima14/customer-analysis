# Emotion-Topic Analysis System

## Overview

The Emotion-Topic Analysis System integrates emotion detection, topic classification, and sentiment scoring to analyze user reviews. It leverages pre-trained transformer models for emotion detection and BERTopic for topic extraction, computing an aggregated sentiment score called Adorescore. The system supports multilingual reviews, making it adaptable for global applications across various industries.

## Installation

To set up the environment, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

##  Train & Save Model

Run `topic_analysis/synthetic_data_gen.py` to generate synthetic data `synthetic_delivery_reviews.txt`

Run `emotion_detection/usage.py` and `topic_analysis/topic_model.py` to train and save model

## Load Model, Prediction and Analysis

Run `emotion_analysis_sys.py`


