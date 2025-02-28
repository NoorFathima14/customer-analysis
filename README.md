# Emotion-Topic Analysis System

## Overview

The Emotion-Topic Analysis System integrates emotion detection, topic classification, and sentiment scoring to analyze user reviews. It leverages pre-trained transformer models for emotion detection and BERTopic for topic extraction, computing an aggregated sentiment score called Adorescore. The system supports multilingual reviews, making it adaptable for global applications across various industries.

## Installation

To run the Python application, first create a `venv`.

```bash
# Windows
py -m venv venv

# Linux/Mac
python3 -m venv venv
```

Move to the directory and activate the virtual environment.

```bash
# Windows
venv\Scripts\Activate

# Linux/Mac
source venv/bin/activate
```

To set up the environment, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

##  Train & Save Model

Run `topic_analysis/synthetic_data_gen.py` to generate synthetic data `synthetic_delivery_reviews.txt`

Run `emotion_detection/usage.py` and `topic_analysis/topic_model.py` to train and save model

## Load Model, Prediction and Analysis

Run `emotion_analysis_sys.py`

## Running the server

Run `uvicorn main:app`. This will start a FastAPI server at `http://localhost:8000/`.

`POST` method at `/predict` takes the following request body: 

```json
{
    "text": "The product was awesome, but the delivery was terrible",
    "language": "english"
}
```

The language must be specified in full and lowercase. Two letter language codes are not supported.

Swagger documentation is available at `http://localhost:8000/docs`.