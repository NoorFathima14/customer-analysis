class EmotionConfig:
    # Model parameters
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    NUM_LABELS = 10  # Number of selected emotions
    DROPOUT = 0.1
    
    # Training params
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Emotion activation thresholds
    HIGH_ACTIVATION = 0.7
    MEDIUM_ACTIVATION = 0.4
    LOW_ACTIVATION = 0.1