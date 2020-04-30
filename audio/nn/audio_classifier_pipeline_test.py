# Testing audio_classifier.py EfficientNet pipeline

from audio_classifier import TfliteSoundClassifier
from pathlib import Path

MODEL_PATH = Path("./models/ensemble/main_model_e50")

urbanClassifier = TfliteSoundClassifier(MODEL_PATH/'model.tflite', MODEL_PATH/'labels.txt')

urbanClassifier.predict(\"dog.wav\")

# should be 'dog-bark'