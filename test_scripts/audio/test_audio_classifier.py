# Testing audio_classifier.py EfficientNet pipeline

from audio_classifier import TfliteSoundClassifier
from pathlib import Path

MODEL_PATH = Path("../../models/audio_classifier")

urbanClassifier = TfliteSoundClassifier(MODEL_PATH/'model.tflite', MODEL_PATH/'labels.txt')

print(urbanClassifier.predict("Dog.wav"))

# dog-bark
