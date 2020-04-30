# Model directory for audio classifier

## Requirements

In order for the `scavear.py` and `audio_classifier_test.py` to function correctly, a tflite model and a text file of it's associated class labels must be present in this folder.

* `model.tflite`: compiled tflite model
* `labels.txt`: text file with `n` class labels for the `model.tflite`, ordered by the classification indicies of the model output, `[0-(n-1)]`.  Text file should only contain `n` class labels on `n` lines with.