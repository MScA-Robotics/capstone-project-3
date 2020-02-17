import os
import picamera
import cv2
import numpy as np
import importlib.util

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

def take_picture(path):
    if path is None:
        path = "/home/pi/Pictures"
    camera = picamera.PiCamera()
    try:
       camera.capture(os.path.join(path, "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S'))))
    finally:
       camera.close()

class ObjectClassificationModel:

    def __init__(self, model_dir, image_dir, min_conf_threshold=0.5, use_TPU=False):

        self.model_dir = model_dir
        self.image_dir = image_dir
        self.min_conf_threshold = float(min_conf_threshold)
        self.use_TPU = use_TPU

