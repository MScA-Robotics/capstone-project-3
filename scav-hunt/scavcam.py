import os
import picamera
import cv2
import numpy as np
import importlib.util

# If using TPU, need to load a different library
from tensorflow.lite.python.interpreter import Interpreter

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

