import os
import time
from datetime import datetime

from easygopigo3 import EasyGoPiGo3
import picamera

import scavcam

class ScavBot:

    def __init__(self, image_model_dir, image_dir):
        self.image_model = scavcam.ObjectClassificationModel(
            model_dir = image_model_dir,
            image_dir = image_dir)
