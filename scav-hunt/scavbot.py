import os
import time
from datetime import datetime

from easygopigo3 import EasyGoPiGo3
import picamera

from scaveye import ObjectClassificationModel
from scavnav import NavigationControl

class ScavBot:

    def __init__(self, image_model_dir, image_dir):
        # The bot
        self.gpg = EasyGoPiGo3() 

        # Bot Controls
        self.driver = NavigationControl(self.gpg)

        # Image Model
        self.image_model = ObjectClassificationModel( 
            model_dir = image_model_dir,
            image_dir = image_dir)

        
if __name__ == '__main__':
    bot = ScavBot(image_model_dir='Sample_TFlite_model', image_dir='/home/pi/Pictures')

