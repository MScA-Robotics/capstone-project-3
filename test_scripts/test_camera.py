import time
import picamera
from datetime import datetime
from easygopigo3 import EasyGoPiGo3
from time import sleep
import os
import argparse


def take_picture(path):
    if path is None:
        path = "/home/pi/Pictures"
    camera = picamera.PiCamera()
    try:
       camera.capture(os.path.join(path, "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S'))))
    finally:
       camera.close()
       

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', default=None)
    args = parser.parse_args()

    IMAGE_PATH = args.imagedir 

    take_picture(IMAGE_PATH)

