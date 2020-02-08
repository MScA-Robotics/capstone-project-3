import time
import picamera
from datetime import datetime
from easygopigo3 import EasyGoPiGo3
from time import sleep


DEFAULT_CAMERA_PATH = "/home/pi/Pictures/"


def take_picture(path = DEFAULT_CAMERA_PATH):
    camera = picamera.PiCamera()
    try:
       camera.capture(path + "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S')))
    finally:
       camera.close()
       

if __name__ == '__main__':

    take_picture()