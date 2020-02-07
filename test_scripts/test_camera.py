
import time
import picamera
from datetime import datetime
from easygopigo3 import EasyGoPiGo3
from time import sleep


usb_path = "/home/pi/Desktop/pics/"


def take_picture(path = usb_path):
    camera = picamera.PiCamera()
    try:
       camera.capture(path + "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S')))
    finally:
       camera.close()
       
take_picture()