
import time
import picamera
from datetime import datetime

usb_path = "/home/pi/Desktop/cone_images/"
camera = picamera.PiCamera()

try:
   camera.capture(usb_path + "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S')))
finally:
   camera.close()
