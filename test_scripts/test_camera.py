import time
import picamera
from datetime import datetime

usb_path = "/media/pi/PORTFOLIO"
camera = picamera.PiCamera()

try:
   camera.start_preview()
   time.sleep(5)
   camera.capture(usb_path + "image_{0}.jpg".format(datetime.now()))
   camera.stop_preview()
finally:
   camera.close()
