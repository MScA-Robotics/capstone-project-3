
import time
import picamera
from datetime import datetime

usb_path = "/media/pi/PORTFOLIO/"
camera = picamera.PiCamera()

try:
   camera.start_preview()
   time.sleep(2)
   camera.capture(usb_path + "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S')))
   camera.stop_preview()
finally:
   camera.close()
