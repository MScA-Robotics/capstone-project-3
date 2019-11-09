import time
import picamera

usb_path = "/media/pi/PORTFOLIO"
camera = picamera.PiCamera()

try:
   camera.start_preview()
   time.sleep(5)
   camera.capture(usb_path)
   camera.stop_preview()
finally:
   camera.close()
