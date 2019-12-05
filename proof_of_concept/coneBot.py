import coneDetection
import colorDetection
import pdb
import os

DIR_PATH = '/home/pi/Desktop/cone_images/'

class ConeBot:

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.calibrated = False
        
    def calibrate(self):
        pic_name = self.dir_path + 'img.jpg'
        coneDetection.take_picture(pic_name)
        boundaries = colorDetection.get_hsv_boundaries(pic_name)
        self.boundaries = boundaries
        self.calibrated = True
        return self.boundaries
          
    def findCone(self, boundaries = None, imageSource = 'camera'):
        if boundaries is None:
            boundaries = self.boundaries
        return coneDetection.findCone(colorBounds = boundaries, imageSource = imageSource)


bot = ConeBot(DIR_PATH)
boundaries = bot.calibrate()
print(boundaries)

cone = bot.findCone()
print(cone)
