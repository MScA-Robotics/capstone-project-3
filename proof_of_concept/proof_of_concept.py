from coneDetection import *
from colorDetection import *
import pdb
import os

DIR_PATH = '/home/pi/Desktop/cone_images/'

class ConeBot

    def __init__(self, dir_path):
        self.dir_path = dir_path
        
    def calibrate(self):
        pic_name = dir_path + 'img.jpg'
        take_picture(pic_name)
        boundaries = get_hsv_boundaries(pic_name)
        self.boundaries = boundaries
        


bot = ConeBot()DIfrom coneDetection import *
from colorDetection import *
import pdb
import os

DIR_PATH = '/home/pi/Desktop/cone_images/'

class ConeBot

    def __init__(self, dir_path):
        self.dir_path = dir_path
        
    def calibrate(self):
        pic_name = dir_path + 'img.jpg'
        take_picture(pic_name)
        boundaries = get_hsv_boundaries(pic_name)
        self.boundaries = boundaries
        
from coneDetection import *
from colorDetection import *
import pdb
import os

DIR_PATH = '/home/pi/Desktop/cone_images/'

class ConeBot

    def __init__(self, dir_path):
        self.dir_path = dir_path
        
    def calibrate(self):
        pic_name = dir_path + 'img.jpg'
        take_picture(pic_name)
        boundaries = get_hsv_boundaries(pic_name)
        self.boundaries = boundaries
        

cone = findCone(colorBounds = boundaries, imageSource='camera')
print(cone)
