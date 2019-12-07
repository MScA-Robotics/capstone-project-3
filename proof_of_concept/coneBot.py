# Sytem libraries
import pdb
import os
import time

# GoPiGo Moduels
from easygopigo3 import EasyGoPiGo3

# Custom modules
import coneDetection
import colorDetection
import circumNav

DIR_PATH = '/home/pi/Desktop/cone_images/'

class ConeBot:

    def __init__(self, dir_path, params=None):
        self.dir_path = dir_path
        self.calibrated = False
        self.gpg = EasyGoPiGo3()
        if params is None:
            self.rad= 200
            self.h_spd= 400
            self.m_spd= 200
            self.l_spd= 30
        else:
            self.rad = params['rad']
            self.h_spd = params['h_spd']
            self.m_spd = params['m_spd']
            self.l_spd = params['l_spd']
        
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

    def centerCone(self):
        assert(self.calibrated)
        centered = False
        current_degree = 0 
        while not centered:
            time.sleep(.5)
            cone_x = self.findCone()
            if cone_x is False:
                if current_degree > 360:
                    return false
                self.gpg.turn_degrees(45)
                current_degree += 45
            if cone_x > .55:
                self.gpg.turn_degrees(10)
            elif cone_x < .45:
                self.gpg.turn_degrees(-10)
            else:
                centered = True
        return True

    def circumNavigate(self):
        circumNav.circumNav(self.gpg)

    def solve_q1_task(self):
        self.centerCone()
        self.circumNavigate()


bot = ConeBot(DIR_PATH)
boundaries = bot.calibrate()# 
print(boundaries)

cone = bot.findCone()
print(cone)

# bot.solve_q1_task()
