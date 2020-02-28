import os
import time
from datetime import datetime

from easygopigo3 import EasyGoPiGo3
import picamera

from scaveye import ObjectClassificationModel
from scavnav import NavigationControl


class ScavBot:

    def __init__(self, image_model_dir, image_dir, params):
        self.gpg = EasyGoPiGo3()
        self.dist_sensor = gpg.init_distance_sensor()
        self.params = param

        # Image Model
        self.image_model = ObjectClassificationModel( 
            model_dir = image_model_dir,
            image_dir = image_dir)
        
    def drive_to_cone(self):
        # Drive to cone at full bore
        self.gpg.set_speed(self.params['h_spd'])
        ob_dist = self.dist_sensor.read_mm()
        while ob_dist >= self.params['radius']:
            self.gpg.forward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))
        self.gpg.stop()

        # Back away to the exact distance at a slower speed
        self.gpg.set_speed(self.params['l_spd'])
        while ob_dist < rad:
            self.gpg.backward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))  
        self.gpg.stop()
        print("MADE IT!")

    def circum_navigate(self):
        # Set the speed to medium speed
        self.gpg.set_speed(self.params['m_spd'])
        print("I will now cicle the cone at {} mm ".format(self.params['radius']))

        # Circumscibe a circle around the cone
        # rotate gpg 90 degrees to prep for the orbit
        gpg.turn_degrees(-90)

        # Complete the orbit
        gpg.orbit(180, (2*self.params['radius']/10))

        # Rotate back to facing the cone
        self.gpg.turn_degrees(90)
        ob_dist = self.dist_sensor.read_mm()
        print("The cone is now at: {} mm ".format(ob_dist))

        # Return to a base position
        print("That was fun... I go home now") 
        gpg.drive_cm(-20,True)

if __name__ == '__main__':
import config

    bot = ScavBot(
        image_model_dir='Sample_TFlite_model', 
        image_dir='/home/pi/Pictures',
        params=config.params
    )

