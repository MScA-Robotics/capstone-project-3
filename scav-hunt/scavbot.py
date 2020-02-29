import os
import time
from datetime import datetime

from easygopigo3 import EasyGoPiGo3
import picamera

from scaveye import ObjectClassificationModel, take_picture
from coneutils import detect

class ScavBot:

    def __init__(self, image_model_dir, image_dir, params, boundaries):
        self.gpg = EasyGoPiGo3()
        self.dist_sensor = self.gpg.init_distance_sensor()
        self.params = params
        self.boundaries = boundaries

        # Image Model
        self.image_model = ObjectClassificationModel( 
            model_dir = image_model_dir,
            image_dir = image_dir)
        
    def find_cone(self, color):
        bounds = self.boundaries[color]
        return detect.findCone(bounds)
        
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
        self.gpg.turn_degrees(-90)

        # Complete the orbit
        self.gpg.orbit(180, (2*self.params['radius']/10))

        # Rotate back to facing the cone
        self.gpg.turn_degrees(90)
        ob_dist = self.dist_sensor.read_mm()
        print("The cone is now at: {} mm ".format(ob_dist))

        # Return to a base position
        print("That was fun... I go home now") 
        self.gpg.drive_cm(-20,True)

if __name__ == '__main__':
    import config
    from coneutils import calibrate

    boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')

    bot = ScavBot(
        image_model_dir='Sample_TFLite_model', 
        image_dir='/home/pi/Pictures',
        params=config.params,
        boundaries = boundaries_dict
    )

