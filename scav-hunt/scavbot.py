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
        self.servo = self.gpg.init_servo("SERVO1")
        self.servo.rotate_servo(100)
        
        self.params = params
        self.boundaries = boundaries
        self.image_dir = image_dir

        # Image Model
        self.image_model = ObjectClassificationModel( 
            model_dir = image_model_dir,
            image_dir = image_dir)
        
    def find_cone(self, color):
        bounds = self.boundaries[color]
        return detect.findCone(bounds)

    def center_cone(self, color):
        centered = False
        current_degree = 0 
        while not centered:
            time.sleep(.5)
            cone_x = self.find_cone(color)
            if cone_x is False:
                if current_degree > 360:
                    return false
                self.gpg.turn_degrees(-20)
                current_degree += -20
            if cone_x > .6:
                self.gpg.turn_degrees(10)
            elif cone_x < .4:
                self.gpg.turn_degrees(-10)
            else:
                centered = True
        return True
        
    def drive_to_cone(self):
        # Drive to cone at full bore
        self.gpg.set_speed(self.params['h_spd'])
        ob_dist = self.dist_sensor.read_mm()
        while ob_dist >= self.params['cone_dist']:
            self.gpg.forward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))
        self.gpg.stop()

        # Back away to the exact distance at a slower speed
        self.gpg.set_speed(self.params['l_spd'])
        while ob_dist < self.params['cone_dist']:
            self.gpg.backward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))  
        self.gpg.stop()
        print("MADE IT!")

    def circum_navigate(self, color):
        # Set the speed to medium speed
        self.gpg.set_speed(self.params['m_spd'])
        print("I will now cicle the cone at {} mm ".format(self.params['radius']))

        # Circumscibe a circle around the cone
        # rotate gpg 90 degrees to prep for the orbit
        self.gpg.turn_degrees(-90)
        self.servo.rotate_servo(30)

        # Complete the orbit
        radius = 2*self.params['radius']/10
        self.orbit_and_take_picture(20, radius, color)
        self.orbit_and_take_picture(100, radius, color)
        self.orbit_and_take_picture(110, radius, color)
        self.gpg.orbit(40, radius)

        # Rotate back to facing the cone
        self.gpg.turn_degrees(90)
        ob_dist = self.dist_sensor.read_mm()
        print("The cone is now at: {} mm ".format(ob_dist))

        # Return to a base position
        print("That was fun... I go home now") 
        self.gpg.drive_cm(-20,True)

    def orbit_and_take_picture(self, degrees, radius, color):
        self.gpg.orbit(degrees, radius)
        picture_path = os.path.join(self.image_dir, color)
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)      
        take_picture(picture_path)

    def main(self):
        self.center_cone('orange')
        self.drive_to_cone()
        self.circum_navigate('orange')

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

