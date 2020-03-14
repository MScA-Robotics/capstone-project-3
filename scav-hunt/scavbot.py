import os
import time
from datetime import datetime, date

from easygopigo3 import EasyGoPiGo3
import picamera

from scaveye import ObjectClassificationModel, take_picture
from coneutils import detect


class ScavBot:

    def __init__(self, image_model_dir, image_dir, params, boundaries, log_dir='logs'):
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

        # Log File
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = os.path.join(log_dir, 'log_'+ str(date.today())+'.txt')

    def log(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write('\n')
        
    def find_cone(self, color):
        bounds = self.boundaries[color]
        return detect.findCone(bounds)

    def center_cone(self, color):
        print('Finding {} cone'.format(color))
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
        print('Found {} cone!'.format(color))
        return True
        
    def drive_to_cone(self, color):
        self.center_cone(color)
        print('Driving to {} cone'.format(color))
        # Drive to cone at full bore
        self.gpg.set_speed(self.params['h_spd'])
        ob_dist = self.dist_sensor.read_mm()
        t0 = time.time()
        while ob_dist >= self.params['cone_dist']:
            self.gpg.forward()
            ob_dist = self.dist_sensor.read_mm()
            # Every three seconds, recenter the cone
            if time.time() - t0 > 3:
                self.gpg.stop()
                print('Recentering')
                self.center_cone(color)
                t0 = time.time()
        self.gpg.stop()
        print("Distance Sensor Reading: {} mm ".format(ob_dist))

        # Back away to the exact distance at a slower speed
        self.gpg.set_speed(self.params['l_spd'])
        while ob_dist < self.params['radius']:
            self.gpg.backward()
            ob_dist = self.dist_sensor.read_mm()
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
        self.orbit_and_take_picture(40, radius, color)
        self.orbit_and_take_picture(100, radius, color, turn_90=True)
        self.orbit_and_take_picture(110, radius, color)
        self.orbit_and_take_picture(40, radius, color)
        
        self.servo.rotate_servo(100)

    def orbit_and_take_picture(self, degrees, radius, color, turn_90=False):
        self.gpg.orbit(degrees, radius)
        picture_path = os.path.join(self.image_dir, color)
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)

        if turn_90:
            self.servo.rotate_servo(100)
            self.gpg.turn_degrees(90)
            self.gpg.drive_cm(-20)
            take_picture(picture_path)
            self.gpg.drive_cm(20)
            self.gpg.turn_degrees(-90)
            self.servo.rotate_servo(30)
        else:
            take_picture(picture_path)

    def classify_and_log(self, color):
        image_dir = os.path.join(self.image_model.image_dir, color)
        classes, probs, objects = self.image_model.classify(image_dir)
        txt = ','.join([str(datetime.now()), color, str(objects)])
        self.log(txt)
        print('Logged: ', txt)
        return txt

    def main(self, color):
        self.center_cone(color)
        self.drive_to_cone(color)
        self.circum_navigate(color)
        self.classify_and_log(color)


if __name__ == '__main__':
    import config
    from coneutils import calibrate

    from scavear import Listener

    boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')

    bot = ScavBot(
        image_model_dir='Sample_TFLite_model', 
        image_dir='/home/pi/Pictures/scav_hunt',
        params=config.params,
        boundaries = boundaries_dict
    )

    l = Listener()
