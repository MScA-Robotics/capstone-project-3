import os
import time
from datetime import datetime, date

from easygopigo3 import EasyGoPiGo3
import picamera
import glob

from scaveye import ObjectClassificationModel, ConeClassificationModel,take_picture, record_video
from coneutils import detect

from threading import Thread

class ScavBot:
    def __init__(self, image_model_dir, cone_model_dir, image_dir, cone_image_dir,params, boundaries,logger, log_dir='logs'):
        self.gpg = EasyGoPiGo3()
        self.dist_sensor = self.gpg.init_distance_sensor(port="AD1")
        self.servo = self.gpg.init_servo("SERVO1")
        self.servo.rotate_servo(100)
        
        self.params = params
        self.boundaries = boundaries
        self.image_dir = image_dir

        # Image Model
        self.image_model = ObjectClassificationModel( 
            model_dir = image_model_dir,
            image_dir = image_dir,
            min_conf_threshold=0.3,
            use_TPU=True)
        
        # Cone Detection Model
        self.cone_detection_model = ConeClassificationModel( 
            model_dir = cone_model_dir,
            image_dir = cone_image_dir,
            graph_name='cone_detect.tflite',
            min_conf_threshold=0.3, 
            use_TPU=True)

        # Log File
        self.logger = logger
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
    
    def find_cone_new(self, color,cones):
        #bounds = self.boundaries[color]
        return detect.findcone_mod(color,cones)

    def center_cone_with_tfmodel(self, color):
        print('Finding {} cone'.format(color))
        color_dict = {'blue':0,
                      'green':1,
                      'orange':2,
                      'purple':3,
                      'red':4,
                      'yellow':5}
        conecolor_index = color_dict[color]
        centered = False
        current_degree = 0 
        cone_image_path = '/home/pi/Pictures/Cones/'+color+'/'
        backup_image_path = '/home/pi/Pictures/Cones/backup/'+color+'/'
        while not centered:
            time.sleep(.5)
            #cone_x = self.find_cone(color)
            take_picture(cone_image_path)
            cones = self.cone_detection_model.classify(cone_image_path)
            # if cone_x is False:
            #     if current_degree > 360:
            #         return false
            #     self.gpg.turn_degrees(-20)
            #     current_degree += -20
            #print(cones)

            cone_x = self.find_cone_new(conecolor_index,cones)
            print('Cone is at : ',cone_x)
            if cone_x is False:
                if current_degree > 360:
                    return false
                self.gpg.turn_degrees(-20)
                current_degree += -20
            if cone_x > .65:
                self.gpg.turn_degrees(10)
            elif cone_x <.35:
                self.gpg.turn_degrees(-10)
            else:
                centered = True
            files = glob.glob(cone_image_path+'*')
            filename = os.path.basename(files[0])

            print('destination:{}'.format(backup_image_path+filename))
            os.rename(files[0], backup_image_path+filename)
            #os.remove(files[0])
        print('Found {} cone!'.format(color))
        return True
        
    def drive_to_cone(self, color):
        self.center_cone_with_tfmodel(color)
        print('Driving to {} cone'.format(color))
        # Drive to cone at full bore
        self.gpg.set_speed(self.params['h_spd'])
        ob_dist = self.dist_sensor.read_mm()
        t0 = time.time()
        while ob_dist >= self.params['cone_dist'] or ob_dist ==0:#sometimes distance sensor gives 0mm reading erroneously
            self.gpg.forward()
            ob_dist = self.dist_sensor.read_mm()
            print('Distance to cone:',ob_dist)
            # Every three seconds, recenter the cone
            if time.time() - t0 > 3:
                self.gpg.set_speed(self.params['m_spd'])
                self.gpg.stop()
                print('Recentering')
                self.center_cone_with_tfmodel(color)
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
        #print("I will now cicle the cone at {} mm ".format(self.params['radius']))

        # Circumscibe a circle around the cone
        # rotate gpg 90 degrees to prep for the orbit
        self.gpg.turn_degrees(-90)
        
        #radius = 2*self.params['radius']/10

        if color == 'red':
            radius = 40
            print('orbiting red cone at {}'.format(radius))
            print("I will now cicle the cone at {} mm ".format(radius))
            self.gpg.set_speed(self.params['h_spd']) #high speed for red cone
            self.gpg.orbit(300, radius)
        elif color == 'green':
            radius = 50 
            print("I will now cicle the cone at {} mm ".format(radius))
            self.orbit_and_take_picture(150, radius, color, turn_90=True)
            self.orbit_and_take_picture(100, radius, color)
        elif color == 'purple':
            radius = 60
            print("I will now cicle the cone at {} mm ".format(radius))
            self.orbit_and_take_picture(150, radius, color, turn_90=True)
            self.orbit_and_take_picture(100, radius, color)
        elif color =='yellow':
            radius = 70
            print("I will now cicle the cone at {} mm ".format(radius))
            self.orbit_and_take_picture(150, radius, color, turn_90=True)
            self.orbit_and_take_picture(100, radius, color)
            # Complete the orbit
            #self.servo.rotate_servo(50)
            # if color=='purple':
            #     #bigger radius because of large object
            #     radius = 70
            # else:
            #     radius = 50

            # #self.orbit_and_take_picture(40, radius, color)
            # self.orbit_and_take_picture(150, radius, color, turn_90=True)
            # self.orbit_and_take_picture(100, radius, color)
            # #self.orbit_and_take_picture(40, radius, color)
        
        
        #self.servo.rotate_servo(90)

    def orbit_and_take_picture(self, degrees, radius, color, turn_90=False):
        self.gpg.orbit(degrees, radius)
        picture_path = os.path.join(self.image_dir, color)
        video_path = '/home/pi/Videos/'
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)

        if not os.path.exists(video_path):
            os.makedirs(video_path)

        if turn_90:
            #self.servo.rotate_servo(100)
            self.gpg.turn_degrees(90)
            drive_cm = 10
            self.gpg.drive_cm(-drive_cm)
            #take_picture(picture_path)
            record_video(video_path,cone_color=color,duration=3)
            # cone_object = self.image_model.classify_video(video_path+color)
            # print('!!!!!!!!!!!!Behind {} cone I found {}'.format(color,cone_object))
            self.start_classification(video_path,color)
            self.gpg.drive_cm(drive_cm)
            self.gpg.turn_degrees(-90)
            #self.servo.rotate_servo(30)
        else:
            #take_picture(picture_path)
            pass

    #Running classification on a separate thread to make it run when robot is running
    def classify_and_log_object_thread(self,video_path,color):
        cone_object = self.image_model.classify_video(video_path+color)
        print('!!!!!!!!!!!!Behind {} cone I found {}'.format(color,cone_object))
        txt = ','.join([str(datetime.now()), color, str(cone_object)])
        self.log(txt)
        color_name = color
        class_name = cone_object
        self.logger.info('%s,%s' % (color_name, class_name))
        print('Logged: ', txt)

    #Start the object classification thread
    def start_classification(self,video_path,color):
        print('Starting a new classification thread')
        Thread(target=self.classify_and_log_object_thread,args=(video_path,color)).start()
        print('Running classification on a different thread')
        return self



    def classify_and_log(self, color):
        image_dir = os.path.join(self.image_model.image_dir, color)
        classes, probs, objects = self.image_model.classify(image_dir)
        txt = ','.join([str(datetime.now()), color, str(objects)])
        self.log(txt)
        print('Logged: ', txt)
        return txt

    def hunt(self, color):
        #self.center_cone(color)
        self.drive_to_cone(color)
        self.circum_navigate(color)
        #self.classify_and_log(color)


if __name__ == '__main__':
    import config
    from coneutils import calibrate

    boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')

    bot = ScavBot(
        image_model_dir='Sample_TFLite_model', 
        image_dir='/home/pi/Pictures/scav_hunt',
        params=config.params,
        boundaries = boundaries_dict
    )
