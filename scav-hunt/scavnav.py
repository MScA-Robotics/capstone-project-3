from easygopigo3 import EasyGoPiGo3


class NavigationControl:

    def __init__(self, gpg=None, params=None):
        if gpg is None:
            gpg = EasyGoPiGo3()

        self.dist_sensor = gpg.init_distance_sensor()

             #Set required variables
        if params is None:
            self.radius = 200
            self.h_spd  = 400
            self.m_spd  = 200
            self.l_spd  = 30
        else:
            self.radius = params['radius']
            self.h_spd  = params['h_spd']
            self.m_spd  = params['m_spd']
            self.l_spd  = params['l_spd']

    def drive_to_cone(self):
        # Drive to cone at full bore
        self.gpg.set_speed(h_spd)
        ob_dist = self.dist_sensor.read_mm()
        while ob_dist >= self.rad:
            self.gpg.forward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))
        self.gpg.stop()

        # Back away to the exact distance at a slower speed
        self.gpg.set_speed(l_spd)
        while ob_dist < rad:
            self.gpg.backward()
            ob_dist = self.dist_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(ob_dist))  
        self.gpg.stop()
        print("MADE IT!")

    def circum_navigate(self):
        # Set the speed to medium speed
        self.gpg.set_speed(self.m_spd)
        print("I will now cicle the cone at {} mm ".format(self.radius))

        # Circumscibe a circle around the cone
        # rotate gpg 90 degrees to prep for the orbit
        gpg.turn_degrees(-90)

        # Complete the orbit
        gpg.orbit(180, (2*self.radius/10))

        # Rotate back to facing the cone
        self.gpg.turn_degrees(90)
        ob_dist = self.dist_sensor.read_mm()
        print("The cone is now at: {} mm ".format(ob_dist))

        # Return to a base position
        print("That was fun... I go home now") 
        gpg.drive_cm(-20,True)
