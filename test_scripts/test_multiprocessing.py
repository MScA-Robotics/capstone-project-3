from test_camera import take_picture
import time
import multiprocessing as mp
from easygopigo3 import EasyGoPiGo3
gpg = EasyGoPiGo3()

def drive(seconds):
    gpg.set_speed(h_spd)
    gpg.forward()
    time.sleep(seconds)
    gpg.stop()

def pictures(n_pics):
    for i in range(n_pics):
        take_picture()
        time.sleep(1)

p1 = mp.Process(target=drive, args=(5,))
p2 = mp.Process(target=pictures, args=(5,))

p1.start()
p2.start()
p1.join()
p2.join()



