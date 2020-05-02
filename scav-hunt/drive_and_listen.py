#record while drive

from scavear_main import Listener
from scavear_main import Scavear
#import multiprocessing as mp
from threading import Thread
import time
from easygopigo3 import EasyGoPiGo3
from datetime import date

gpg = EasyGoPiGo3()


def drive(seconds = 150):
    now = time.time()
    while time.time() - now < seconds:
        drive_rectangle()

def drive_rectangle(side=5):
    # gpg.set_speed(h_spd)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(95)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(95)
    gpg.forward()

    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(95)
    gpg.forward()
    time.sleep(side)
    gpg.stop()
    gpg.turn_degrees(95)


if __name__ == '__main__':

    today = str(date.today)

    ear = Scavear(
        model_dir = 'models/audio',
        model_name = 'hmm_cvbest_f1_56437703.pkl',
        audio_path = 'data/audio{}'.format(today)
    )

    seconds = 150

    p1 = Thread(target=drive)
    p2 = Thread(target=ear.listen_record_classify_log)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

