#record while drive

from recordAudio import record_chunk
import multiprocessing as mp
import time
import multiprocessing as mp
from easygopigo3 import EasyGoPiGo3
gpg = EasyGoPiGo3()

def drive(seconds):
    now = time.time()
    while time.time() - now < seconds:
        drive_rectangle()

def drive_rectangle(side=5):
    # gpg.set_speed(h_spd)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()
    gpg.turn_degrees(90)


if __name__ == '__main__':
    seconds = 90

    p1 = mp.Process(target=drive, args=(seconds,))
    p2 = mp.Process(target=record_chunk, args=(seconds,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

