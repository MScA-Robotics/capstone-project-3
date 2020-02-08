from test_camera import take_picture
import time
from threading import Thread

def pause():
    print('Starting!')
    i = 0
    t0 = time.time()
    while i < 5000000:
        i += 1
    print('Paused for {} seconds!'.format(time.time() - t0))
    
def picture():
    take_picture()
    print('Picture taken')

t1 = Thread(name='pause', target = pause)
t2 = Thread(name='picture', target=picture)

t1.start()
t2.start()

t1.join()
t2.join()