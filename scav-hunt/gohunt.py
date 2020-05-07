import sys
sys.path.append('/home/pi/Dexter/GoPiGo3/Software/Python')

from multiprocessing import Process
import scavbot
import config
from coneutils import calibrate
import scavear
import time
from threading import Thread

import logging
main_logger = logging.getLogger('gpg')
main_logger.setLevel(logging.DEBUG)
fname = 'gopigo.log' # Any name for the log file

# Create the FileHandler object. This is required!
fh = logging.FileHandler(fname, mode='w')
fh.setLevel(logging.INFO)  # Will write to the log file the messages with level >= logging.INFO

# The following row is strongly recommended for the GoPiGo Test!
fh_formatter = logging.Formatter('%(relativeCreated)d,%(name)s,%(message)s')
fh.setFormatter(fh_formatter)
main_logger.addHandler(fh)
mic_logger = logging.getLogger('gpg.mic')
visual_logger = logging.getLogger('gpg.find_cone')


boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')
image_model_dir = '/home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/custom_model_edgeTPU/'
cone_model_dir = '/home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/custom_cone_model_edgetpu/'
image_dir='/home/pi/Pictures/'
cone_image_dir='/home/pi/Pictures/Cones/orange'


#initialize bot

bot = scavbot.ScavBot(
	image_model_dir=image_model_dir ,
	cone_model_dir = cone_model_dir,   
	image_dir=image_dir,
	cone_image_dir=cone_image_dir,
	params=config.params,
	boundaries = boundaries_dict,
	logger=visual_logger
)

def wait_for_start_signal():
	print('listening for start signal...')
	import pyaudio
	import audioop as ao
	THRESHOLD = 3000
	FORMAT=pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	CHUNK = 1024

	#listen for start signal
	silence=True
	swidth = 2
	audio  = pyaudio.PyAudio()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
					rate=RATE, input=True,
					frames_per_buffer=CHUNK)
	while(silence):
		#print('Waiting for start signal...')
		trigger = stream.read(CHUNK)
		volume = int(ao.rms(trigger, swidth))
		#print(volume)
		if (volume > THRESHOLD):
			silence = False
			mic_logger.info('Start')
			stream.stop_stream()
			stream.close()
			audio.terminate()
	print('GO!!!!!')

wait_for_start_signal()
print('Starting The Hunt')
time.sleep(2) #wait for 2 seconds to let the beep end then start scavear.

#scavear.perform_scavear(mic_logger)
#start scavear on different thread
p = Thread(target=scavear.perform_scavear,args=(mic_logger,))
p.start()
#p.join()



#for testing only
# bot.hunt('green')
# bot.hunt('yellow')
# bot.hunt('purple')



bot.servo.rotate_servo(90)
bot.hunt('green')
bot.hunt('red')
bot.hunt('purple')
bot.hunt('red')
bot.hunt('yellow')
bot.drive_to_cone('red')
bot.gpg.set_speed(300)
bot.gpg.drive_cm(20)
bot.gpg.turn_degrees(90)
print('Reached Base: HUNT OVER!!!!')
# p.terminate()

mic_logger.info('Finish')


# #submit gopigo.log to ilykei server
from robo_client import connection
print('Submitting log to Ilykei')  
HOST, PORT = 'datastream.ilykei.com', 30078
login = 'xxxx'
password = 'xxx'
split_id = 19
filename = 'gopigo.log'
status = connection(HOST, PORT, login, password, split_id, filename)   
if(status):
	print('Log submitted successfuly')