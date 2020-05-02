import sys
sys.path.append('/home/pi/Dexter/GoPiGo3/Software/Python')


import scavbot
import config
from coneutils import calibrate
boundaries_dict = calibrate.load_boundaries('coneutils/boundaries.json')
image_model_dir = '/home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/custom_model_edgeTPU/'
cone_model_dir = '/home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/custom_cone_model_edgetpu/'
image_dir='/home/pi/Pictures/'
cone_image_dir='/home/pi/Pictures/Cones/orange'

bot = scavbot.ScavBot(
	image_model_dir=image_model_dir ,
	cone_model_dir = cone_model_dir,   
	image_dir=image_dir,
	cone_image_dir=cone_image_dir,
	params=config.params,
	boundaries = boundaries_dict
)

#just to center the servo at the beginnning
bot.servo.rotate_servo(90)

print('Starting The Hunt')
print('Looking for Green cone')
bot.drive_to_cone("green")
print('Reached Green cone')
bot.circum_navigate("green")
print('Green cone done, looking for base cone')

bot.drive_to_cone("red")
print('Reached base')
bot.circum_navigate("red")

print('Looking for Yellow cone now')
bot.drive_to_cone("yellow")
print('Reached Yellow cone')
bot.circum_navigate("yellow")
print('Yellow cone done, looking for base cone')

bot.drive_to_cone("red")
print('Reached base')
bot.circum_navigate("red")

print('Looking for Purple cone now')
bot.drive_to_cone("purple")
print('Reached Purple cone')
bot.circum_navigate("purple")
print('Purple cone done, looking for base cone')

bot.drive_to_cone("red")
bot.gpg.set_speed(300)
bot.gpg.drive_cm(20)
bot.gpg.turn_degrees(90)
print('Reached Base: HUNT OVER!!!!')