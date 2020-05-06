# Script to test the cone model detection with EDGE TPU
#
# Make sure to run this from within the scav-hunt directory

import os
import sys
sys.path.append('/home/pi/Dexter/GoPiGo3/Software/Python')
sys.path.append('..')
import scavbot
import config
from coneutils import calibrate

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# e.g. /home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/tests
HUNT_DIR = str.replace(TESTS_DIR, '/tests', '')
# e.g. /home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt
MODEL_DIR = os.path.join(HUNT_DIR, 'models')
# e.g. /home/pi/Desktop/code/capstone_project/capstone-project-3/scav-hunt/models

#import pdb
#pdb.set_trace()

image_model_dir = os.path.join(MODEL_DIR, 'visual', 'custom_model_edgeTPU')
cone_model_dir = os.path.join(MODEL_DIR, 'visual', 'custom_cone_model_edgetpu')
image_dir='/home/pi/Pictures'
cone_image_dir= os.path.join(image_dir, 'scav_hunt')

print('Test the loading of the TFLITE runtime and make sure the TPU is plugged in')
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
model_name='cone_edgetpu.tflite'
model_path = os.path.join(cone_model_dir, model_name)
interpreter = Interpreter(model_path,
  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
print('TFLITE and EdgeTPU successfully loaded')

from scaveye import ConeClassificationModel
model = ConeClassificationModel(
    model_dir = cone_model_dir,
    image_dir = cone_image_dir,
    graph_name='cone_detect.tflite',
    min_conf_threshold=0.3,
    use_TPU=True)

boxes_list, classes_list, scores_list, objects_detected, objects_dict = model.classify(os.path.join(model.image_dir, 'archive/orange'))
