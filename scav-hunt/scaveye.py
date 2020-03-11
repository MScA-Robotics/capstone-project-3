#Changes by Raghav for video classification
import os
import glob
import picamera
import cv2
import numpy as np
import importlib.util
from datetime import datetime
import videorecorder as vr
import time

# If using TPU, need to load a different library
from tensorflow.lite.python.interpreter import Interpreter


def take_picture(path):
    if path is None:
        path = "/home/pi/Pictures"
    camera = picamera.PiCamera()
    try:
        camera.capture(os.path.join(path, "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S'))))
    finally:
        print('Picture taken')
        camera.close()

def record_video(path=None,cone_color='green',duration=5,runid=0):
    if path is None:
        path="/home/pi/Videos"
    path = os.path.join(path,cone_color)
    try:
        recorder = vr.VideoRecorder(path,runid)
        print('Loaded Video Recorder')
        recorder.start_recording()
        time.sleep(duration)
        recorder.stop_recording()
    except:
        print('Video Recording failed')
    finally:
        print('Video recorded')


class ObjectClassificationModel:

    def __init__(self, model_dir, image_dir, graph_name='detect.tflite', min_conf_threshold=0.5, use_TPU=False):

        self.model_dir = model_dir
        self.image_dir = image_dir
        self.min_conf_threshold = float(min_conf_threshold)
        self.use_TPU = use_TPU
        self._load_model(model_dir=model_dir, graph_name=graph_name)

    def _load_model(self, model_dir, graph_name):
        CWD_PATH = os.getcwd()

        # Load model labels
        PATH_TO_LABELS = os.path.join(CWD_PATH, model_dir, 'labelmap.txt')
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        if labels[0] == '???':
            del(labels[0])
        self.labels = labels

        # Load the model 
        PATH_TO_CKPT = os.path.join(CWD_PATH, model_dir, graph_name)
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def classify(self, image_dir):
        images = glob.glob(image_dir + '/*')
        classes_list = []
        scores_list = []
        for image_path in images:
            print('Classifying: {}'.format(image_path))
            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()

            # Retrieve detection results
            # We are not using the boxes right now since we do not need to know 
            # where picture the object is, only that it is there.

            # boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
            # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            classes_list.append(classes[scores > self.min_conf_threshold])
            scores_list.append(scores[scores > self.min_conf_threshold])

        objects_detected = {}
        for classes in classes_list:
            objects = set([self.labels[int(c)] for c in classes])
            for obj in objects:
                if obj in objects_detected.keys():
                    objects_detected[obj] += 1
                else:
                    objects_detected[obj] = 1

        return classes_list, scores_list, objects_detected
    
    
    def classify_video(self, video_dir):
        pass

if __name__ == '__main__':

    model = ObjectClassificationModel('Sample_TFLite_model', '/home/pi/Pictures/scav_hunt')
    classes, scores, objects = model.classify(os.path.join(model.image_dir, 'archive/orange'))


