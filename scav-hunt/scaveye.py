import os
import picamera
import cv2
import numpy as np
import importlib.util

# If using TPU, need to load a different library
from tensorflow.lite.python.interpreter import Interpreter


def take_picture(path):
    if path is None:
        path = "/home/pi/Pictures"
    camera = picamera.PiCamera()
    try:
       camera.capture(os.path.join(path, "image_{0}.jpg".format(datetime.now().strftime('%m%d%Y%H%M%S'))))
    finally:
       camera.close()


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
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def classify(self, image_dir):
        full_image_dir = os.path.join(os.getcwd(), image_dir)
        images = glob.glob(image_dir + '/*')
        classes_list = []
        scores_list = []
        for image_path in images:
            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            # We are not using the boxes right now since we do not need to know 
            # where picture the object is, only that it is there.

            # boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            classes_list.append(classes)
            scores_list.append(scores)

        return classes_list, scores_list




