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
from collections import Counter

# If using TPU, need to load a different library
#from tensorflow.lite.python.interpreter import Interpreter


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

        pkg = importlib.util.find_spec('tensorflow')
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                print('Loading tflite interpreter')
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                print('Loading tflite interpreter')
                from tflite_runtime.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (graph_name == 'detect.tflite'):
                graph_name = 'edgetpu.tflite'
        # Load the model 
        PATH_TO_CKPT = os.path.join(CWD_PATH, model_dir, graph_name)
        #self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
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
        """Function to detect objects in video file"""
        #1. Get the list of all video files from the directory passed in
        videos = glob.glob(video_dir + '/*')

        #2. Check the number of *.avi files in the folder
        num_videos = len(videos)
        #3. Do not run classification if number of videos in the folder are more than 10 and alert
        if num_videos > 10:
            print('Found more than 10 videos in the directory: {}'.format(video_dir))
            return
        #4. For each video file
        for video_file in videos:
            video_name=os.path.basename(video_file)
            print('Processing video: {}'.format(video_name))
            #4.1 Open the video file 
            video = cv2.VideoCapture(video_file)
            imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            collect_labels = []
            #4.1.1 pass frame by frame to the  detection model
            index = 0
            while(video.isOpened()):
                # Acquire frame and resize to expected shape [1xHxWx3]
                ret, frame = video.read()
                #print('Processing frame: {}'.format(index+1))
                if frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
                input_data = np.expand_dims(frame_resized, axis=0)
                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if self.floating_model:
                    input_data = (np.float32(input_data) - self.input_mean) / self.input_std

                # Perform the actual detection by running the model with the image as input
                self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
                self.interpreter.invoke()

                # Retrieve detection results
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects

                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                        # Draw label
                        object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        #label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        #cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                        collect_labels.append(object_name)
                index += 1
            video.release()
            #cv2.destroyAllWindows()
            most_common,num_most_common = Counter(collect_labels).most_common(1)[0]
            max_object = most_common
            print('Maximum detected object :{}'.format(max_object))
            print(Counter(collect_labels))
        return max_object


if __name__ == '__main__':

    model = ObjectClassificationModel('Sample_TFLite_model', '/home/pi/Pictures/scav_hunt')
    classes, scores, objects = model.classify(os.path.join(model.image_dir, 'archive/orange'))


