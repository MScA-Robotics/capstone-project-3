# Copyright 2020 Audrey Salerno. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Classify sound file using tflite model and spectrograms"""

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
from pathlib import Path
import librosa
import os
import matplotlib.pyplot as plt

def save_spectrogram(filepath, sr = 8000, replace = True): 
    """ Utility function to process .wav and save librosa melspectrogram as .png image.

    Saves melspectrogram .png from .wav file and saves it in a child directory "spectrograms", 
    relative to the .wav file, with the same base name as the .wav file.

    Uses default parameters of librosa.feature.melspectrogram()

    Creates the image using pyplot, and stretches to fit the figsize if needed.

    Example:
        Save melspectrogram image for "./audio/file.wav", replacing if exists (default)

        >>> print(save_spectrogram('./audio/file.wav', 8000))
        "C:/Project/audio/file.png"

        Example will replace PNG file at "C:/Project/audio/spectrogram/file.png" if already exists.

    Args:
        filepath (str): path to wav file
        sr (int): sample rate to downsample audio file.  Default 8000 for model.
        replace (bool): indicator to replace wav file if one already exists. Default True.

    Returns:
        string: path to spectrogram .png image.

    """
    spectrogram_path = f'{os.path.dirname(filepath)}/spectrogram'
    imgname = os.path.basename(filepath).replace('.wav','.png')
    img_path  = f'{spectrogram_path}/{imgname}'

    if (not os.path.isfile(img_path)) | replace:
        if not os.path.exists(spectrogram_path):
            os.makedirs(spectrogram_path)

        try:
            signal, sampling_freq = librosa.load(filepath, sr)
        except Exception as e:
            print(e)
            print("Failed to read {}".format(filepath))

        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        S = librosa.feature.melspectrogram(y=signal, sr=sampling_freq)
        #S = S.T[20:S.shape[1]-2].T
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(img_path, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

    return img_path

def load_labels(saved_labels):
    """ Loads labels .txt saved from exported tflite model.
    
    Args:
        saved_labels (str or Path): path to saved .txt labels file.
    
    Retrns:
        list object of labels in order of model predicition indicies. 
    """
    
    with open(saved_labels, 'r') as f:
        return [line.strip() for line in f.readlines()]

class TfliteSoundClassifier:
    """Class to load and predict from a tflite classifier for .wav file sounds.
    
    Args:
        saved_model (str or Path): path to the saved .tflite model file.
        saved_labels (str or Path): path to the saved .txt labels file.
    
    Returns:
        Tflite classifier
    """
    
    def __init__(self, saved_model, saved_labels):
        
        self.saved_model = str(saved_model)
        self.labels = load_labels(saved_labels)
        self._initialize_interpreter()

    def _initialize_interpreter(self):
        """ Initialize tflite Interpreter from saved model """
        
        self.interpreter = tflite.Interpreter(self.saved_model)
        self.interpreter.allocate_tensors()
        
        self.input_spec = self.interpreter.get_input_details()
        self.output_spec = self.interpreter.get_output_details()
        
    def predict_spectrogram_proba(self, image_path):
        """ Predict class probabilities of single image 
        
        Args: 
            image_path (str or Path): Path to image to be classified.
        
        Returns:
            Predited probabilties of classes, sorted by class index.
        """
        input_data = self.process_image(image_path)
        self.interpreter.set_tensor(self.input_spec[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_spec[0]['index'])
        proba = np.squeeze(output_data)
        return proba
    
    def predict_spectrogram_class(self, image_path, predict_label = True):
        """ Predict class of single image 
        
        Args: 
            image_path (str or Path): Path to image to be classified
            predict label (bool): Flag to predict class label or class index. 
                Default True.
        
        Returns:
            Either predicted class label (str) or index (int) of image.
        """
        predict_proba = self.predict_spectrogram_proba(image_path)
        top_1 = predict_proba.argmax()
        if predict_label:
            return self.labels[top_1]
        else:
            return top_1

    def process_image(self, image_path):
        """ Load and process spectrogram image to meet model input requirements.

        Load spectrogram image from path, convert to RGB, and resize to model input image size
        and input dimensions.  Also scales to [0,1].

        Args:
            image_path (str or Path): Path to image file for processing

        Returns:
            Numpy array of shape (1, h, w, 3) 
                where h and w are the height and width of the image.
        """
        input_img_shape = (self.input_spec[0]['shape'][1], self.input_spec[0]['shape'][2])
        img = Image.open(image_path).convert('RGB').resize(input_img_shape)
        input_data = np.expand_dims(img, axis = 0)
        input_data = np.float32(input_data)/255
        return input_data
    
    def predict(self, audio_file, predict_label = True):
        """ Predict class of single audio wav file 
        
        Args: 
            audio_file_path (str or Path): Path to audio wav file to be classified
            sr (int): sample rate of audio file
            predict label (bool): Flag to predict class label or class index. 
                Default True.
        
        Returns:
            Either predicted class label (str) or index (int) of audio file.
        """
        
        spectrogram_path = save_spectrogram(audio_file, replace = True)
        return self.predict_spectrogram_class(spectrogram_path, predict_label)
