#!/usr/bin/env python
# coding: utf-8

import os

import pickle
import audioop as ao
import numpy as np
import time
from datetime import datetime, date
from datetime import timedelta as td


import wave
import pyaudio
from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal

import noisereduce as nr

#from UrbanHMM import *
#from UrbanHMM import UrbanHMMClassifier

from audio_classifier import TfliteSoundClassifier
from pathlib import Path


def perform_scavear(logger):
    from datetime import date
    today = str(date.today())

    # initialize scavear and listener
    runtime = 500
    log_dir='logs'

    RECORD_SECONDS = 4.5
    THRESHOLD = 3000
    SHORT_NORMALIZE = (1.0/32768.0)
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    swidth = 2
    Max_Seconds = 10
    silence = True
    Time=0
    all_=[]

    TimeoutSignal=int((RATE / CHUNK * Max_Seconds) + 2),

    #initialize model
    model_dir = 'models/audio/'
    model_name = 'hmm_cvbest_f1_56437703.pkl'
    audio_path='data/audio/{}'.format(today)

    # with open(os.path.join(model_dir, model_name), 'rb') as model_file:
    #     model = pickle.load(model_file)
    
    nn_model =  TfliteSoundClassifier(os.path.join(model_dir, 'model.tflite'), os.path.join(model_dir, 'labels.txt'))
    
    # Log File
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'log_'+str(date.today())+'.txt')



    # def classify(audio_file_path):
    #     #return model.predict(audio_file_path, prediction_type = "labels")
    #     return nn_model.predict(audio_file_path)


    def log(txt):
        print('logging...')
        with open(log_path, 'a') as f:
            f.write(txt)
            f.write('\n')

    def filter_clip(target_filename):
        print('filtering...')
        #load original recording
        rate, data = wavfile.read(target_filename)
        data = data.astype(np.float16)

        #get noise data
        noisy_part = data[174760:196608] # last 0.5 seconds of recording

        #perform filtering
        recovered_signal = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False).astype(np.int)

        newname = audio_clip_path.replace('.wav','')
        filtered_audio_clip_path = newname + '_filtered.wav' 

        #save filtered file
        scipy.io.wavfile.write(filtered_audio_clip_path, rate, np.asarray(recovered_signal[0:174760], dtype=np.int16))

        return filtered_audio_clip_path

    def open_stream():
        p = pyaudio.PyAudio()
        stream = p.open(
            format = FORMAT,
            channels = CHANNELS,
            rate = RATE,
            input = True,
            output = True,
            frames_per_buffer = CHUNK)
        return stream, p

    def close_stream(stream, p):
        stream.stop_stream()
        stream.close()
        p.terminate()

    def listen_trigger_record(seconds, print_rms = False ):
        stream, p = open_stream()
        print('listening...')
        frames = []
        rms_values = np.empty(1,dtype=float)
        while (True):
            trigger = stream.read(CHUNK)
            rms_value = int(ao.rms(trigger,2))
            rms_values = np.append(rms_values,rms_value)
            if(print_rms == True):
                print(rms_value)
            if (rms_value > THRESHOLD):
                print('recording...')
                frames.append(trigger)
                logtime = str(datetime.now())
                for i in range(0, int(RATE / CHUNK * seconds)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                if not os.path.exists(audio_path):
                    os.makedirs(audio_path)

                filename = audio_path + '/' + 'recording_' + format(datetime.now().strftime('%m%d%Y%H%M%S')) + '.wav'
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(swidth)
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                return filename, logtime


    def classify(audio_file_path):
        print('classifying...')
        #return model.predict(audio_file_path, prediction_type = "labels")
        return nn_model.predict(audio_file_path)

    def log(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write('\n')

    start = time.time()
    while time.time() - start < runtime:
        # listen, trigger, and record
        audio_clip_path, logtime = listen_trigger_record(RECORD_SECONDS, print_rms = False)
        # send for filtering
        filtered_audio_clip_path = filter_clip(audio_clip_path)
        #classify
        audio_class = classify(filtered_audio_clip_path)
        #audio_class = classify(audio_clip_path)
        # log
        txt = ','.join([str(logtime), str(audio_class)])
        logger.info(audio_class)
        print('Logged: ', txt)