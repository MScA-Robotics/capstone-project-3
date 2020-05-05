#!/usr/bin/env python
# coding: utf-8

import os
import sys
import struct
import math
import pickle
import audioop as ao
import numpy as np
import time
from datetime import datetime, date
from datetime import timedelta as td

import librosa
import wave
import pyaudio
from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal
from array import *

import soundfile as sf
import sounddevice as sd
from UrbanHMM import *
from UrbanHMM import UrbanHMMClassifier
#from UrbanAudio import *
#from UrbanAudio import UrbanHMMClassifier


# In[79]:


def perform_scavear():
    from datetime import date
    today = str(date.today())
    
    
    # initialize scavear and listener
    runtime = 60
    log_dir='logs'
    
    noise_path = 'noise' 
    #audio_path='/home/pi/Desktop/scav_hunt/audio'
    
    RECORD_SECONDS = 4
    THRESHOLD = 1000
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

    noise_thresh = np.load(os.path.join(noise_path, 'noise_thresh.npy'))
    mean_freq_noise = np.load(os.path.join(noise_path, 'mean_freq.npy'))
    std_freq_noise = np.load(os.path.join(noise_path, 'std_freq.npy'))
    noise_stft_db = np.load(os.path.join(noise_path, 'noise_db.npy'))
    
    #initialize model
    model_dir = 'models/audio/'
    model_name = 'hmm_cvbest_f1_56437703.pkl'
    audio_path='data/audio/{}'.format(today)

    with open(os.path.join(model_dir, model_name), 'rb') as model_file:
        model = pickle.load(model_file)
        # Log File
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'log_'+str(date.today())+'.txt')
    
    
    def classify(audio_file_path):
        return model.predict(audio_file_path, prediction_type = "labels")

    def log(txt):
        with open(log_path, 'a') as f:
            f.write(txt)
            f.write('\n')
            
    
    def _stft(y, n_fft, hop_length, win_length):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _istft(y, hop_length, win_length):
        return librosa.istft(y, hop_length, win_length)

    def _amp_to_db(x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

    def _db_to_amp(x,):
        return librosa.core.db_to_amplitude(x, ref=1.0)

    def remove_noise(
        
        audio_clip_path,
        n_grad_freq=2,
        n_grad_time=4,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_std_thresh=1.5,
        prop_decrease=1.0):
        
        rate, data = wavfile.read(audio_clip_path)
        data = data / 32768
        audio_clip = data.astype(float)
        
        """Remove noise from audio based upon a clip containing only noise
        Args:
            audio_clip (array): The first parameter.
            noise_clip (array): The second parameter.
            n_grad_freq (int): how many frequency channels to smooth over with the mask.
            n_grad_time (int): how many time channels to smooth over with the mask.
            n_fft (int): number audio of frames between STFT columns.
            win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
            hop_length (int):number audio of frames between STFT columns.
            n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
            prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
            visual (bool): Whether to plot the steps of the algorithm
        Returns:
            array: The recovered signal with noise subtracted
        """      
        
        # STFT over signal
        sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
        sig_stft_db = _amp_to_db(np.abs(sig_stft))
        # Calculate value to mask dB to
        mask_gain_dB = np.min(self._amp_to_db(np.abs(sig_stft)))

        # Create a smoothing filter for the mask in time and frequency
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        # calculate the threshold for each frequency/time bin
        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T
        # mask if the signal is above the threshold
        sig_mask = sig_stft_db < db_thresh
        # convolve the mask with a smoothing filter
        sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * prop_decrease
        # mask the signal
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
        )  # mask real
        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
            1j * sig_imag_masked
        )
        # recover the signal
        recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
        recovered_spec = _amp_to_db(
            np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
        )
        
        newname = audio_clip_path.replace('.wav','')
        filtered_audio_clip_path = newname + '_filtered.wav' 
        #print(filtered_audio_clip_path)
        wavfile.write(filtered_audio_clip_path, rate, recovered_signal)
        
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
                frames.append(trigger)
                logtime = str(datetime.now())
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                if not os.path.exists(audio_path):
                    os.makedirs(audio_path)

                filename = audio_path + '/' +                            'recording_' +                            format(datetime.now().strftime('%m%d%Y%H%M%S')) +                            '.wav'
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(swidth)
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                close_stream(stream, p)
                rms_filename = 'rms.txt' #+ \
                           #format(datetime.now().strftime('%m%d%Y%H%M%S')) + \
                           #'.txt'
                np.savetxt(rms_filename, rms_values, delimiter = ',')
                return filename, logtime
            
            
    def classify(audio_file_path):
        return model.predict(audio_file_path, prediction_type = "labels")

    def log(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write('\n')       
    
    
    start = time.time()
    while time.time() - start < runtime:
        audio_clip_path, logtime = listen_trigger_record(4, print_rms = True)
        #filtered_audio_clip_path = remove_noise(audio_clip_path)
        audio_class = classify(audio_clip_path)
        txt = ','.join([str(logtime), str(audio_class)])
        # insert logging function
        print('Logged: ', txt)   

