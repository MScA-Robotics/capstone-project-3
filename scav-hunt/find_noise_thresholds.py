#!/usr/bin/env python
# coding: utf-8


#~~~~~~~~~~~~
#Place this file in the /noise directory and run to get GPG noise thresholds for a particular mic and environment.
#~~~~~~~~~~~

from scipy.io import wavfile
import scipy.signal
import numpy as np
import librosa
import time
from datetime import timedelta as td
from datetime import datetime
import pyaudio
import wave
import multiprocessing as mp
import time
from easygopigo3 import EasyGoPiGo3


gpg = EasyGoPiGo3()

def drive(drive_seconds):
    now = time.time()
    while time.time() - now < drive_seconds:
        drive_rectangle()

def drive_rectangle(side=5):
    # gpg.set_speed(h_spd)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()

    gpg.turn_degrees(90)
    gpg.forward()
    time.sleep(side)
    gpg.stop()
    gpg.turn_degrees(90)



def record_chunk(record_secs = 5,
                 chans = 1,
                 samp_rate = 44100,
                 chunk = 4096,
                 #dev_index = 0,
                 wav_output_filename = 'noise_signal.wav'): # name of .wav file

    form_1 = pyaudio.paInt16 # 16-bit resolution

    audio = pyaudio.PyAudio() # create pyaudio instantiation

    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans,
                    #input_device_index = dev_index,
                    input = True, \
                    frames_per_buffer=chunk)
    print("recording")
    frames = []

    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk,exception_on_overflow = False)
        frames.append(data)

    print("finished recording")
 
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def findNoiseThresh(
    noise_clip,n_fft = 2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5
):
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    return noise_thresh, mean_freq_noise, std_freq_noise,noise_stft_db


if __name__ == '__main__':
    seconds = 60
    wav_output_filename = 'noise_signal.wav'

    p1 = mp.Process(target=record_chunk, args=(seconds,))
    p2 = mp.Process(target=drive, args=(seconds,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

print('Processes completed:')

wav_loc = "noise_signal.wav"
noise_rate, noise_data = wavfile.read(wav_loc)
noise_clip = noise_data.astype(float)
noise_thresh, mean_freq, std_freq,noise_db = findNoiseThresh(noise_clip = noise_clip)

np.save('noise_thresh.npy', noise_thresh)
np.save('mean_freq.npy', mean_freq)
np.save('std_freq.npy', std_freq)
np.save('noise_db.npy', noise_db)

