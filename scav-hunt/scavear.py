import os
import sys
import struct
import math

import numpy as np
import time
from datetime import datetime
from datetime import timedelta as td

import librosa
import wave
import pyaudio
from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal

import soundfile as sf
import sounddevice as sd


def record_chunk(dur = 60, fs = 44100, chans = 1, output_file = 'recording.wav'):
    rec = sd.rec(int(dur*fs), samplerate=fs, channels=chans, blocking=True)
    sf.write(output_file, rec, fs)


class Listener:

    @staticmethod
    def _stft(y, n_fft, hop_length, win_length):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    @staticmethod
    def _istft(y, hop_length, win_length):
        return librosa.istft(y, hop_length, win_length)

    @staticmethod
    def _amp_to_db(x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

    @staticmethod
    def _db_to_amp(x,):
        return librosa.core.db_to_amplitude(x, ref=1.0)

    def __init__(self, 
                 noise_path='noise',
                 THRESHOLD = 30,
                 SHORT_NORMALIZE = (1.0/32768.0),
                 CHUNK = 4096,
                 FORMAT = pyaudio.paInt16,
                 CHANNELS = 1,
                 RATE = 44100,
                 swidth = 2,
                 Max_Seconds = 10,
                 silence = True,
                 FileNameTmp = 'test2.wav',
                 Time=0,
                 all_=[]):

        self.THRESHOLD = THRESHOLD
        self.SHORT_NORMALIZE = SHORT_NORMALIZE
        self.CHUNK = CHUNK
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE
        self.swidth = swidth
        self.Max_Seconds = Max_Seconds
        self.FileNameTmp = FileNameTmp
        self.Time = Time
        self.all = all_

        self.TimeoutSignal=int((RATE / CHUNK * Max_Seconds) + 2),

        self.noise_thresh = np.load(os.path.join(noise_path, 'noise_thresh.npy'))
        self.mean_freq_noise = np.load(os.path.join(noise_path, 'mean_freq.npy'))
        self.std_freq_noise = np.load(os.path.join(noise_path, 'std_freq.npy'))
        self.noise_stft_db = np.load(os.path.join(noise_path, 'noise_db.npy'))

    def remove_noise(
        audio_clip,
        n_grad_freq=2,
        n_grad_time=4,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_std_thresh=1.5,
        prop_decrease=1.0):
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
        noise_thresh = self.noise_thresh
        mean_freq_noise = self.mean_freq_noise
        std_freq_noise = self.std_freq_noise
        noise_stft_db = self.noise_stft_db

        # STFT over signal
        sig_stft = self._stft(audio_clip, n_fft, hop_length, win_length)
        sig_stft_db = self._amp_to_db(np.abs(sig_stft))
        # Calculate value to mask dB to
        mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
        
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
        return recovered_signal

    def rms(self, frame):
        count = len(frame)/self.swidth
        format = "%dh"%(count)
        # short is 16 bit int
        shorts = struct.unpack(format, self.frame)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * self.SHORT_NORMALIZE
            sum_squares += n*n
        # compute the rms 
        rms = math.pow(sum_squares/count, 0.5);
        return rms * 1000

    def filter_stream(self, stream):
        #convert bytestream to 16bit PCM
        sig = np.frombuffer(stream, dtype='<i2').reshape(-1, CHANNELS)
        # Change shape and type for noise removal function
        sig = sig.T[0].astype('float')
        #GoPiGo noise removal
        output = remove_noise(
            audio_clip=sig,
            n_std_thresh=2,
            prop_decrease=0.95)
        return(output)

    def open_stream(self):
        p = pyaudio.PyAudio()
        self.stream = p.open(
            format = self.FORMAT,
            channels = self.CHANNELS,
            rate = self.RATE,
            input = True,
            output = True,
            frames_per_buffer = self.CHUNK)
        return self.stream

    def listen(self, with_filter = False):
        print("listening now...")
        silence = True
        while silence:
            try:
                input = self.stream.read(CHUNK)
                print('new stream chunk')
            except:
                continue
            if (with_filter):
                filtered = filter_stream(input)
                filtered_tuple = tuple(filtered)
                rms_value = self.rms(filtered_tuple, bytestream = False)
            else:
                rms_value = self.rms(input, bytestream = True)
                print(rms_value)
            if (rms_value > self.THRESHOLD):
                print ("hello doubladay, you should trigger recording here....")
                silence = False
                # trigger recording function here

