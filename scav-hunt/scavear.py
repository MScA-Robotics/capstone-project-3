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

    def __init__(self, noise_path='noise'):
        self.noise_thresh = np.load(os.path.join(noise_path, 'noise_thresh.npy'))
        self.mean_freq_noise = np.load(os.path.join(noise_path, 'mean_freq.npy'))
        self.std_freq_noise = np.load(os.path.join(noise_path, 'std_freq.npy'))
        self.noise_stft_db = np.load(os.path.join(noise_path, 'noise_db.npy'))


    def removeNoise(
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




