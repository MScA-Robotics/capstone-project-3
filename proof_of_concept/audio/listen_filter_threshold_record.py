import pyaudio
import math
import struct
import wave
import sys

from scipy.io.wavfile import write
from scipy.io import wavfile
import scipy.signal
import numpy as np
import librosa
import time
from datetime import datetime
from datetime import timedelta as td

noise_thresh = np.load('noise_thresh.npy')
mean_freq_noise = np.load('mean_freq.npy')
std_freq_noise = np.load('std_freq.npy')
noise_stft_db = np.load('noise_db.npy')


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)



def findNoiseThresh(
    noise_clip,n_fft = 2048,win_length=2048,hop_length=512,n_std_thresh=1.5
):    
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    return noise_thresh, mean_freq_noise, std_freq_noise,noise_stft_db
    
    
def removeNoise(
    audio_clip,
    noise_thresh,
    mean_freq_noise,
    std_freq_noise,
    noise_stft_db,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0
):
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




#Assuming Energy threshold upper than 30 dB
Threshold = 30

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
swidth = 2
Max_Seconds = 10

silence = True
FileNameTmp = 'test2.wav'
Time=0
all =[]





def GetStream(chunk):
    return stream.read(chunk) 
def rms(frame, bytestream = True):
    count = len(frame)/swidth
    if (bytestream):
        format = "%dh"%(count)
        # short is 16 bit int
        frame = struct.unpack(format, frame )

    sum_squares = 0.0
    for sample in frame:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms 
    rms = math.pow(sum_squares/count,0.5);
    return rms * 1000

def filter_stream (stream):
    #convert bytestream to 16bit PCM
    sig = np.frombuffer(stream, dtype='<i2').reshape(-1, CHANNELS)
    
    
   # Change shape and type for noise removal function
    sig = sig.T[0].astype('float')
    #GoPiGo noise removal
    output = removeNoise(
    audio_clip=sig,
    noise_thresh=noise_thresh,
    mean_freq_noise = mean_freq_noise,
    std_freq_noise = std_freq_noise,
    noise_stft_db = noise_stft_db,
    n_std_thresh=2,
    prop_decrease=0.95
    )
    return(output)
    
    
def WriteSpeech(WriteData):
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(FileNameTmp, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(WriteData)
    wf.close()

def KeepRecord(TimeoutSignal, LastBlock):
    all.append(LastBlock)
    for i in range(0, TimeoutSignal):
        try:
            data = GetStream(chunk)
        except:
            continue
        #I chage here (new Ident)
        all.append(data)

    print("end record after timeout")
    data = b''.join(all)
    print("write to File")
    WriteSpeech(data)
    silence = True
    Time=0
    listen(silence,Time)     

def listen(silence,with_filter = True):
    print("listening now...")
    while silence:
        try:
            input = GetStream(chunk)
            #print('new stream chunk')
        except:
            continue
        if (with_filter):
            filtered = filter_stream(input)
            filtered_tuple = tuple(filtered)
            rms_value = rms(filtered_tuple, bytestream = False)
        else:
            rms_value = rms(input, bytestream = True)
        print(rms_value)
        if (rms_value > Threshold):
            print ("hello doubladay, you should trigger recording here....")
            silence = False
            # enter record function here
        
        
p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = chunk)



listen(silence,with_filter = False)
    






