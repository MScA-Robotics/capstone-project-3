import pyaudio
import wave
import audioop as ao
from datetime import datetime
import numpy as np

def record_chunk(record_secs = 5,
                 chans = 1,
                 samp_rate = 44100,
                 chunk = 4096,
                 dev_index = 0,
                 wav_output_filename = 'recording_{}.wav'.format(datetime.now().strftime('%m%d%Y%H%M%S'))): # name of .wav file

    form_1 = pyaudio.paInt16 # 16-bit resolution

    audio = pyaudio.PyAudio() # create pyaudio instantiation

    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    #input_device_index = dev_index,
                    input = True, \
                    input_device_index = dev_index, \
                    frames_per_buffer=chunk)
    print("recording")
    frames = []
    rms_values = np.empty(1,dtype=float)
    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk,exception_on_overflow = False)
        rms_value = ao.rms(data,2)
        print(rms_value)
        rms_values = np.append(rms_values,rms_value)
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

    # save rms data in text file
    rms_filename = 'rms.txt'
    np.savetxt(rms_filename,rms_values, delimiter = ',')

if __name__ == '__main__':

    record_chunk()
