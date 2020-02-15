import soundfile as sf
import sounddevice as sd

def record_chunk2(dur = 60, fs = 44100, chans = 1, output_file = 'recording.wav'):
    rec = sd.rec(int(dur*fs), samplerate=fs, channels=chans, blocking=True)
    sf.write(output_file, rec, fs)

if __name__ == '__main__':
    record_chunk2(5)
