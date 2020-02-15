from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

fs, data = wavfile.read('./test2.wav')

print(fs)


#plt.plot(data[100:200])
#plt.show()
