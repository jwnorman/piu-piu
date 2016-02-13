from scipy.io import wavfile
#import wave
import numpy as np
import matplotlib.pyplot as plt

# creae
fs, data = wavfile.read("/Users/gab/Documents/MSAN/ML2/classy.wav")
#print 'fs', fs
#print 'data', data
#print len(data)
ch1 = list()
ch2 = list()
for dp in data:
	ch1.append(dp[0])
	ch2.append(dp[1])

#print len(ch1)

# creates one flat array
#x = np.fromfile(open('/Users/gab/Documents/MSAN/ML2/classy.wav'),np.int16)[24:]
#n = len(x)

#x = np.fromfile(open('/Users/gab/Downloads/06-Loro.mp3'),np.int16)[24:]

#plt.plot(x[0:88200])
plt.plot(ch1)
plt.show()


#X = wave.open('/Users/gab/Documents/MSAN/ML2/classy.wav')
#print X.Wave_read.getsampwidth()

fd = np.fft.fft(ch1[0:44100])
t = np.arange(len(ch1[0:44100]))
print t.shape[-1]
freq = np.fft.fftfreq(len(fd),1/float(t.shape[-1]))
print min(freq),max(freq)
#plt.plot(freq,fd)
#plt.show()

