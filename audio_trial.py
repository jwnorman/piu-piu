from scipy.io import wavfile
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn
#%matplotlib inline

fs, data = wavfile.read("/Users/gab/Documents/MSAN/ML2/classy.wav")
ch1 = list()
ch2 = list()
for dp in data:
	ch1.append(dp[0])
	ch2.append(dp[1])

plt.plot(ch1)
plt.show()

plt.plot(ch2)
plt.show()

piu_hash = defaultdict(list)

# need to do this for each song:
meta = 'ARTIST/SONG'
i = 0
for sec in np.array_split(ch1, np.ceil(len(ch1)/float(44100))):
    fd = abs(np.fft.fft(sec))
    t = np.arange(len(sec))
    freq = abs(np.fft.fftfreq(len(fd),1/float(44100)))
    #plt.plot(freq[30:300],fd[30:300])
    #plt.show()
    a = int(freq[30+np.argmax(fd[30:40])])
    b = int(freq[40+np.argmax(fd[40:80])])
    c = int(freq[80+np.argmax(fd[80:120])])
    d = int(freq[120+np.argmax(fd[120:180])])
    e = int(freq[180+np.argmax(fd[180:300])])
    piu_hash[(a,b,c,d,e)].append([i,meta])
    i += 1


#piu_hash