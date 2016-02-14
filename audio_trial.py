from scipy.io import wavfile
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn
#%matplotlib inline

class PiuHash(object):  
    def __init__(self, window_length=1, bins=[30,40,80,120,180,300]):
        self.piu_hash = defaultdict(list)
        self.window_length = window_length
        self.bins = bins
        self.bins = zip(self.bins, np.roll(self.bins, -1))
        self.bins.pop()

    def hash_song(self, channel=None, meta=""):
        """ Hash one song

            channel: a list or numpy array of frequencies from a wav file
            meta: unique id of wav file
        """

        num_splits = np.ceil(len(channel)/float(44100*self.window_length))
        song_segments = np.array_split(channel, num_splits)
        for i, song_segment in enumerate(song_segments):
            fd = abs(np.fft.fft(song_segment))
            t = np.arange(len(song_segment))
            freq = abs(np.fft.fftfreq(len(fd),1/float(44100)))
            hash_temp = tuple([self.argmax_frequency(fd, freq, bin) for bin in self.bins])
            self.piu_hash[hash_temp].append([i, meta])

    @staticmethod
    def argmax_frequency(fd, freq, bin):
        relative_argmax = np.argmax(fd[bin[0]:bin[1]])
        return int(freq[bin[0] + relative_argmax])

def main():
    try:
        fs, data = wavfile.read("/Users/gab/Documents/MSAN/ML2/classy.wav")
    except:
        fs, data = wavfile.read("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/classy.wav")
    channel1 = data[:,0]
    meta = 'ARTIST_SONG_ID'
    # plt.plot(channel1)
    # plt.show()

    piu = PiuHash()
    piu.hash_song(channel1, meta)
    print piu.piu_hash

if __name__ == '__main__':
    main()