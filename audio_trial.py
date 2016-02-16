from scipy.io import wavfile
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn
#%matplotlib inline

class PiuHash(object):
    def __init__(self, window_length=1, bins=[[30,40,80,120,180,300]]):
        self.num_hashes = len(bins)
        self.piu_hash = [defaultdict(list) for i in xrange(self.num_hashes)]
        self.window_length = window_length
        self.bins = [self.create_bin_tuples(bin_itr) for bin_itr in bins]

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
            freq = abs(np.fft.fftfreq(len(fd),1/float(44100))) # we may want to consider soft coding this
            for hash_num in range(self.num_hashes):
                hash_temp = tuple([self.argmax_frequency(fd, freq, bin_itr) for bin_itr in self.bins[hash_num]])
                self.piu_hash[hash_num][hash_temp].append([i, meta])

    def print_hashes(self):
        for hash_num in range(self.num_hashes):
            print 'hash ' + str(hash_num) + ' with bins: ' + str(self.bins[hash_num])
            for k,v in self.piu_hash[hash_num].iteritems():
                print '\t' + str(k) + ':\t\t' + str(v)
            print ''

    @staticmethod
    def create_bin_tuples(bins):
        """
        input: [30,40,80,120,180,300]
        output: [(30,40), (40,80), (80,120), (120,180), (180,300)]
        """
        bins = zip(bins, np.roll(bins, -1))
        bins.pop()
        return bins

    @staticmethod
    def argmax_frequency(fd, freq, bin_itr):
        relative_argmax = np.argmax(fd[bin_itr[0]:bin_itr[1]])
        return int(freq[bin_itr[0] + relative_argmax])

def main():
    try:
        fs, data = wavfile.read("/Users/gab/Documents/MSAN/ML2/classy.wav")
    except:
        try:
            fs, data = wavfile.read("/Users/jacknorman1/Documents/USF/MSAN/Module3/ML2/Project/el_nino.wav")
        except:
            fs, data = wavfile.read("/Users/ben/src/msan/adv_machineLearning/music_wav/el_nino.wav")
    channel1 = data[:,0]
    meta = 'ARTIST_SONG_ID'
    # plt.plot(channel1)
    # plt.show()

    piu = PiuHash(bins=[[30,40,80,120,180,300],[0, 100, 200, 300]])
    piu.hash_song(channel1, meta)
    piu.print_hashes()

if __name__ == '__main__':
    main()