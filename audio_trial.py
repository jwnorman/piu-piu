from __future__ import division
from scipy.io import wavfile
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn
import operator
import glob
import sys
#%matplotlib inline

class PredictSong(object):
    def __init__(self, song, num_predictions):
        self.song = song
        self.num_predictions = num_predictions
        self.counters = [Counter() for i in range(num_predictions)]
        self.props = [Counter() for i in range(num_predictions)]

    @staticmethod
    def song_streamer(song_segments):
        for i, song_segment in enumerate(song_segments):
            fd = abs(np.fft.fft(song_segment))
            freq = abs(np.fft.fftfreq(len(fd),1/float(44100))) # we may want to consider soft coding this
            yield fd, freq

class PiuHash(object):
    def __init__(self, window_length=1, bins=[[30,40,80,120,180,300]]):
        self.num_hashes = len(bins)
        self.piu_hash = [defaultdict(list) for i in xrange(self.num_hashes)]
        self.window_length = window_length
        self.bins = [self.get_lit(bin_itr) for bin_itr in bins]

    def hash_song(self, channel=None, meta=""):
        """ Hashes a song 

            channel: a list or numpy array of frequencies from a wav file
            meta: unique id of wav file
        """
        num_splits = np.ceil(len(channel)/float(44100*self.window_length))
        song_segments = np.array_split(channel, num_splits)
        for i, song_segment in enumerate(song_segments):
            fd = abs(np.fft.fft(song_segment))
            freq = abs(np.fft.fftfreq(len(fd),1/float(44100))) # we may want to consider soft coding this
            self.hash_seg(fd, freq, i, meta=meta)

    def hash_dir(self, directory, channel = None, meta=""):
        """ Reads and hashes each .wav song in the directory <dir>
        """
        files = glob.glob(directory + "/*.wav")
        n = len(files)
        i = 1
        for filename in files:
            sys.stdout.write('%.2f%%\r' % (i / n * 100))
            sys.stdout.flush()
            i += 1
            fs, data = wavfile.read(filename)
            channel1 = data[:,0]
            piu.hash_song(channel1, str(i))


    def hash_seg(self, fd, freq, i=None, test=False, meta=None):
        """ Hashes a fourier transformed segment of a song. 
        """
        ret = list()
        for hash_num in range(self.num_hashes):
            hash_temp = tuple([self.argmax_frequency(fd, freq, bin_itr) for bin_itr in self.bins[hash_num]])
            if not test: # building
                self.piu_hash[hash_num][hash_temp].append([i, meta])
            else: # testing
                ret.append(hash_temp)
        if test: return ret

    def predict(self, song):
        """
        song: the output from wavfile.read()
        """
        pred_song = PredictSong(song, self.num_hashes)
        channel1 = song[:,0]
        num_splits = np.ceil(len(channel1)/float(44100*self.window_length))
        song_segments = np.array_split(channel1, num_splits)
        streamer = pred_song.song_streamer(song_segments)

        match = False
        while not match:
            fd, freq = streamer.next()
            res = self.hash_seg(fd, freq, test=True) # [(24,48,80,111,200), (11,111,200)]
            for i, key in enumerate(res):
                pred_song.counters[i] += Counter([elem[1] for elem in self.piu_hash[i][key]]) # running sum
                pred_song.props[i] = {k: 1/sum(pred_song.counters[i].values()) for k,v in pred_song.counters[i].iteritems()} # proportion        
                max_key = max(pred_song.props[i].iteritems(), key=operator.itemgetter(1))[0]
                if pred_song.props[i][max_key] >= .8:
                    print pred_song.props[i]
                    return max_key
            print 'No matches found'
            return 0


    def print_hashes(self):
        for hash_num in range(self.num_hashes):
            print 'hash ' + str(hash_num) + ' with bins: ' + str(self.bins[hash_num])
            for k,v in self.piu_hash[hash_num].iteritems():
                print '\t' + str(k) + ':\t\t' + str(v)
            print ''

    @staticmethod
    def get_lit(bins):
        """ Given a list of breakpoints it returns tuples of start,end of each bin.
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

    piu = PiuHash(bins=[[30,40,80,120,180,300],[0, 100, 200, 300]])
    piu.hash_song(channel1, meta)
    piu.print_hashes()
    blah = piu.predict(data)
    print blah

if __name__ == '__main__':
    main()