from __future__ import division
from scipy.io import wavfile
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn
import operator
import glob
import sys
import fnmatch
import os, re
import shutil
import uuid
import sounddevice as sd
#%matplotlib inline

class PredictSong(object):
    def __init__(self, song, piu_hash_obj):
        self.song = song
        self.piu_hash_obj = piu_hash_obj
        self.counters = [Counter() for i in range(piu_hash_obj.num_hashes)]
        self.props = [Counter() for i in range(piu_hash_obj.num_hashes)]

    def song_streamer(self, song_segments):
        for i, song_segment in enumerate(song_segments):
            yield self.get_fft(song_segment)
    
    @staticmethod
    def get_fft(song_segment):
        fd = abs(np.fft.fft(song_segment))
        freq = abs(np.fft.fftfreq(len(fd),1/float(44100))) # we may want to consider soft coding this
        return fd, freq

    def predict(self):
        """
        song: the output from wavfile.read()
        """
        channel1 = self.song[:,0]
        num_splits = np.ceil(len(channel1)/float(44100*self.piu_hash_obj.window_length))
        song_segments = np.array_split(channel1, num_splits)
        streamer = self.song_streamer(song_segments)
        match = False
        itr_num = 1
        while not match:
            try:
                fd, freq = streamer.next()
            except StopIteration as e:
                return 0
            match = self.predict_iteration(fd, freq, itr_num)
            itr_num += 1
    
    def predict_iteration(self, fd, freq, itr_num):
        res = self.piu_hash_obj.hash_segment(fd, freq, test=True) # [(24,48,80,111,200), (11,111,200)]
        print 'res: ' + str(res)
        print 'itr_num: ' + str(itr_num)
        for i, key in enumerate(res):
            print 'blah: ' + str(len(self.piu_hash_obj.piu_hash[i][key]))
            if len(self.piu_hash_obj.piu_hash[i][key]) > 0:
                print '\tblah: ' + str(len(self.piu_hash_obj.piu_hash[i][key]))
                self.counters[i] += Counter([elem[1] for elem in self.piu_hash_obj.piu_hash[i][key]]) # running sum
                self.props[i] = {k: self.counters[i][k]/sum(self.counters[i].values()) for k,v in self.counters[i].iteritems()} # proportion
                print '\t' + 'counters: ' + str(self.counters)
                print '\t' + 'proportions: ' + str(self.props)
                max_key = max(self.props[i].iteritems(), key=operator.itemgetter(1))[0]
                print ''
                if (self.props[i][max_key] >= .8) and (itr_num >= 5):
                    print '\t' + 'success'
                    print '\t\t' + 'proportions: ' + str(self.props[i])
                    print '\t\t' + 'max_key: ' + str(max_key)
                    print ''
                    return max_key
        #print 'No matches found'
        return False

class PiuHash(object):
    def __init__(self, window_length=1, bins=[[30,40,80,120,180,300]]):
        self.num_hashes = len(bins)
        self.piu_hash = [defaultdict(list) for i in xrange(self.num_hashes)]
        self.window_length = window_length
        self.bins = [self.get_lit(bin_itr) for bin_itr in bins]
        self.meta = {}

    def convert_and_load_songs(self, directory, filter_by= ''):
        for root, dirnames, filenames in os.walk(directory):
            leaves = fnmatch.filter(filenames, "*" + filter_by)
            for i, filename in enumerate(leaves): #only look at .mp3s for noe
                if re.match('.DS', filename):
                    continue
                #display progress
                sys.stdout.write('\r%d' % (i))
                sys.stdout.flush()
                #create unique id 
                _id = uuid.uuid4().hex

                song_path = os.path.join(root, filename) #make full accessible path
                formatted_path = os.path.join(root, re.sub(' ', '', filename))
                shutil.copy(song_path, os.path.join(root, formatted_path))

                new_filename = _id + '.wav'

                print 'converting {} to {}'.format(formatted_path, new_filename)
                #convert to wav and save as <uuid>.wav
                os.system('ffmpeg -i {} {}'.format(formatted_path, new_filename))
                #clean the song name but preserve spaces () etc, just get rid of whitespace, .mp3,
                #and random numbers that are appended to the front of some songs
                orig = re.sub(r'^[-0-9]+', '', filename.strip('.*'))
                artist, album = root.split('/')[-2:] #arist, album last two levels of root path
                song = orig.strip() # 'displayable' song #converting to wav eventually, dont need .mp3 

                meta = {'artist': artist, 'album':album,'song': song}
                self.meta[_id] = meta
                self.hash_song(new_filename, _id, meta)
                os.system('rm {}'.format(new_filename))
                os.system('rm {}'.format(os.path.join(root, formatted_path)))

    def hash_song(self, filename, uuid, meta=""):
        """ Hash one song

            channel: a list or numpy array of frequencies from a wav file
            meta: unique id of wav file
        """
        fs, data = wavfile.read(filename)
        try:
            channel = data[:,0]
        except:
            channel = data
        num_splits = np.ceil(len(channel)/float(44100*self.window_length))
        song_segments = np.array_split(channel, num_splits)
        for song_segment in song_segments:
            fd = abs(np.fft.fft(song_segment))
            freq = abs(np.fft.fftfreq(len(fd),1/float(44100))) # we may want to consider soft coding this
            self.hash_segment(fd, freq, i, meta=meta)

    def hash_segment(self, fd, freq, i=None, test=False, meta=None):
        ret = list()
        for hash_num in range(self.num_hashes):
            hash_temp = tuple([self.argmax_frequency(fd, freq, bin_itr) for bin_itr in self.bins[hash_num]])
            if not test: # building
                self.piu_hash[hash_num][hash_temp].append([i, meta])
            else: # testing
                ret.append(hash_temp)
        if test: return ret

    def print_hashes(self):
        for hash_num in range(self.num_hashes):
            print 'hash ' + str(hash_num) + ' with bins: ' + str(self.bins[hash_num])
            for k,v in self.piu_hash[hash_num].iteritems():
                print '\t' + str(k) + ':\t\t' + str(v)
            print ''

    @staticmethod
    def get_lit(bins):
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

    piu = PiuHash(bins=[[30,40,80,120,180,300],[0, 100, 200, 300]])
    piu.hash_song(channel1, meta)
    piu.print_hashes()
    pred = PredictSong(data, piu, 10)
    pred.predict()

if __name__ == '__main__':
    pass
    #main()
