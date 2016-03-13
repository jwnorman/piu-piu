"""
Contains PiuHash and PredictSong classes. 
Music Recognition Software
Piu-Piu, Inc. @MSAN-USF
"""

from __future__ import division
from scipy.io import wavfile
import numpy as np
from collections import defaultdict, Counter
import operator
import glob
import sys
import fnmatch
import os
import re
import time
import shutil
import uuid
import pipes
import sounddevice as sd
import pickle


__author__ = "Gabby Corbett, Ben Miroglio, Jack Norman"


class StreamSong(object):
    def __init__(self, piu_hash_obj, samplerate=44100, timeout_limit=15, threshold=.7,\
                 viz=False, device=0, truth=None, demo=False):
        self.samplerate = samplerate
        self.timeout_limit = timeout_limit
        self.timeout_counter = 1
        self.timeout_flag = False
        self.finished_flag = False
        self.threshold = threshold
        self.viz = viz
        self.device = device
        self.demo = demo
        self.predictions = PredictSong(None, piu_hash_obj, threshold=self.threshold, demo=self.demo)
        self.truth = truth


    def callback(self, indata, frames, time, status):
        song_segment = indata[:,0]
        fd, freq = self.predictions.get_fft(song_segment)
        self.finished_flag = self.predictions.predict_iteration(fd, freq, self.timeout_counter, \
                                                                self.truth)
        if self.timeout_counter == self.timeout_limit:
            self.timeout_flag = True
        else:
            self.timeout_counter += 1
        
    def stream(self):
        sdis = sd.InputStream(channels=2,
            blocksize=44100,
            samplerate=self.samplerate,
            callback=self.callback,
            device = self.device)
        sdis.start()
        while (not self.timeout_flag) and (not self.finished_flag):
            pass
        sdis.stop()
        if self.truth is not None:
            print self.predictions.bin_results
        return self.finished_flag



class PredictSong(object):
    def __init__(self, song, piu_hash_obj, threshold=.7, demo=False):
        self.song = song
        self.piu_hash_obj = piu_hash_obj
        self.counters = [Counter() for i in range(piu_hash_obj.num_hashes)]
        self.props = [Counter() for i in range(piu_hash_obj.num_hashes)]
        self.hash_counter = Counter()
        self.threshold = threshold
        self.demo = demo
        self.bin_results = defaultdict(Counter)

    def song_streamer(self, song_segments):
        """
        Takes song_segments and retrieves fd and freq 
        for one segment at a time (generator)
        """
        for i, song_segment in enumerate(song_segments):
            yield self.get_fft(song_segment)

    @staticmethod
    def get_fft(song_segment):
        """Performs Fast Fourier Transform on a song segment"""
        fd = abs(np.fft.fft(song_segment))
        freq = abs(np.fft.fftfreq(len(fd), 1/float(44100)))
        return fd, freq

    def predict(self):
        """
        Segments/hashes the 1st channel in song, 
        compares to train hashes
        """
        try:
            channel1 = self.song[:,0]
        except:
            channel1 = self.song

        num_splits = np.ceil(len(channel1)/float(44100*self.piu_hash_obj.window_length))
        song_segments = np.array_split(channel1, num_splits)
        streamer = self.song_streamer(song_segments)
        match = False
        itr_num = 1
        while itr_num is not match:
            print 'num_iter', itr_num
            try:
                fd, freq = streamer.next()
            except StopIteration as e:
                pass
            match= self.predict_iteration(fd, freq, itr_num)
            itr_num += 1
            if match:
                return match
        return False
    
    def predict_iteration(self, fd, freq, itr_num, truth=None):
        """
        Takes fft info and checks train hashes, updating 
        counter for the test instance
        """
        res = self.piu_hash_obj.hash_segment(fd, freq, test=True)
        print 'Iteration Number:    ' + str(itr_num)
        for i, key in enumerate(res):
            val = 0
            if len(self.piu_hash_obj.piu_hash[i][key]) > 0:
                self.hash_counter[i] += 1
                self.counters[i] += Counter([elem[0] \
                                    for elem in self.piu_hash_obj.piu_hash[i][key]]) # running sum
                self.props[i] = {k: self.counters[i][k]/sum(self.counters[i].values()) \
                                    for k,v in self.counters[i].iteritems()} # proportion

                max_key = max(self.props[i].iteritems(), key=operator.itemgetter(1))[0]
                if (self.props[i][max_key] >= self.threshold) and (itr_num >= 5):
                    if truth:
                            if max_key == truth:
                                val = 1
                            if (max_key != truth):
                                val = -1
                    else:
                        if self.demo:
                            try:
                                return self.piu_hash_obj.meta[max_key]
                            except KeyError as e:
                                return max_key
                        return max_key
            self.bin_results[i][val] += 1
        if truth is not None:
            return False


class PiuHash(object):
    def __init__(self, window_length=1, bins=[[30,40,80,120,180,300]]):
        self.num_hashes = len(bins)
        self.piu_hash = [defaultdict(list) for i in xrange(self.num_hashes)]
        self.window_length = window_length
        self.bins = [self.get_lit(bin_itr) for bin_itr in bins]
        self.meta = {}

    def convert_and_load_songs(self, directory, filter_by=''):
        """
        Takes a 'Music' Directory and recursively finds 
        all files underneath, grabbing all associated 
        meta data. Converts songs to wav and adds to PiuHash
        """
        i = 1
        for root, dirnames, filenames in os.walk(directory):
            leaves = fnmatch.filter(filenames, "*" + filter_by)
            for i, filename in enumerate(leaves):  # only look at .mp3s for noe
                if re.match('.DS', filename):
                    continue
                # display progress
                sys.stdout.write('\rPROCESSING: %s, file #: %d' % (' - '.join(root.split('/')[-2:]), i))
                sys.stdout.flush()

                # create unique id 
                _id = uuid.uuid4().hex

                song_path = os.path.join(root, filename)  # make full accessible path
                formatted_path = os.path.join(root, re.sub(' ', '', filename))
                shutil.move(song_path, os.path.join(root, formatted_path))

                new_filename = _id + '.wav'
                # print 'converting {} to {}'.format(formatted_path, new_filename)

                # convert to wav and save as <uuid>.wav
                os.system('ffmpeg -loglevel quiet -i {} {}'.format(pipes.quote(formatted_path), new_filename))

                # clean the song name but preserve spaces () etc, just get rid of whitespace, .mp3,
                # and random numbers that are appended to the front of some songs
                orig = re.sub(r'^[-0-9]+', '', filename.strip('.*'))
                artist, album = root.split('/')[-2:]
                song = orig.strip()

                meta = {'artist': artist, 'album':album, 'song': song}
                self.meta[_id] = meta
                self.hash_song(new_filename, _id, meta)
                pickled_hash = open('phash', 'w')
                pickled_meta = open('pmeta', 'w')
                pickle.dump(self.piu_hash, pickled_hash)
                pickle.dump(self.meta, pickled_meta)
                i += 1

    def hash_song(self, filename, uuid, meta=""):
        """ 
        Take a path to a file, reads into wav 
        and adds to hash
        """
        fs, data = wavfile.read(filename)
        try:
            channel0 = data[:,0]
        except:
            channel0 = data

        # create 10 different windows, starting at begining of song, 1/10 into first second, 
        # and 2/10, into first second, etc. -- creates 10 overlapping windows
        windows = [channel0]
        for i in range(1, 10):
            windows.append(channel0[int(44100/10)*i:])
        for channel in windows:
            num_splits = np.ceil(len(channel)/float(44100*self.window_length))
            song_segments = np.array_split(channel, num_splits)
            for song_segment in song_segments:
                fd = abs(np.fft.fft(song_segment))
                freq = abs(np.fft.fftfreq(len(fd), 1/float(44100)))
                self.hash_segment(fd, freq, uuid, meta=meta)

    def hash_segment(self, fd, freq, uuid=None, test=False, meta=None):
        """helper function for hash_song"""
        ret = list()
        for hash_num in range(self.num_hashes):
            hash_temp = tuple([self.argmax_frequency(fd, freq, bin_itr) \
                                for bin_itr in self.bins[hash_num]])
            if not test: # building
                self.piu_hash[hash_num][hash_temp].append([uuid, meta])
                self.meta[uuid] = meta
            else: # testing
                ret.append(hash_temp)
        if test: return ret

    def print_hashes(self):
        """pretty print for hashes"""
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


def load_new_song(song_path, uuid, meta, 
                  piu_path ='/Users/ben/src/msan/adv_machineLearning/piu-piu/piu_obj.pkl'):
    '''
    Loads a new song into the Hash and Meta attributes of the pickled piu
    object stored in <piu_path> and rewrites the obj to a pickle with the updated
    data
    '''
    piu = pickle.load(open(piu_path, 'r'))
    piu.hash_song(song_path, 
                  uuid = 'riri', 
                  meta = meta)
    pickle.dump(piu, open(song_path, 'w'))