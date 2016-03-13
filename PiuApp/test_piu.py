import glob
import pickle
import pytest
import numpy as np
from audio_trial import *


# runs with the following command:
# $python -m pytest test_piu.py

# prepare piu object
print "Loading hash and meta into Piu Object..."
h = open('new_hash/phash', 'r')
m = open('wav_songs_10/pmeta', 'r')
buckets = [[30,40,80,120,180,300],[0, 100, 200, 300], \
               [0, 350, 3000], np.arange(0, 3000, 100), [0, 200, 1000, 3000], \
               [300, 1000, 3000], [0, 300, 500, 1000, 2000, 3000], \
               [0, 100, 200, 400, 800, 1600, 3000]]
piu = PiuHash(bins=buckets)
piu.piu_hash = pickle.load(h)
# piu.meta = pickle.load(m)


three_song_path = './wav_songs_samp/*.wav'
ten_hash_path   = './new_hash/*.wav'

SONG_PATH = ten_hash_path

# add riri to hash
# riri = '/Users/ben/Desktop/same_dir/same_ole_mistakes.wav'
# piu.hash_song(riri, uuid = 'riri', meta= {'song':'Same Ole Mistakes', 'artist':'Rihanna'})

def _test_songs(path, split='/'):
	'''
	NOT a test function (hence the '_'), general function to test a directory
	of songs. 
	'''
	for f in glob.glob(path):
		print f
		truth = f.split(split)[-1].strip('.wav')  # uuid string
		fs, data = wavfile.read(f)
    	P = PredictSong(data, piu)
    	assert P.predict() == truth

@pytest.mark.skipif(False, reason='Time')  # skips if first arg is True
def test_clean_songs():
	"""
	Testing 110 songs in unaltered state
	"""
	path = SONG_PATH
	for f in glob.glob(path):
		print f
		truth = f.split('/')[-1].rstrip('.wav')  # uuid string
		fs, data = wavfile.read(f)
    	P = PredictSong(data, piu)
    	assert P.predict() == truth

@pytest.mark.skipif(True, reason='Time')  # skips if first arg is True
def test_noise_songs():
	'''
	Testing 3 songs with added noise, timing is not affected
	'''
	_test_songs('./noise_songs/*', split='_')

@pytest.mark.skipif(True, reason = 'too much variance')
def test_noise_songs_random_start_point():
	songs  = glob.glob('./noise_songs/*.wav')
	nsongs = len(songs)
	start_points = [int(np.random.random(1) * 44100) for i in range(nsongs)]
	result = []
	for f in songs:
		truth = f.split('_')[-1].strip('.wav')  # uuid string 
		fs, data = wavfile.read(f)
		# pop from start_points to get random start point 
		P = PredictSong(data[start_points.pop():], piu)
		result.append(P.predict() == truth)
	assert sum(result) / float(len(result)) > 0

@pytest.mark.skipif(False, reason='Time')  # skips if first arg is True
def test_random_start_point(thres=.8):  # iterate over grid of threshold to find best
	'''
	Tests 110 songs that do not start exactly on time using
	random start points, returns accuracy. Results will vary
	due to randomness
	'''
	songs  = glob.glob(SONG_PATH)
	nsongs = len(songs)
	start_points = [int(np.random.random(1) * 44100) for i in range(nsongs)]
	result = []
	for f in glob.glob(SONG_PATH):
		truth = f.split('/')[-1].strip('.wav')  # uuid string 
		fs, data = wavfile.read(f)
		# pop from start_points to get random start point 
		P = PredictSong(data[start_points.pop():], piu, threshold=thres)
		result.append(P.predict() == truth)
	assert sum(result) / float(len(result)) == 1









