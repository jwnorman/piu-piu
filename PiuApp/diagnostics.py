from __future__ import division
from audio_trial import *
import numpy as np
import glob
import pickle

"""
Performs model diagnostics using phash with 10 start points
"""

# load phash (meta still broken...getting a different error now)
try:
	HASH_PATH = 'new_hash/phash'
	h = open(HASH_PATH, 'r')
except IOError:
	print "Enter your hash path"

buckets = [[30,40,80,120,180,300],[0, 100, 200, 300], \
               [0, 350, 3000], np.arange(0, 3000, 100), [0, 200, 1000, 3000], \
               [300, 1000, 3000], [0, 300, 500, 1000, 2000, 3000], \
               [0, 100, 200, 400, 800, 1600, 3000]]
piu = PiuHash(bins=buckets)
piu.piu_hash = pickle.load(h)

def get_accuracy_standard(song_path):
	'''
	Predicts songs starting at true start-point
	'''
	results = []  # lst of True/False
	for song in glob.glob(song_path + '*.wav'):
		truth = song.split('/')[-1].rstrip('.wav')  # uuid string
		fs, data = wavfile.read(song)
    	P = PredictSong(data, piu)
    	results.append( (P.predict() == truth) )
	return sum(results) / len(results)

def get_accuracy_random_start(song_path):
	'''
	Predicts songs starting at random start-point
	'''
	result = []
	songs  = glob.glob(song_path + '*.wav')
	nsongs = len(songs)
	start_points = [int(np.random.random(1) * 44100 * np.random.randint(0, 30)) \
	                for i in range(nsongs)]
	for song in glob.glob(song_path + '*.wav'):
		truth = song.split('/')[-1].rstrip('.wav')
		fs, data = wavfile.read(song)
		# pop from start_points to get random start point 
		P = PredictSong(data[start_points.pop():], piu)
		result.append( (P.predict() == truth) )
	return sum(results) / len(results)


# standard accuracy is static and wont change
acc_standard = get_accuracy_standard('new_hash/')
print 'Static Accuracy: {}'.format(acc_standard)

# test random start-points 100 time and take average to 
# get better representation of performance. (will take awhile)
acc_random = 0
for i in range(100):
	acc_random_i = get_accuracy_random_start('new_hash/')
	acc_random += acc_random_i
acc_random /= 100

print 'Random Accuracy: {}'.format(acc_random)






















