#script to grab all .wav files recursively from Music directory
import fnmatch
import os
import shutil

song_lst = {}
for root, dirnames, filenames in os.walk('Music'): # recursive generator for Music directory
	for filename in fnmatch.filter(filenames, "*.wav"):
		song_path = os.path.join(root, filename)
		new_path = "/Users/ben/src/msan/adv_machineLearning/music_wav/" + filename
		if song_path not in song_lst: # no dups
			shutil.move(song_path, new_path)
			song_lst[song_path] = True