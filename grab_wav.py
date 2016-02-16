#script to grab all .wav files recursively from Music directory
import fnmatch
import os, re
import shutil

song_lst = {}
for root, dirnames, filenames in os.walk('Music'): # recursive generator for Music directory
	for filename in fnmatch.filter(filenames, "*.wav"):
		song_path = os.path.join(root, filename)
		new_filename = re.sub(r'^[0-9]+_', '', \
					   re.sub(r'wav.*', '.wav', \
			           re.sub(r'[(),.-]', '',  \
			           re.sub(r'\s', '_', filename))))

		new_path = "/Users/ben/src/msan/adv_machineLearning/music_wav/" + new_filename
		print new_filename
		if song_path not in song_lst: # no dups
			shutil.move(song_path, new_path)
			song_lst[song_path] = True