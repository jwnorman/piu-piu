#script to grab all .wav files recursively from Music directory
import fnmatch
import os, re
import shutil
import uuid

song_lst = {}
with open('song_meta.csv', 'w') as outfile:
	outfile.write('id, artist, album, song, format_song')
	outfile.write('\n')
	for root, dirnames, filenames in os.walk('/Users/ben/Desktop/Orig_songs/Music'): # recursive generator for Music directory
		for filename in fnmatch.filter(filenames, "*.mp3"):
			song_path = os.path.join(root, filename)
			new_filename = re.sub(r'^[0-9]+_', '', \
						   re.sub(r'mp3.*', '.mp3', \
				           re.sub(r'[(),.-]', '',  \
				           re.sub(r'\s', '_', filename))))

			new_path = "/Users/ben/src/msan/adv_machineLearning/music_wav/" + new_filename
			meta = [uuid.uuid4().hex]#id + \
			        root.split('/')[-2:]#artist, album + \
			       [re.sub(r'^[0-9]+', '', filename.strip(".mp3"))]#original song name + \
			       [new_filename.strip('.mp3')] #formatted song name
			       
			outfile.write(','.join(meta))
			outfile.write('\n')

			if song_path not in song_lst: # no dups
				shutil.move(song_path, new_path)
				song_lst[song_path] = True