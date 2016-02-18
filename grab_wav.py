#script to grab all .wav files recursively from Music directory
import fnmatch
import os, re
import shutil
import uuid

song_lst = {}
with open('song_meta.tsv', 'w') as outfile:
	#create headers for tsv file
	outfile.write('\t'.join(['id', 'artist', 'album', 'song', 'format_song']))
	outfile.write('\n')

	#Grabbing all mp3 files that lie within 'Music' directory
	for root, dirnames, filenames in os.walk('/Users/ben/Desktop/Orig_songs/Music'): # recursive generator for Music directory
		for filename in fnmatch.filter(filenames, "*.mp3"):
			song_path = os.path.join(root, filename)

			#clean the song name to make it programmable
			new_filename = re.sub(r'^[0-9]+_', '', \
						   re.sub(r'mp3.*', '.mp3', \
				           re.sub(r'[(),.-]', '',  \
				           re.sub(r'\s', '_', filename)))).strip()


			#clean the song name but preserve spaces () etc, just get rid of whitespace, .mp3,
			#and random numbers that are appended to the front of some songs
			orig = re.sub(r'^[-0-9]+', '', filename.strip(".mp3"))

			#generating unique id for each song with uuid.uuid4()
			_id = uuid.uuid4().hex
			artist, album = root.split('/')[-2:] #arist, album last two levels of root path
			song = orig.strip() 
			format_song = new_filename.strip('.mp3') #converting to wav eventually, dont need .mp3 


			new_path = "/Users/ben/src/msan/adv_machineLearning/music_wav/" + format_song

			meta = [_id, artist, album, song, format_song]
			print meta
			outfile.write('\t'.join(meta))
			outfile.write('\n')

			#code to move files initially, using this script to test formatting
			if song_path not in song_lst: # no dups
				shutil.move(song_path, new_path + '.wav')
				song_lst[song_path] = True