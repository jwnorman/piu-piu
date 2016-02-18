import glob, shutil, re

#since im manually converting mp3 to wav but selecting * in itunes, 
#I have to run this regex once again on the filenames because itunes
#reconverts them back to normal. 

for f in glob.glob('*.wav'):
	shutil.move(f, re.sub(r'^[0-9]+', '', f.strip()))