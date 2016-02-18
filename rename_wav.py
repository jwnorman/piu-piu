import glob, shutil, re

#since im manually converting mp3 to wav but selecting * in itunes, 
#I have to run this regex once again on the filenames because itunes
#reconverts them back to normal. 

for f in glob.glob('*.wav'):
	new_filename = re.sub(r'^[0-9-]+_', '', \
						   re.sub(r"[\\'()-]", '', 
				           re.sub(r"[\\[],]", '',  \
				           re.sub(r'\s', '_', f)))).strip()
	print new_filename
	shutil.move(f, new_filename)