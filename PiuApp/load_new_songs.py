import uuid
import pickle
from audio_trial import *
import glob

with open('piu_LM_TAKE2.pkl', 'r') as pkl:
	piu = pickle.load(pkl)

def load_song(filename, artist, album, song, album_art):
	global piu
	_uuid = uuid.uuid4().hex
	piu.hash_song(filename , uuid = _uuid)
	piu.meta[_uuid] = '''
	<!doctype html>
	<title></title>

	<p> Artist: {} </p>
	<p> Album:  {} </p>
	<p> Song:   {} </p>
	<img src="/static/{}" alt="album art"
	style="margin-left:0px;width:160px;height=160">
	'''.format(artist, album, song, album_art)

def load_meta(uuid, artist, album, song, album_art):
	global piu
	piu.meta[uuid] = '''
	<!doctype html>
	<title></title>

	<p> Artist: {} </p>
	<p> Album:  {} </p>
	<p> Song:   {} </p>
	<img src="/static/{}" alt="album art"
	style="margin-left:0px;width:160px;height=160">
	'''.format(artist, album, song, album_art)

# load_song('/Users/ben/Desktop/hotline.wav', 'Drake', 'Hotline Bling', 'Hotline Bling', 'HLB.jpeg')
# load_meta('83cd229138c940bfba2245836729e84f.wav', 'Stephen Malkmus and the Jicks', 'Pig Lib', 'Animal Midnight', 'am.jpeg')
# load_meta('9120894b0bea4561948da446732f6b2a.wav', 'The Helio Sequence', 'Keep Your Eyes Ahead', 'Keep Your Eyes Ahead', 'helio.jpg')
# load_song('/Users/ben/Desktop/2ne.wav', '2NE1', 'Unknown Album', 'Fire', '2ne1.jpg')
# load_song('/Users/ben/Desktop/jack.wav', 'Jack Norman', 'Biz Strat Class', "Jack's Morning", 'j.jpg')
# load_song('/Users/ben/Desktop/st.wav', 'Ohio Players', 'Skin Tight', 'Skin Tight', 'st.jpeg')
# buckets = [[30,40,80,120,180,300], [100, 300, 500, 1000, 3000], \
#            np.arange(0, 3000, 100), [300, 1000, 3000], [800, 1600, 3200], \
#            [1000, 2000, 3000], [2000, 3000, 4000], [500, 1000, 1500, 2000, 2500, 3000]]

print'led'
load_song('/Users/ben/Desktop/zep.wav', 'Led Zeppelin', 'Led Zeppelin III', 'Immigrant Song', 'zep.jpeg')
print 'rih'
load_song('/Users/ben/Desktop/umbrella.wav', 'Rihanna', 'Good Girl Gone Bad: Reloaded', 'Umbrella', 'rihanna.jpg')
# print 'adelle'
# load_song('/Users/ben/Desktop/adelle.wav', 'Adele', '25', 'Hello', 'adelle.jpeg')



# piu = PiuHash(bins = buckets)

# for f in glob.glob('/Users/ben/Desktop/samp_new_buckets/*'):
# 	piu.hash_song(f, uuid = f.split('/')[-1], meta='')




with open('piu_LM_TAKE4.pkl', 'w') as fout:
	pickle.dump(piu, fout)
	
print 'done'

