from audio_trial import *
from flask import Flask

app = Flask(__name__)
piu = pickle.load(open('/Users/ben/src/msan/adv_machineLearning/piu-piu/piu_obj1.pkl', 'r'))

@app.route('/')
def home_page():
    return 'hey fam'

@app.route('/predict')
def pred():
    global piu
    S = StreamSong(piu)
    return S.stream()



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8885)