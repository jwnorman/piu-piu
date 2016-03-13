from audio_trial import *
from flask import Flask, jsonify, render_template, request, send_file
'''
Flask Application that records and predicts songs without any 
page re-rendering. See <index.html> and <html_templ.html> for
formatting / jQuery code.

Album art images in static/ directory
'''

app = Flask(__name__)
with open('piu_LM_TAKE3.pkl', 'r') as f:
	piu = pickle.load(f)

@app.route('/loading')
def loading():
	'''
	This path is called by jQuery and returns the results on the home page (/)
	Displays "Listing..." + a GIF while the user waits for a prediction to render
	'''
	return jsonify(result = '''<!doctype html>
        <h5>
        <div id="spinner"
    	<i class="fa fa-spinner fa-spin fa-5x"></i>
        </div>
        </h5>
    	''')

@app.route('/predict_song')
def predict_song():
    '''
    This path is called by jQuery and returns the results on the home page (/)
    If the song has metadata, returns the prediction and associated album art.
    Returns the UUID otherwise or an error message if no match.
    '''
    S = StreamSong(piu, demo=True)
    prediction = S.stream()
    if not prediction:  # S.stream() == False
    	return jsonify(result='No Match Found :(')
    return jsonify(result=prediction)

@app.route('/')
def index():
	'''Home Page'''
	return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8885)