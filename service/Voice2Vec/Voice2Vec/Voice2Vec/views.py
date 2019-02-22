"""
Routes and views for the flask application.
"""

import os
from datetime import datetime
from flask import render_template, request
from werkzeug import secure_filename
from Voice2Vec import app
import numpy as np

from Voice2Vec.wav_reader import get_fft_spectrum

if __name__ == '__main__':
   app.run(debug = True)

model_path = os.path.join(os.getcwd(), 'base_model.h5')

from keras.models import load_model
base_model = load_model(model_path)

import json
db_path = 'database.json'

if not os.path.exists(db_path):
    with open(db_path, 'w') as fp:
        json.dump([], fp)
        
with open(db_path, 'r') as fp:
    database = json.load(fp)

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='For support and suggestions please write'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/upload')
def upload_file():
   return render_template('upload.html', upload_path = "http://localhost:80/uploader")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        if not request or not request.files:
            return 'no file was choosen'
    f = request.files['file']
	
    id = request.form['id']
    features = get_fft_spectrum(f.filename)
    embedding = np.squeeze(base_model.predict(features))
    database.append((id, np.random.rand(128).tolist()))
    with open(db_path, 'w') as fp:
        json.dump(database, fp)

    ids = [elem[0] for elem in database]
    numpied = np.asarray([elem[1] for elem in database])

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(numpied)
    found = neigh.kneighbors([np.random.rand(128)])
    distances = found[0][0].tolist()
    ids = [ids[index] for index in found[1][0].tolist()]

    for id, distance in zip(ids, distances):
        print((id, distance))
      
    return 'file uploaded successfully'
  
