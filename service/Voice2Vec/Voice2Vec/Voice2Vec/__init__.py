"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

import Voice2Vec.views
import Voice2Vec.constants
import Voice2Vec.sigproc
import Voice2Vec.wav_reader
