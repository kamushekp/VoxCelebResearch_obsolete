"""
This script runs the Voice2Vec application using a development server.
"""

from os import environ
from Voice2Vec import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    app.run(host='0.0.0.0', port=80, debug=True)
