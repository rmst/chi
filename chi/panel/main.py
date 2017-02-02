from flask import Flask, Response, jsonify, send_from_directory, send_file

import requests

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return send_file("index.html")


@app.route('/experiments')
def experiments():
  return jsonify([{'id': 1, 'name': 'ga', 'email': 'gu'}, {'id': 1, 'name': 'ga', 'email': 'gu'}, {'id': 1, 'name': 'ga', 'email': 'gu'}])

if __name__ == '__main__':
  app.run()