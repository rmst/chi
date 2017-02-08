#!/usr/bin/env python3
from flask import Flask, Response, jsonify, send_from_directory, send_file
import flask
import chi
from chi.dashboard.server import get_free_port, Server
import requests
import socket
from chi.logger import logger


import os


@chi.app
def dashboard(host='127.0.0.1', port=5000, loglevel='debug'):
  chi.set_loglevel(loglevel)
  p = os.path.dirname(os.path.realpath(__file__))
  app = Flask(__name__, root_path=p, static_url_path='/')

  if port == 0:
    port = get_free_port(host)
  server = Server(host, port)

  remotes = {}  # spin up all ssh servers in a config

  @app.route("/")
  def index():
    return send_file("components/index.html")

  @app.route('/bower_components/<path:path>')
  def bower(path):
    return send_from_directory('bower_components', path)

  @app.route('/components/<path:path>')
  def comp(path):
    return send_from_directory('components', path)

  @app.route("/exp/")
  def exp():
    return send_file("components/experiment.html")

  @app.route("/info/<string:host>/<path:path>")  # experiment page
  def info(host, path):
    if host == 'local':
      return jsonify(server.info('/'+path))
    else:
      raise Exception('Remote not yet supported')
      # request remote info
      # update urls

  @app.route("/tb/<string:host>/<path:path>")
  def tb(host, path):
    if host == 'local':
      return jsonify(server.tensorboard('/'+path))
    else:
      raise Exception('Remote not yet supported')
      # make local port forward
      # request remote tensorboard
      # update urls

  @app.route("/delete/<path:path>")
  def delete(path):
    return jsonify(server.delete('/'+path))

  @app.route("/trend/<path:path>")
  def trend(path):
    sio = server.trend('/'+path)
    return send_file(sio,  attachment_filename='trend.png', mimetype='image/png')

  @app.route('/experiments')
  def experiments():
    s = server.experiments()

    # add remotes
    # update their hostids

    return jsonify(s)

  app.run(host=host, port=port)