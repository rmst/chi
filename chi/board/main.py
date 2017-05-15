#!/usr/bin/env python
import json
from typing import NamedTuple

from os.path import join

import chi
from chi.board import MAGIC_PORT, CHIBOARD_HOME


# chi.set_loglevel('debug')


@chi.experiment(start_chiboard=False, default_logdir=CHIBOARD_HOME)
def chiboard(self: chi.Experiment, host='localhost', port=MAGIC_PORT, rootdir='',
             loglevel='debug', timeout=24*60*60, port_pool=""):
  from flask import Flask, jsonify, send_from_directory, send_file
  from chi.board.server import Server
  from chi.board.util import rcollect
  from chi.board.util import get_free_port
  from chi.logger import logger

  import os
  import signal
  from time import time, sleep
  from threading import Thread
  from os.path import expanduser as expandu
  from flask_socketio import SocketIO

  def expanduser(p):
    pa = expandu(p)
    return pa if pa.startswith('/') else '/' + pa

  chi.set_loglevel(loglevel)

  if port == 0:
    port = get_free_port(host)
    print(f'{port}')

  self.config.port = port

  p = os.path.dirname(os.path.realpath(__file__))
  app = Flask(__name__, root_path=p, static_url_path='/')

  socketio = SocketIO(app)

  if rootdir == '':
    import os
    rootdir = os.environ.get('CHI_EXPERIMENTS') or '~'
    logger.debug('Rootdir: ' + rootdir)

  if port_pool:
    port_pool = [int(p) for p in port_pool.split(',')]
  else:
    port_pool = range(port + 1, port + 30)

  server = Server(host, port, rootdir, port_pool)

  remotes = []
  p = expanduser('~/.chi/board/remotes.json')
  if os.path.exists(p):
    with open(p) as f:
      remotes = json.load(f)
      # print(remotes)

  state = dict(last_request=time())

  def killer():
    while time() - state['last_request'] < timeout:
      sleep(2)
    logger.error('timeout')
    os.kill(os.getpid(), signal.SIGINT)  # kill self

  Thread(target=killer, daemon=True).start()

  @app.before_request
  def tick():
    state.update(last_request=time())

  @app.route("/")
  def index():
    return send_file("components/index.html")

  @app.route("/favicon")
  def favicon():
    return send_file("components/favicon.png")

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
      return jsonify(server.info(expanduser(path)))
    else:
      raise Exception('Remote not yet supported')
      # request scripts info
      # update urls

  @app.route("/logs/<path:path>")
  def logs(path):
    data = []

    def key(x):
      k = '_' if x == 'stdout' else x
      return k

    path = expanduser(path) + '/logs'

    for p in sorted(os.listdir(path), key=key):
      with open(path + '/' + p, 'r') as f:
        f.seek(0, os.SEEK_END)
        l = f.tell()
        f.seek(max((0, l - 50000)), 0)
        c = f.read()
        while c and c[-1] == '\n':
          c = c[:-1]
        # c = c.replace('\n', '<br>')
        # c = c.replace('<', '&lt;')
        data.append({'name': os.path.basename(p), 'content': c})

    return jsonify(data)

  @app.route("/tb/<string:host>/<path:path>")
  def tb(host, path):
    if host == 'local':
      return jsonify(server.tensorboard(expanduser(path)))
    else:
      raise Exception('Remote not yet supported')
      # make local port forward
      # request scripts tensorboard
      # update urls

  @app.route("/delete/<path:path>")
  def delete(path):
    return jsonify(server.delete(expanduser(path)))

  @app.route("/trend/<path:path>")
  def trend(path):
    sio = server.trend('/' + path)
    return send_file(sio, attachment_filename='trend.png', mimetype='image/png')

  @app.route("/<string:cmd>/<path:path>")
  def command(cmd, path):
    return jsonify(server.command(cmd, expanduser(path)))

  try:
    socketio.on_namespace(server)
    socketio.run(app, host=host, port=port, log_output=loglevel == 'debug')
  finally:
    server.shutdown()
