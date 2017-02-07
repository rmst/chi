from collections import OrderedDict
import os
import json

import shutil
import tensorflow as tf
from time import time, sleep
import fnmatch
import random
import numpy as np
import io
import matplotlib
# matplotlib.use("agg")
from matplotlib import pyplot as plt
from matplotlib import figure
import subprocess
import socket
from threading import Thread
from chi.logger import logger

# local schema: http://<host>:<port>/exp/local/<path>/
# remote schema: ... /exp/ssh/<num>/<path>/

CONFNAME = 'experiment.chi'

class ndict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


class Server:
  def __init__(self, host, port):
    self.host = host
    self.port = port
    self.exps = {}

    self.experiments()  # initial run

  def experiments(self):
    ps = (os.path.dirname(f.path) for d in ['~', '~/.chi/experiments']
                                  for f in rcollect(d, 10) if f.name == CONFNAME)
    # print(list(ps))
    exps = self.exps.copy()
    res = []

    for p in ps:
      e = exps.pop(p, None) or Exp(p, self.host, self.port)
      res.append(e.update())
      self.exps.update({p: e})

    for e, v in exps.items():  # remaining (deleted) exps
      v.delete()
      del self.exps[e]

    return res

  def adde(self, p):
    self.exps.update({p: Exp(p, self.host, self.port)})

  def info(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].update()

  def trend(self, path):
    return self.exps[path].plot_trend()

  def delete(self, path):
    return self.exps[path].rm()

  def tensorboard(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].tensorboard()

  def shutdown(self):
    for e in self.exps:
      e.delete()


class Exp:
  def __init__(self, path, host, port):
    self.host = host
    self.port = port
    self.path = path  # path to exp folder
    self.data = ndict(timestamp=0)
    self.plot_t = time() + 2 * random.random()
    self.plot_cache = None
    self.tb = None
    self.tb_t = 0
    self.tb_port = None

    self.update()

  def update(self):
    if time() - self.data.timestamp < 2:
      return self.data

    conf = self.path + '/' + CONFNAME
    with open(conf) as fd:
      self.data.update(json.load(fd))

    is_new = 'rand' not in self.data
    alive = time() - os.path.getmtime(conf) < 15  # check heartbeat
    outdated = time() - self.plot_t > 10
    if is_new or (alive and outdated):
      # new rand causes frontend trends to update
      self.data.rand = random.randint(0, 1000000)

    status = 'dead' if not alive else 'running' if 't_start' in self.data else 'pending'
    self.data.update(status=status, path=self.path, timestamp=time(), host=self.host, port=self.port, hostid='local')

    return self.data

  def tensorboard(self):
    if not self.tb:
      self.tb_port = get_free_port(self.host)
      cmds = ['tensorboard', '--logdir', "{}".format(self.path), '--host', '0.0.0.0', '--port', str(self.tb_port)]
      print(' '.join(cmds))
      self.tb = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
      self.tb_t = time()
      Thread(target=self.tb_killer, daemon=True).start()
      Thread(target=self.tb_watcher, daemon=True).start()
      return dict(host=self.host, port=self.tb_port, new=True)

    else:
      self.tb_t = time()  # heartbeat
      # print('heartbeat')
      return dict(host=self.host, port=self.tb_port, new=False)

  def tb_watcher(self):
    print('watcher start')
    assert isinstance(self.tb, subprocess.Popen)
    print('1')
    outs, errs = self.tb.communicate()
    print('2')
    returncode = self.tb.returncode
    self.tb = None
    msg = 'tensorboard on {} for {} returned with code {}'.format(self.tb_port, self.path, returncode)
    if returncode == 0:
      logger.debug(msg)
    else:
      logger.warning(msg)
      logger.warning(outs)
      logger.warning(errs)
    print('watcher finish')

  def tb_killer(self):
    tb = self.tb
    while not tb.poll():
      if time() - self.tb_t > 60:
        print("killer")
        assert isinstance(tb, subprocess.Popen)
        tb.terminate()
        logger.debug('tensorboard for {} kill because timeout'.format(self.path))
        # break
      sleep(5)
    print('killer finish')

  def rm(self):
    # print('rm')
    self.delete()
    shutil.rmtree(self.path, ignore_errors=False)
    return {'nothing': None}

  def delete(self):
    if self.tb:
      self.tb.kill()

  def plot_trend(self):
    if time() - self.plot_t < 10 and self.plot_cache:  # cache
      return io.BytesIO(self.plot_cache)

    self.plot_t = time() + 2 * random.random()
    f, ax = plt.subplots()
    f.set_size_inches((15, 4))
    f.set_tight_layout(True)
    name, x = self.trend_data(['dashboard', 'loss', 'return'])
    pl, = ax.plot(x[:, 0], x[:, 1])
    assert isinstance(f, figure.Figure)
    f.legend([pl], [name])
    # ax.plot(i,r)
    sio = io.BytesIO()
    f.savefig(sio, format='png', dpi=60, transparent=True)

    self.plot_cache = sio.getvalue()
    sio.seek(0)
    plt.close(f)
    return sio

  def trend_data(self, keywords):
    import glob
    # a = glob.glob(path + '/**/*.tfevents.*', recursive=True)
    a = [f.path for f in rcollect(self.path, 3) if 'tfevents' in f.name ]
    target = None
    data = []
    for ef in a:
      if not data:
        for e in tf.train.summary_iterator(ef):
          for v in e.summary.value:
            if not target and any(k in v.tag for k in keywords):
              target = v.tag
            if target and target == v.tag:
              data.append((e.step, v.simple_value))

    if len(data) == 0:
      data = np.zeros([1, 2])
    return target, np.array(data)


# Util

def get_free_port(host):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((host, 0))
  port = sock.getsockname()[1]
  sock.close()
  return port


def rcollect(path, depth, filter=None):
  filter = filter or (lambda n: not n.startswith('.'))
  path = os.path.expanduser(path)
  if os.path.exists(path):
    for f in os.scandir(path):
      if filter(f.name):
        if f.is_file():
          yield f
        elif depth > 0:
          for e in rcollect(f.path, depth - 1, filter):
            yield e

