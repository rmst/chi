import glob
import io
import json
import os
import random
import subprocess
from threading import Thread
from time import time, sleep

import requests
import shutil

from chi.board import repeat_until
from chi.experiment import CONFIG_NAME
from chi.logger import logger
from chi.board.util import ndict, Repo, get_free, rcollect

import numpy as np
import tensorflow as tf


class Experiment(Repo):

  def __init__(self, path, host, port, server):
    Repo.__init__(self, [path])

    self.server = server
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

  def on_modified(self, event):
    if event.src_path == os.path.join(self.path, CONFIG_NAME):
      logger.debug(f'{event.src_path} modified')
      self.server.upd()

  def update(self):
    if time() - self.data.timestamp < 2:
      return self.data

    conf = self.path + '/' + CONFIG_NAME

    alt = '/tmp/chi_' + os.environ['USER']
    self.data.update(path=self.path,
                     alternate_path=self.path.replace(os.path.expanduser('~/.chi'), alt),
                     timestamp=time(),
                     host=self.host,
                     port=self.port,
                     hostid='local',
                     jupyter_port=self.server.jupyter_port)

    try:
      with open(conf) as fd:
        self.data.update(json.load(fd))
    except (OSError, json.JSONDecodeError):
      return self.data

    is_new = 'rand' not in self.data
    alive = time() - os.path.getmtime(conf) < 70  # check heartbeat
    outdated = time() - self.plot_t > 10
    if is_new or (alive and outdated):
      # new rand causes frontend trends to update
      self.data.rand = random.randint(0, 1000000)

    status = self.data.get('status', 'dead')
    if status != 'dead' and not alive:
      status = 'dead'

    self.data.update(status=status)

    return self.data

  def command(self, cmd):
    import signal
    if cmd == "kill":
      slurm = self.data.get('slurm')
      if slurm:
        jid = slurm.get('SLURM_JOB_ID') or slurm.get('SLURM_JOBID')
        r = subprocess.check_call(('scancel', str(jid)))
        assert r == 0
        return {}
      pid = self.data.get('pid')
      if pid:
        logger.debug('send kill to '+str(pid))
        os.kill(pid, signal.SIGTERM)
      return dict()

    if cmd == "run":
      from chi.util import run_daemon
      e = self.data.get('sys_executable')
      a = self.data.get('sys_argv')
      k = self.data.get('args')


      if e and a and k:
        run_daemon(a[0], k, e)
        return dict()
      else:
        logger.debug('run failed because of exec ' + str(e) + ' argv ' + str(a) + ' args ' + str(k))
        return dict()

  def tensorboard(self):
    has_event_files = glob.glob(self.path+'**/*.tfevents*', recursive=True)
    if not has_event_files:
      return dict(no_event_files=True)

    elif not self.tb:
      self.tb_port = get_free(self.server.port_pool)
      cmds = ['tensorboard', '--logdir', "{}".format(self.path), '--host', '0.0.0.0', '--port', str(self.tb_port)]
      logger.debug('Start tensorboard with: ' + ' '.join(cmds))
      self.tb = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
      Thread(target=self.tb_watcher, daemon=True).start()

      @repeat_until(timeout=6.)
      def check_tb():
        try:
          url = "http://{}:{}".format(self.host, self.tb_port)
          r = requests.get(url)  # requests.head not supported by tensorboard
          available = r.status_code == 200
          sleep(.3)
          logger.debug('tb on {} status {}, {}'.format(url, r.status_code, r.reason))
          return available
        except requests.ConnectionError:
          return False

      if not check_tb:
        logger.warning('tb could not be started')

      self.tb_t = time()
      Thread(target=self.tb_killer, daemon=True).start()
      return dict(host=self.host, port=self.tb_port, new=True, available=check_tb, no_event_files=False)

    else:
      self.tb_t = time()  # heartbeat
      # print('heartbeat')
      return dict(host=self.host, port=self.tb_port, new=False, available=True, no_event_files=False)

  def tb_watcher(self):
    assert isinstance(self.tb, subprocess.Popen)
    outs, errs = self.tb.communicate()
    returncode = self.tb.returncode
    self.tb = None
    msg = 'tensorboard on {} for {} returned with code {}'.format(self.tb_port, self.path, returncode)
    if returncode == 0:
      logger.debug(msg)
    else:
      logger.warning(f'{msg}\n out: {outs}\n err: {errs}')
    logger.debug('tb watcher finished')

  def tb_killer(self):
    tb = self.tb
    while tb and not tb.poll():
      if time() - self.tb_t > 60:
        assert isinstance(tb, subprocess.Popen)
        tb.terminate()
        logger.debug('tensorboard for {} kill because timeout'.format(self.path))
        # break
      sleep(5)
    logger.debug('killer finish')

  def rm(self):
    # print('rm')
    self.delete()
    shutil.rmtree(self.path, ignore_errors=False)
    return {'nothing': None}

  def delete(self):
    if self.tb:
      self.tb.kill()

  def plot_trend(self):
    from matplotlib import figure
    from matplotlib import pyplot as plt

    if time() - self.plot_t < 10 and self.plot_cache:  # cache
      return io.BytesIO(self.plot_cache)

    # try:
    name, x = self.trend_data(['board', 'loss', 'return'])
    # except DataLossError:
    #   name, x = 'None', np.array([[], []])

    self.plot_t = time() + 2 * random.random()
    f, ax = plt.subplots()
    f.set_size_inches((8, 2.5))
    f.set_tight_layout(True)

    pl, = ax.plot(x[:, 0], x[:, 1])
    assert isinstance(f, figure.Figure)
    f.legend([pl], [name])
    # ax.plot(i,r)
    sio = io.BytesIO()
    f.savefig(sio, format='png', dpi=100, transparent=True)

    self.plot_cache = sio.getvalue()
    sio.seek(0)
    plt.close(f)
    return sio

  def trend_data(self, keywords):
    # a = glob.glob(path + '/**/*.tfevents.*', recursive=True)
    a = [f.path for f in rcollect(self.path, 3) if 'tfevents' in f.name ]
    target = None
    data = []
    for ef in a:
      l = 0
      for e in tf.train.summary_iterator(ef):
        l += 1
        if not target:
          for v in e.summary.value:
            if any(k in v.tag for k in keywords):
              target = v.tag

      if target:
        for i, e in enumerate(tf.train.summary_iterator(ef)):
          if i % int(l / 500 + 1):
            for v in e.summary.value:
              if target == v.tag:
                data.append((e.step, v.simple_value))
        break

    if len(data) == 0:
      data = np.zeros([1, 2])
    return target, np.array(data)
