import fcntl
import json
import os
import subprocess
import sys
import time
from os.path import join

import tensorflow as tf
from contextlib import contextmanager

import inspect
import collections

import numpy as np

from chi.logger import logger
from .ops import after



def ClippingOptimizer(opt: tf.train.Optimizer, low, high):
  original = opt.apply_gradients

  def apply_gradients(grads_and_vars, *a, **kw):
    app = original(grads_and_vars, *a, **kw)
    with tf.name_scope('clip'):
      # clip = [v.assign_add(tf.maximum(high-v, 0)+tf.minimum(low-v, 0)) for g, v in grads_and_vars]
      clip = [v.assign(tf.clip_by_value(v, low, high)) for g, v in grads_and_vars]

    step = after(app, clip, name='step')
    return step

  opt.apply_gradients = apply_gradients
  return opt


def type_and_shape_from_annotation(an):
  if isinstance(an, tf.DType):
    return tf.as_dtype(an), None  # e.g.: myfun(x: float)
  elif isinstance(an, collections.Iterable):
    if isinstance(an[0], tf.DType):
      shape = shape_from_annotation(an[1])
      return tf.as_dtype(an[0]), shape  # e.g.: myfun(x: [float, (3, 5)])
    else:
      shape = shape_from_annotation(an)
      return tf.float32, shape  # e.g.: myfun(x: (3, 5))
  else:
    raise ValueError('Can not interpret annotation')


def shape_from_annotation(an: collections.Iterable):
  if isinstance(an[0], collections.Iterable):
    return [None, *an[0]]  # e.g.: myfun(x: [[3, 5]])  # will be used for automatic batch dimension
  else:
    return tuple(an)  # e.g.: myfun(x: (3, 5))  # return as tuple


def in_collections(var):
  return [k for k in tf.get_default_graph().get_all_collection_keys() if var in tf.get_collection(k)]


def basename(name):
  return name.split('/')[-1].split(':')[0]


def relpath(name, scope_name):
  m = scope_name + '/'
  end = ':0'
  if not (name.startswith(m) and name.endswith(end)):
    raise Exception("'{}' should start with '{}' and end with {}.".format(name, m, end))

  return name[len(m):-len(end)]


# def get_value(var):
#   return var.eval(session=fu.get_session())
#
#
# def set_value(var, value):
#   var.initializer.run({var.initial_value: value}, fu.get_session())


# class SummaryWriter:
#   def __init__(self, tf_summary_writer):
#     self.writer = tf_summary_writer
#
#   def write_scalar(self, tag, val):
#     s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
#     self.writer.add_summary(s, self.t)

def cmd_args(**kwargs):
  from shlex import quote
  a = []
  for k, v in kwargs.items():
    if v is False:
      a.append('--no-'+k)
    elif v is True:
      a.append('--'+k)
    elif v is not None:
      s = str(v)
      if s:
        a += ['--'+k, quote(s)]
  return a


def rln(adr, src, dst):
  subprocess.call(['ssh', adr, 'ln -s {} {}'.format(src, dst)])


def store_path(adr, path):
  return expanduser(adr+':~/.chi/cache/')+os.path.relpath(path, os.path.expanduser('~'))


def expanduser(path: str):
  return path.replace('~', '/home/'+path.split('@')[0]) if is_remote(path) else os.path.expanduser(path)


# def join(a: str, b):
#   return a+b if a.endswith('/') else a + '/' + b


def write_config(path, data):
  path = join(path, 'experiment.chi')
  if is_remote(path):
    import tempfile
    tf = tempfile.NamedTemporaryFile()
    with open(tf.name, 'w+') as f:
      json.dump(data, f, indent=1)
    subprocess.call(["scp", tf.name, path])

  else:
    with open(path, 'w+') as f:
      json.dump(data, f, indent=1)


def read_config(path):
  path = join(path, 'experiment.chi')
  if is_remote(path):
    adr, path = path.split(':')
    return json.loads(subprocess.check_output(["ssh", "cat {}".format(path)]))
  else:
    with open(path) as f:
      return json.load(f)


def is_remote(path):
  return path.find('@') != -1


def exists(path):
  if is_remote(path):
    adr, dir = path.split(':')
    return subprocess.check_output(["ssh", adr, "ls", dir]).split()
  else:
    return os.path.exists(path)


def rmr(path):
  try:
    if is_remote(path):
      adr, p = path.split(':')
      subprocess.call(["ssh", adr, "rm -r {}".format(p)])
    else:
      subprocess.call(["rm -r {}".format(path)])
  except FileNotFoundError:
    pass


def mkdirs(path):
  if is_remote(path):
    adr, dir = path.split(':')
    with open(os.devnull, 'w') as nl:
      subprocess.call(["ssh", adr, "mkdir", dir, "-p"],
                      # stdout=nl, stderr=nl
                      )
  elif not os.path.exists(path):
    os.makedirs(path)


def copydir(src, dst, with_src=True):
  # Example call: scp -r foo your_username@remotehost.edu:/some/remote/directory/bar
  # print("Transfering files ...")
  mkdirs(dst)
  if not with_src:
    src += '/'
  subprocess.call(['rsync', '-uz', '-r', '-l', src, dst])  # scp / rsync, z compress, u update-mode


def copy(src, dst):
  mkdirs(os.path.dirname(dst))
  subprocess.call(['scp', src, dst])


def seed(self):
  import random
  import numpy as np
  import tensorflow as tf
  self.flags.seed = self.flags.seed or np.random.randint(1000000)
  random.seed(self.flags.seed)
  np.random.seed(self.flags.seed)
  tf.set_random_seed(self.flags.seed)


def remote_install_dependency(address, module):
  user, host = address.split('@')
  rem = f"/home/{user}/.chi/cache"
  repo = join('/', *module.__file__.split('/')[:-2])
  target = address + ':' + rem + repo
  logger.debug(f"Uploading {repo} to {target}")
  copydir(repo, target, with_src=False)
  cmd = f'pip3 install --user -e {rem+repo}'

  try:
    out = subprocess.check_output(['ssh', address, f'echo "{cmd}"; {cmd}'], universal_newlines=True)
  except subprocess.CalledProcessError as e:
    logger.error(f'Install failed with code {e.returncode} and output:\n{e.output}')
    raise e


def run_daemon(script, kwargs, executable=sys.executable):
  args = ['nohup', executable, script] + cmd_args(**kwargs)
  logger.debug(' '.join(args))
  return subprocess.Popen(args,
                          stdout=open('/dev/null', 'w'),
                          stderr=open('/dev/null', 'w'),
                          preexec_fn=os.setpgrp)


class Config:
  _data = dict()
  _locked = False
  _f = None

  def __init__(self, path):
    import json
    self._path = path
    try:
      with open(path) as f:
        old_data = json.load(f)
    except json.JSONDecodeError:
      logger.warning('Could not decode config')
      old_data = {}
    except OSError:
      logger.debug('No config file')
      old_data = {}

    for i in range(10):
      try:
        self._f = open(path, 'w+')
        fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._locked = True
        break
      except BlockingIOError:
        import signal
        pid = old_data.get('pid')
        if pid:
          logger.info(f'Config file is locked (try {i}). Killing previous instance {pid}')
          os.kill(pid, signal.SIGTERM)
          time.sleep(.05)
        else:
          logger.error(f'Config file is locked and no pid to kill')
    assert self._locked

  def upd(self):
    self._data.update(json.load(self._f))

  def update(self, *args, **kwargs):
    assert self._locked
    self._data.update(*args, **kwargs)
    # print(self._data)

    with open(self._path, 'w+') as f:
      json.dump(self._data, f, indent=1)

  def __getattr__(self, key):
    if key.startswith('_'):
      super().__getattr__(self, key)
    else:
      return self._data[key]

  def __setattr__(self, key, value):
    if key.startswith('_'):
      super().__setattr__(key, value)
    else:
      self.update({key: value})

  def get(self, *args, **kwargs):
    return self._data.get(*args, **kwargs)

  def release(self):
    fcntl.flock(self._f, fcntl.LOCK_UN)
    self._f.close()