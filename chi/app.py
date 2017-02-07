import inspect
import os
import subprocess
from collections import OrderedDict
import json
from contextlib import contextmanager
import sys
from datetime import datetime
import time
import threading
from .logger import logger
import atexit
import signal
import chi

def app(f=None):
  return App(f)


def experiment(f=None, local_dependencies=None):
  return BatchJob(f, local_dependencies) if f else lambda f: BatchJob(f, local_dependencies)


class App:
  def __init__(self, f: callable, extra_args=None):
    self.f = f
    extra_args = extra_args or {}

    self.args = OrderedDict(extra_args)
    sig = inspect.signature(f)
    self.native_args = []
    for n, v in sig.parameters.items():
      d = None if v.default == inspect.Parameter.empty else v.default
      a = "" if v.annotation == inspect.Parameter.empty else v.annotation
      self.args.update({n: (d, a)})
      self.native_args.append(n)

    if sys.modules[f.__module__].__name__ == '__main__':
      import argparse
      parser = argparse.ArgumentParser(description=f.__name__)

      for n, (v, a) in self.args.items():
        parser.add_argument('--{}'.format(n), default=v, help=a)

      ag = parser.parse_args()

      kwargs = {n: v for n, (v, _) in self.args.items()}
      kwargs.update(ag.__dict__)
      self.__call__(**kwargs)

  def filter_arguments(self, args):
    return args

  def __call__(self, *args, **kwargs):
    # make all args kwargs
    args = {n: v for (n, _), v in zip(self.args, args)}
    args.update(kwargs)

    args = self.filter_arguments(args)

    # only use valid args
    args = {n: v for n, v in args.items() if n in self.native_args}
    self.f(**args)

  def close(self, *a, **kw):
    pass


class BatchJob(App):
  def __init__(self, f, local_dependencies=None):
    self.f = f
    self.local_dependencies = local_dependencies or []
    self.should_stop = False
    self.config = None
    self.logdir = None

    extra_args = {"logdir": (None, None),
                  "confdir": (None, None)}

    super().__init__(f, extra_args)

  def filter_arguments(self, args: dict):
    confdir = args.get("confdir")
    if confdir:
      # print(confdir)
      config = read_config(confdir)
      # print(config)
      args = config.get("args")
      logdir = args.get("logdir")

    else:
      logdir = args.get("logdir")
      if not logdir:
        # no output, do nothing here
        return args

      logdir = expanduser(logdir)
      if logdir[-1] == '+':  # automatic naming
        dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
        logdir = logdir[:-1] + '_'.join([dstr, self.f.__name__])

      args.update(logdir=logdir)

      rmr(logdir)
      mkdirs(logdir)

      config = dict(t_creation=time.time(), name=self.f.__name__)

      if is_remote(logdir):
        self.run_remote(args, logdir, config)  # exits here
        raise Exception("Should not have reached this code")


    # local execution

    config.update(t_start=time.time(), args=args)
    write_config(logdir, config)

    self.config = config
    self.logdir = logdir

    threading.Thread(target=self.heartbeat, daemon=True).start()

    self.start_chiboard()

    return args

  def heartbeat(self):
    while 1:
      write_config(self.logdir, self.config)
      time.sleep(5)

  def start_chiboard(self):
    import subprocess
    from chi import dashboard
    port = 5000
    r = dashboard.port2pid(port)
    if not r:
      from chi.dashboard import main
      chiboard = main.__file__
      subprocess.Popen([sys.executable, chiboard, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    logger.info("{} started. Check progress at http://localhost:{}/exp/#/local{}".format(self.f.__name__, port, self.logdir))

  def close(self, *a, status=0, **kw):
    self.should_stop = True
    if self.config:
      self.config.update(t_end=time.time())
    sys.exit(status)

  def run_remote(self, args, logdir, config):
    # transfer files and execute remotely
    script = sys.modules['__main__'].__file__
    # sdir, sname = os.path.split(script)

    adr, path = logdir.split(':')
    rscript = store_path(adr, script)
    logger.info("Uploading script to {}".format(rscript))
    copy(script, rscript)
    for dep in self.local_dependencies:
      d = os.path.dirname(dep.__file__)
      p = store_path(adr, d)
      logger.debug("Uploading dependency to {}".format(p))
      copydir(d, p, with_src=False)
      rln(adr, p.split(':')[1], os.path.dirname(rscript.split(':')[1])+'/'+dep.__name__)

    args.update(logdir=path)
    config.update(args=args)
    write_config(logdir, config)

    logger.info("Starting process via ssh")
    process = subprocess.Popen(["ssh", "-tt", adr, "python3", rscript.split(':')[1], "--confdir", path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    # TODO: handle process errors
    def onexit(*a, **kw):
      logger.error('on exit')
      process.terminate()
      try:
        process.wait(1)
      except:
        process.kill()
      assert self.should_stop
    atexit.register(onexit)

    # signal.signal(signal.SIGINT, self.close)
    # signal.signal(signal.SIGTERM, self.close)

    def errw():
      for c in iter(lambda: process.stderr.read(1), ''):
        sys.stderr.write(c.decode())

    def stdw():
      for c in iter(lambda: process.stdout.read(1), ''):
        sys.stdout.write(c.decode())
      self.close(status=0)  # finished successfully

    threading.Thread(target=errw, daemon=True).start()
    threading.Thread(target=stdw, daemon=True).start()

    try:
      while not self.should_stop:
        time.sleep(1)

    except Exception as e:
      logger.error('Exception during remote execution \n {} \n {}'.format(e.message, e.args))
      process.kill()
      self.close(status=1)
    except:
      logger.error('System exception during remote execution (e.g. Keyboard Interrupt)')
      process.kill()
      self.close(status=2)


# Util

# def submit(self, args):
#   """submit a slurm job"""
#   prerun = ''
#   python = sys.executable
#
#   run_cmd = ' '.join(['cd', outdir, ';', prerun, python] + argv)
#
#   info = {}
#   info['run_cmd'] = run_cmd
#   info['flags'] = getattr(FLAGS, "__flags")
#
#   # create slurm script
#   jscr = ("#!/bin/bash" + '\n' +
#           "#SBATCH -o " + outdir + '/out' + '\n' +
#           "#SBATCH --mem-per-cpu=" + "5000" + '\n' +
#           "#SBATCH -n 4" + '\n' +
#           "#SBATCH -t 24:00:00" + "\n" +
#           "#SBATCH --exclusive" + "\n" +
#           ('#SBATCH -C nvd \n' if FLAGS.nvd else '') + "\n" +
#           "source ~/.bashrc" + "\n" +
#           run_cmd)
#
#   with open(outdir + "/slurmjob", "w") as f:
#     f.write(jscr)
#
#   cmd = "sbatch " + outdir + "/slurmjob"
#
#   # submit slurm job
#   out = subprocess.check_output(cmd, shell=True)
#   print("SUBMIT: \n" + out)
#
#   # match job id
#   import re
#   match = re.search('Submitted batch job (\d*)', out)
#   if not match:
#     raise RuntimeError('SLURM submit problem')
#   jid = match.group(1)
#
#   # write experiment info file
#   info['job_id'] = jid
#   info['job'] = True
#   info['run_status'] = 'pending'
#   write_config(outdir, info)

def rln(adr, src, dst):
  subprocess.call(['ssh', adr, 'ln -s {} {}'.format(src, dst)])

def store_path(adr, path):
  return expanduser(adr+':~/.chi/cache/')+os.path.relpath(path, os.path.expanduser('~'))


def expanduser(path: str):
  return path.replace('~', '/home/'+path.split('@')[0]) if is_remote(path) else os.path.expanduser(path)


def join(a: str, b):
  return a+b if a.endswith('/') else a + '/' + b


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
      subprocess.call(["ssh", adr, "mkdir", dir, "-p"], stdout=nl, stderr=nl)
  elif not os.path.exists(path):
    os.makedirs(path)


def copydir(src, dst, with_src=True):
  # Example call: scp -r foo your_username@remotehost.edu:/some/remote/directory/bar
  # print("Transfering files ...")
  mkdirs(dst)
  if not with_src:
    src += '/'
  subprocess.call(['rsync', '-uz', '-r', src, dst])  # scp / rsync, z compress, u update-mode


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