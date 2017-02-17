import argparse
import os
import subprocess
import json
import sys
from datetime import datetime
import time
import threading
from .logger import logger
import atexit
from .app import App

def experiment(f=None, local_dependencies=None):
  return Experiment(f, local_dependencies) if f else lambda f: Experiment(f, local_dependencies)


class Experiment(App):
  current_exp = None  # TODO: make this work with multiple apps in one program (consider multithreading)

  def __init__(self, f, local_dependencies=None):
    Experiment.current_exp = self
    self.f = f
    self.local_dependencies = local_dependencies or []
    self.should_stop = False
    self.config = None
    self.logdir = None


    extra_args = {"_confdir": (None, argparse.SUPPRESS),
                  "submit": (None, None)}  # hidden argument

    App.__init__(self, f, extra_args)

  def filter_args(self, kwargs):
    logdir = kwargs.get("logdir")

    if not logdir:
      # no output, do nothing here
      return kwargs

    if logdir.endswith('/'):
      logdir = logdir[:-1]

    if logdir == '+':
      logdir = '~/.chi/experiments/+'

    logdir = expanduser(logdir)
    if logdir[-1] == '+':  # automatic naming
      dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
      logdir = logdir[:-1] + '_'.join([dstr, self.f.__name__])

    self.logdir = logdir
    kwargs.update(logdir=logdir)

    rmr(logdir)
    mkdirs(logdir)

    if is_remote(logdir):
      self.run_remote(kwargs)  # exits here
      raise Exception("Should not have reached this code")

    return kwargs


  def run(self, **kwargs):
    confdir = kwargs.get("confdir")
    if confdir:
      # print(confdir)
      self.config = read_config(confdir)
      # print(config)
      kwargs = self.config.get("args")
      self.logdir = kwargs.get("logdir")

    else:
      self.config = dict(t_creation=time.time(), name=self.f.__name__)
      kwargs = self.filter_args(kwargs)

    # local execution

    self.config.update(t_start=time.time(), args=kwargs)
    write_config(self.logdir, self.config)

    threading.Thread(target=self.heartbeat, daemon=True).start()

    self.start_chiboard()

    result = super().run(**kwargs)

    self.should_stop = True
    if self.config:
      self.config.update(t_end=time.time())

    return result

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
    process = subprocess.Popen(["ssh", "-tt", adr, "python3", rscript.split(':')[1], "--_confdir", path],
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
        time.sleep(.1)

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