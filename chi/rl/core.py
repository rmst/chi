import time
from datetime import datetime
import subprocess
import shutil
import json
import os
import sys
import tensorflow as tf

import random
import numpy as np


# TODO: clean up

class Flags:
  """a container for hyperparameters and settings falling back to tf.flags.FLAGS i.e. gflags
  """

  def __init__(self, parent=None):
    self.__dict__['__parent'] = parent
    self.__dict__['__flags'] = {}
    self.__dict__['__finalized'] = False

  def get_child(self):
    return Flags(self)

  def __parent(self):
    return self.__dict__['__parent']

  def __flags(self):
    return self.__dict__['__flags']

  def __finalized(self):
    return self.__dict__['__finalized']

  def finalize(self):
    if not self.__finalized():
      tf.flags.FLAGS._parse_flags()
      self.__dict__['__finalized'] = True

  def __setattr__(self, name, v):
    if not isinstance(v, tuple):
      v = (v,)
    assert type(v[0]) in (str, int, float, bool)
    if len(v) == 1:
      self.__flags()[name] = {'value': v[0]}
    elif len(v) == 2:
      assert type(v[0]) in (str, int, float, bool)
      self.__flags()[name] = {'value': v[0], 'description': v[1]}
    else:
      assert type(v[0]) in (str, int, float, bool)
      self.__flags()[name] = {'value': v[0], 'description': v[1], 'cmd_option': v[2]}

  def __getattr__(self, name):
    if not self.__finalized():
      self.finalize()
    p = self.__parent() or tf.flags.FLAGS

    if self.__flags().has_key(name):
      return self.__flags()[name]['value']
    else:
      return getattr(p, name)

  def make_command_line_options(self, keys=None):
    if self.__parent():
      self.__parent().make_command_line_options()
    fun = {str: "DEFINE_string", int: "DEFINE_integer", float: "DEFINE_float", bool: "DEFINE_bool"}
    for k, v in self.__flags().iteritems():
      d = v.get('description', "")
      c = v.get('cmd_option', True)
      if c if not keys else k in keys:
        getattr(tf.app.flags, fun[type(v)])(k, v, d)
        del self.__flags()[k]

  def merge_kwargs(self, kwargs):
    for k, v in kwargs.iteritems():
      if type(v) in (str, int, float, bool):
        setattr(self, k, v)


class Run():
  flags = Flags()
  flags.outdir = '~/rl-flow-results/+', 'destination folder for results'
  flags.seed = random.randint(0, 1000), ''

  def __init__(self):
    import atexit, signal

    self.t_start = time.time()

    on_exit_do = []

    def on_exit():
      if on_exit_do:
        on_exit_do[0]()

    atexit.register(on_exit)

    signal.signal(signal.SIGINT, self.on_kill)
    signal.signal(signal.SIGTERM, self.on_kill)
    on_exit_do.append(self.on_exit)

  def parse_command_line_options(self, **kwargs):
    """
    Args:
      **kwargs:
        outdir: 
        seed: 
    """
    self.flags.merge_kwargs(kwargs)
    slurm = kwargs.get('slurm', False)
    self.flags.copy = False, 'copy code folder to outdir', slurm
    self.flags.tag = '', 'name tag for experiment', slurm
    self.flags.job = False, 'submit slurm job', slurm
    self.flags.nvd = False, 'run on Nvidia-Node', slurm
    self.flags.autodel = 0., 'auto delete experiments terminating before DEL minutes', slurm
    self.flags.gdb = False, 'open gdb on error', slurm
    self.flags.fulltrace = False, 'display full traceback on error', slurm

    self.flags.make_command_line_options()
    self.flags.finalize()

    self.init()
    self.seed()

  def init(self):
    argv = sys.argv

    FLAGS = self.flags

    script = sys.modules['__main__'].__file__
    scriptdir, scriptfile = os.path.split(script)

    f, n = os.path.split(FLAGS.outdir)

    if n == '+':
      outdir = create(scriptdir, f)
      i = argv.index('--outdir')  # TODO: handle --outdir=... case
      argv[i + 1] = outdir
      FLAGS.outdir = outdir
      try:
        argv.remove('--copy')
      except:
        pass
    else:
      create(scriptdir, f, n)

    print("outdir: " + FLAGS.outdir)

    script = (FLAGS.outdir + '/' + scriptfile) if FLAGS.copy else script
    argv[0] = script

    if remote(FLAGS.outdir):
      adr, outdir = FLAGS.outdir.split(':')
      i = argv.index('--outdir')  # TODO: handle --outdir=... case
      argv[i + 1] = outdir
      argv[0] = outdir + '/' + scriptfile
      # print(argv)
      subprocess.call(['ssh', adr, 'python'] + argv)
      sys.exit()

    elif FLAGS.job:
      argv.remove('--job')
      submit(argv, FLAGS.outdir)
      sys.exit()

  def seed(self):
    self.flags.seed = self.flags.seed or np.random.randint(1000000)
    random.seed(self.flags.seed)
    np.random.seed(self.flags.seed)
    tf.set_random_seed(self.flags.seed)

  def on_exit(self):
    elapsed = time.time() - self.t_start
    # self.info['end_time'] = time.time()
    # xwrite(self.outdir, self.info)
    print('Elapsed seconds: {}\n'.format(elapsed))
    # if not self.flags.job and elapsed <= FLAGS.autodel*60.:
    #   print('Deleted output folder because runtime < ' + str(FLAGS.autodel) + " minutes")
    #   shutil.rmtree(self.outdir,ignore_errors=False)

  def on_kill(self, *args):
    self.info['run_status'] = 'aborted'
    print("Experiment aborted")
    sys.exit()

  def execute(self):
    """ execute locally """
    try:
      self.info = xread(self.outdir)
    except:
      self.info = {}

    self.t_start = time.time()

    try:
      self.info['start_time'] = self.t_start
      self.info['run_status'] = 'running'
      xwrite(self.outdir, self.info)

      self.main()

      self.info['run_status'] = 'finished'
    except Exception as e:
      self.on_error(e)

  def on_error(self, e):
    FLAGS = self.flags

    self.info['run_status'] = 'error'

    # construct meaningful traceback
    import traceback, sys, code
    type, value, tb = sys.exc_info()
    tbs = []
    tbm = []
    while tb is not None:
      stb = traceback.extract_tb(tb)
      filename = stb[0][0]
      tdir, fn = os.path.split(filename)
      maindir = os.path.dirname(sys.modules['__main__'].__file__)
      if tdir == maindir or FLAGS.fulltrace:
        tbs.append(tb)
        tbm.append("{} : {} : {} : {}".format(fn, stb[0][1], stb[0][2], stb[0][3]))

      tb = tb.tb_next

    # print custom traceback
    print("\n\n- Experiment error traceback (use --gdb to debug) -\n")
    print("\n".join(tbm) + "\n")
    print("{}: {}\n".format(e.__class__.__name__, e))

    # enter interactive mode (i.e. post mortem)
    if FLAGS.gdb:
      print("\nPost Mortem:")
      for i in reversed(range(len(tbs))):
        print("Level {}: {}".format(i, tbm[i]))
        # pdb.post_mortem(tbs[i])
        frame = tbs[i].tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(banner="", local=ns)
        print("\n")


def create(self, run_folder, exfolder, exname=None):
  ''' create unique experiment folder '''
  FLAGS = self.flags

  if not exname:
    # generate unique name and create folder
    mkdir(exfolder)

    dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
    exname = '_'.join([dstr, FLAGS.env, FLAGS.tag])

  path = os.path.join(exfolder, exname)

  mkdir(path)

  # copy program to folder
  if remote(path):
    FLAGS.copy = True

  if FLAGS.copy:
    copy(run_folder, path)

  return path


def submit(self, argv, outdir):
  """
  submit a slurm job
  """
  FLAGS = self.flags

  prerun = ''
  python = sys.executable

  run_cmd = ' '.join(['cd', outdir, ';', prerun, python] + argv)

  info = {}
  info['run_cmd'] = run_cmd
  info['flags'] = getattr(FLAGS, "__flags")

  # create slurm script
  jscr = ("#!/bin/bash" + '\n' +
          "#SBATCH -o " + outdir + '/out' + '\n' +
          "#SBATCH --mem-per-cpu=" + "5000" + '\n' +
          "#SBATCH -n 4" + '\n' +
          "#SBATCH -t 24:00:00" + "\n" +
          "#SBATCH --exclusive" + "\n" +
          ('#SBATCH -C nvd \n' if FLAGS.nvd else '') + "\n" +
          "source ~/.bashrc" + "\n" +
          run_cmd)

  with open(outdir + "/slurmjob", "w") as f:
    f.write(jscr)

  cmd = "sbatch " + outdir + "/slurmjob"

  # submit slurm job
  out = subprocess.check_output(cmd, shell=True)
  print("SUBMIT: \n" + out)

  # match job id
  import re
  match = re.search('Submitted batch job (\d*)', out)
  if not match:
    raise RuntimeError('SLURM submit problem')
  jid = match.group(1)

  # write experiment info file
  info['job_id'] = jid
  info['job'] = True
  info['run_status'] = 'pending'
  xwrite(outdir, info)


def xwrite(path, data):
  with open(path + '/ezex.json', 'w+') as f:
    json.dump(data, f)


def xread(path):
  with open(path + '/ezex.json') as f:
    return json.load(f)


# Util
#

def remote(path):
  return path.find('@') != -1


def exists(path):
  if remote(path):
    adr, dir = path.split(':')
    return subprocess.check_output(["ssh", adr, "ls", dir]).split()
  else:
    return os.path.exists(path)


def mkdir(path):
  if remote(path):
    adr, dir = path.split(':')
    with open(os.devnull, 'w') as nl:
      subprocess.call(["ssh", adr, "mkdir", dir], stdout=nl, stderr=nl)
  elif not os.path.exists(path):
    os.makedirs(path)


def copy(src, dst):
  if remote(dst):
    # Example call: scp -r foo your_username@remotehost.edu:/some/remote/directory/bar
    # print("Transfering files ...")
    subprocess.call(['rsync', '-r', src + '/', dst])  # scp / rsync



def rcopy(src, dst, symlinks=False, ignore=None):
  import shutil
  ign = shutil.ignore_patterns(ignore)
  copytree(src, dst, symlinks, ign)


def copytree(src, dst, symlinks=False, ignore=None):
  import os
  import shutil
  import stat
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass  # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)


try:
  import cPickle as pickle
except:
  import pickle


def add(root, val):
  root = os.path.abspath(root)
  m = (os.listdir(root) or ['-1']).sort()[-1]
  n = '{0:08d}'.format(int(m) + 1)
  n = root + '/' + n
  with open(n, 'wb') as f:
    pickle.dump(val, f)


def lst(root): return os.listdir(root).sort()


def get(path):
  with open(path, 'rb') as f:
    return pickle.load(f)
