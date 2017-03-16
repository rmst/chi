import json
import os
import sys
import threading
import time
from datetime import datetime

import chi
from chi.util import expanduser, join, rmr, mkdirs, run_daemon, Config
from .app import App, SigtermException
from .function import Function
from .logger import logger, capture_std

CONFIG_NAME = 'chi_experiment.json'


def experiment(f=None, local_dependencies=None, start_chiboard=True):
  """
  Decorator that transforms the decorated function into a chi.Experiment
  :param start_chiboard:
  :param f:
  :param local_dependencies:
  :return:
  """
  if not f:
    return lambda f: experiment(f, local_dependencies, start_chiboard)
  else:
    if sys.modules[f.__module__].__name__ == '__main__':
      a = Experiment(f, local_dependencies, start_chiboard)
      a.parse_args_and_run()

    return Experiment(f, local_dependencies, start_chiboard).run


class Experiment(App):
  def __init__(self, f, local_dependencies=None, start_chiboard=True):
    chi.set_loglevel('debug')
    super().__init__(f)
    self.start_chiboard = start_chiboard
    self.f = f
    self.local_dependencies = local_dependencies or []
    self.should_stop = False
    self.config = None
    self.logdir = None
    self.writers = {}
    self.global_step = None
    from inspect import Parameter
    params = dict(daemon=Parameter('daemon',
                                   Parameter.KEYWORD_ONLY,
                                   default=False,
                                   annotation="run in background"),
                  logdir=Parameter('logdir',
                                   Parameter.KEYWORD_ONLY,
                                   default=""))
    params.update(self.params)
    self.params = params

  def filter_args(self, kwargs):
    logdir = kwargs.get("logdir")

    if not logdir:
      rd = os.environ.get('CHI_ROOTDIR')
      logdir = rd + '/+' if rd else '~/.chi/experiments/+'

    if logdir.endswith('/'):
      logdir = logdir[:-1]

    logdir = expanduser(logdir)
    if logdir[-1] == '+':  # automatic naming
      dstr = datetime.now().strftime('%Y%m%d_%H%M_%S')
      logdir = logdir[:-1] + '_'.join([dstr, self.f.__name__])

    kwargs.update(logdir=logdir)

    rmr(logdir)
    mkdirs(logdir)

    return kwargs

  def _run(self, **kwargs):
    # print('_run')
    logger.debug(kwargs)
    rem = kwargs.get("scripts")
    if rem and "@" in rem:
      kwargs.update(remote='None')
      return self.run_remote(rem, kwargs)

    kwargs = self.filter_args(kwargs)

    if kwargs.get("slurm", False):
      kwargs.update(slurm=False)
      return self._submit_slurm(kwargs)

    elif kwargs.get("daemon", False):
      kwargs.update(daemon=False)
      return self._run_daemon(kwargs)
    else:
      logger.debug(kwargs)
      return self._run_local(kwargs)

  def _submit_slurm(self, kwargs):
    pass

  def _run_daemon(self, kwargs):
    # print('_run_daemon')
    script = sys.modules[self.f.__module__].__file__
    lp = kwargs['logdir'] + '/logs'
    # mkdirs(lp)
    run_daemon(script, kwargs)
    sys.exit(0)

  def _run_local(self, kwargs):
    # print('_run_local')
    self.logdir = kwargs['logdir']
    self.config = Config(join(self.logdir, CONFIG_NAME))
    self.config.update(t_creation=time.time(), name=self.f.__name__)
    self.config.update(sys_argv=self.argv, sys_executable=sys.executable,
                       status='running')

    # local execution
    self.config.update(t_start=time.time(),
                       pid=os.getpid(),
                       args=kwargs)

    self.config.update(slurm={k: v for k, v in os.environ.items() if k.startswith('SLURM_')})
    self.config.update(env=dict(os.environ))
    threading.Thread(target=self.heartbeat, daemon=True).start()

    if self.start_chiboard:
      self.run_chiboard()

    logs = join(self.logdir, 'logs')
    mkdirs(logs)

    # noinspection PyArgumentList
    with Function(logdir=self.logdir, _experiment=self), capture_std(join(logs, 'stdout')):
      try:
        result = super()._run(**kwargs)
        self.config.update(end='success')
      except KeyboardInterrupt:
        self.config.update(end='sigint')
        raise
      except SigtermException:
        self.config.update(end='sigterm')
        raise
      finally:
        self.config.update(status='dead', t_end=time.time())

    return result

  def heartbeat(self):
    while True:
      self.config.update({})
      time.sleep(60)

  def run_chiboard(self):
    pass
    import subprocess
    from chi import board
    from chi.board import CHIBOARD_HOME, MAGIC_PORT

    port = None
    start = False
    cbc = join(CHIBOARD_HOME, CONFIG_NAME)
    if os.path.isfile(cbc):
      with open(cbc) as f:
        import fcntl
        try:
          fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
          start = True
          fcntl.flock(f, fcntl.LOCK_UN)
        except BlockingIOError:  # chiboard is running
          try:
            data = json.load(f)
            port = data.get('port')
          except json.JSONDecodeError:
            port = None

    else:
      start = True

    if start:
      from chi.board import main
      chiboard = main.__file__
      subprocess.check_call([sys.executable, chiboard, '--port', str(MAGIC_PORT), '--daemon'])
      port = MAGIC_PORT

    if port is None:
      logger.warning('chiboard seems to be running but port could not be read from its config')
    else:
      logger.info(f"{self.f.__name__} started. Check progress at http://localhost:{port}/exp/#/local{self.logdir}")

  def run_remote(self, address, kwargs):
    # transfer files and execute remotely
    raise NotImplementedError()
