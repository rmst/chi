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


def experiment(f=None, local_dependencies=None, start_dashboard=False):
  """
  Decorator that transforms the decorated function into a chi.Experiment
  :param f:
  :param local_dependencies:
  :return:
  """

  if sys.modules[f.__module__].__name__ == '__main__':
    a = Experiment(f)
    a.parse_args_and_run()

  return lambda *a, **kw: Experiment(f).run(*a, **kw)


class Experiment(App):
  def __init__(self, f, local_dependencies=None, start_dashboard=False):
    chi.set_loglevel('debug')
    super().__init__(f)
    self.start_dashboard = start_dashboard
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
    rem = kwargs.get("remote")
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
    lp = kwargs['logdir'] + '/chi-logs'
    # mkdirs(lp)
    run_daemon(script, kwargs)
    sys.exit(0)

  def _run_local(self, kwargs):
    # print('_run_local')
    self.logdir = kwargs['logdir']
    self.config = Config(self.logdir + '/experiment.chi')
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

    if self.start_dashboard:
      self.start_chiboard()

    logs = join(self.logdir, 'chi-logs')
    mkdirs(logs)

    # noinspection PyArgumentList
    with Function(logdir=self.logdir, _experiment=self), capture_std(join(logs, 'stdout')):
      try:
        result = super()._run(**kwargs)
        self.config.update(end = 'success')
      except KeyboardInterrupt:
        self.config.update(end = 'sigint')
        raise
      except SigtermException:
        self.config.update(end = 'sigterm')
        raise
      finally:
        self.config.update(status='dead', t_end=time.time())

    return result

  def heartbeat(self):
    while 1:
      self.config.update({})
      time.sleep(60)

  def start_chiboard(self):
    pass
    # import subprocess
    # from chi import dashboard
    # port = dashboard.MAGIC_PORT
    # r = dashboard.port2pid(port)
    # if not r:
    #   from chi.dashboard import main
    #   chiboard = main.__file__
    #   subprocess.Popen([sys.executable, chiboard, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #
    # logger.info("{} started. Check progress at http://localhost:{}/exp/#/local{}".format(self.f.__name__, port, self.logdir))

  def run_remote(self, address, kwargs):
    # transfer files and execute remotely
    pass
