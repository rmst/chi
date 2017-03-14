import getpass
import os
import shutil
import subprocess
from os.path import join
from threading import Thread
from time import sleep

from flask_socketio import Namespace

from chi.dashboard.experiment import ExperimentView, CONFNAME
from chi.dashboard.util import Repo, get_free, rcollect
from chi.logger import logger
from chi.util import mkdirs


# local schema: http://<host>:<port>/exp/local/<path>/
# remote schema: ... /exp/ssh/<num>/<path>/


class Server(Namespace, Repo):
  def __init__(self, host, port, rootdir, port_pool):
    self.port_pool = port_pool
    self.rootdir = rootdir
    self.host = host
    self.port = port
    self.exps = {}

    # Start jupyter
    jpt = shutil.which('jupyter')
    self.jupyter_port = p = get_free(self.port_pool) if jpt else -1
    if jpt:
      csp = str(dict(headers={'Content-Security-Policy':
                              f"frame-ancestors 'self' http://localhost:{self.port}/"}))

      logger.debug(f'Start jupyter ({jpt}) on port {p}')
      self.jupyter = subprocess.Popen([jpt, 'notebook', '--port='+str(p),
                                       '--no-browser', '/',
                                       "--NotebookApp.token=''",
                                       f"--NotebookApp.tornado_settings={csp}",
                                       f"--FileContentsManager.hide_globs=['']"],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      )

    Namespace.__init__(self, '/experiments')

    # Init Repo
    alt = '/tmp/chi_' + getpass.getuser()
    if os.path.exists(alt):
      os.remove(alt)
    os.symlink(os.path.expanduser('~/.chi'), alt, target_is_directory=True)
    roots = [rootdir,
             join(alt, 'experiments'),
             join(alt, 'dashboard'),
             join(alt, 'apps')]
    for p in roots:
      mkdirs(os.path.expanduser(p))

    bashrc = os.path.expanduser('~/.chi') + '/bashrc.sh'
    if os.path.exists(bashrc):
      os.remove(bashrc)
    os.symlink(os.path.expanduser('~/.bashrc'), bashrc)
    self.bashrc = bashrc


    self.connections = 0
    Repo.__init__(self, roots)

    # Poll filesystem
    def scan():
      while True:
        self.experiments()
        sleep(15)

    Thread(target=scan, daemon=True).start()

    # Watch filesystem with inotify
    Repo.observer.start()

  def dispatch(self, event):
    try:
      super().dispatch(event)
    except Exception as e:
      import traceback
      traceback.print_exc()
      logger.error(str(e))

  def on_connect(self):
    if self.connections == 0:
      pass

    self.emit('info', dict(jupyter_port=self.jupyter_port,
                           user=os.environ.get('USER'),
                           bashrc=self.bashrc,
                           ))

    self.upd()

    self.connections += 1
    logger.debug(f'connect ({self.connections})')

  def on_disconnect(self):

    self.connections -= 1
    logger.debug(f'disconnect ({self.connections})')

  def on_json(self, data):
    pass

  def upd(self):
    self.emit('data', [self.info(p) for p in self.exps])

  def on_moved(self, event):

    """
    event.event_type
        'modified' | 'created' | 'moved' | 'deleted'
    event.is_directory
        True | False
    event.src_path
        path/to/observed/file
    """
    logger.debug(str(event))

  def on_created(self, event):
    sleep(3)
    ex = os.path.exists(os.path.join(event.src_path, 'experiment.chi'))
    logger.debug('Folder created ' + str(event))
    logger.debug('is exp ' + str(ex))

    self.on_found(event.is_directory, event.src_path)

  def on_modified(self, event):
    pass

  def on_found(self, is_dir, path):
    if is_dir and os.path.exists(os.path.join(path, 'experiment.chi')):
      e = ExperimentView(path, self.host, self.port, self)
      self.exps.update({path: e})
      if self.socketio:
        self.upd()
        logger.debug(f'{len(self.exps)} experiments')

  def on_deleted(self, event):
    logger.debug(str(event))
    if event.is_directory:
      p = event.src_path
      e = self.exps.get(p)
      if e:
        e.delete()
        del self.exps[p]
        logger.debug('actually deleted exp')
      if self.socketio:
        self.upd()
        logger.debug(f'{len(self.exps)} experiments')

  def experiments(self):
    ps = (os.path.dirname(f.path) for d in [self.rootdir,
                                            '~/.chi/experiments',
                                            '~/.chi/dashboard',
                                            '~/.chi/apps']
          for f in rcollect(d, 10) if f.name == CONFNAME)
    # print(list(ps))
    exps = self.exps.copy()
    res = []

    for p in ps:
      e = exps.pop(p, None) or ExperimentView(p, self.host, self.port, self)
      res.append(e.update())
      self.exps.update({p: e})

    for e, v in exps.items():  # remaining (deleted) exps
      v.delete()
      del self.exps[e]

    self.upd()

  def adde(self, p):
    self.exps.update({p: ExperimentView(p, self.host, self.port, self)})

  def info(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].update()

  def trend(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].plot_trend()

  def delete(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].rm()

  def command(self, cmd, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].command(cmd)

  def tensorboard(self, path):
    if path not in self.exps:
      self.adde(path)
    return self.exps[path].tensorboard()

  def shutdown(self):
    for e in self.exps.values():
      e.delete()

    self.jupyter.kill()
