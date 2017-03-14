import os
import socket
from time import sleep

from watchdog.events import FileSystemEventHandler
from watchdog.observers.inotify import InotifyObserver

from chi.logger import logger


class ndict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


class Repo(FileSystemEventHandler):
  observer = InotifyObserver()

  def __init__(self, paths):
    super().__init__()
    self.watches = set()

    for p in paths:
      p = os.path.expanduser(p)
      logger.debug('watch '+p)
      self.watches.add(Repo.observer.schedule(self, p))
      for f in os.scandir(p):
        isinstance(f, os.DirEntry)
        self.on_found(f.is_dir, f.path)

  def unschedule_all(self):
    for w in self.watches:
      Repo.observer.unschedule(w)

  def on_found(self, is_dir, path):
    pass

  def on_created(self, event):
    super().on_created(event)

  def on_moved(self, event):
    super().on_moved(event)

  def on_modified(self, event):
    super().on_modified(event)

  def on_deleted(self, event):
    super().on_deleted(event)


def get_free(pool):
  for i in range(20):
    av = [p for p in pool if check_free(p)]
    logger.debug('Free ports' + str(av))
    if av:
      break
    sleep(.1)
  if not av:
    logger.error('No ports available')
    return None
  else:
    return av[0]


def list_occupied(start, end):
  av = [p for p in range(start, end) if not check_free(p)]
  out = "\n".join([f'{p} occupied by '])


def check_free(port):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = sock.connect_ex(('127.0.0.1', port))
  sock.close()
  return False if result == 0 else True


def get_free_port(host='127.0.0.1'):
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
        t = 'undefined'
        try:
          t = 'file' if f.is_file() else 'dir' if f.is_dir() else 'undefined'
        except OSError:
          pass
        if t == 'file':
          yield f
        elif t == 'dir' and depth > 0:
          for e in rcollect(f.path, depth - 1, filter):
            yield e