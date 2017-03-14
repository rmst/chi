from time import time, sleep
from threading import Thread
from chi.logger import logger

store = {}


class Resource:
  def release(self):
    logger.debug('release rsrc')
    if hasattr(self, '_release'):
      self._release()
    del store[self.key]


def clean():
  try:
    while True:
      for r in store.values():
        if time() - r.t > r.to:
          r.release()
      sleep(1.)
  except:
    logger.error('error during clean')
    for r in store.values():
      r.release()

Thread(target=clean, daemon=True).start()


def resource(timeout=10000000):
  def outer(f):
    def wrap(*args, **kwargs):
      key = args
      r = store.get(key)
      if not r:
        r = Resource()
        r.key = key
        r.v = f(*args, **kwargs)
        if hasattr(r.v, 'release'):
          r.release = r._release = r.v.release
        r.to = timeout
        logger.debug('resource stored')
        store.update({key: r})
      r.t = time()
      return r.v
    return wrap
  return outer

