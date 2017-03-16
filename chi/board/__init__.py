
import signal
import subprocess
import os
from time import sleep

_p = os.environ.get('CHI_BOARD_PORT')
MAGIC_PORT = int(_p) if _p else 19000
CHIBOARD_HOME = os.path.expanduser('~/.chi/apps/_chiboard_')


def repeat_until(timeout):
  from time import time

  def wrap(f):
    start = time()
    while True:
      success = f()
      if success:
        return True
      elif success == -1 or time()-start > timeout:
        return False
      sleep(.1)
  return wrap


def port2pid(port):
  pids = []
  for lsof in ["lsof", "/usr/sbin/lsof"]:
    try:
      out = subprocess.check_output([lsof, "-t", "-i:" + str(port)])
      for l in out.splitlines():
        pids.append(int(l))
    except Exception:
      pass

  return pids


