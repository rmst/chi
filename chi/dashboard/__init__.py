
import signal
import subprocess
import os


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
  return wrap

def port2pid(port):
  pids = []
  for lsof in ["lsof", "/usr/sbin/lsof"]:
    try:
      out = subprocess.check_output([lsof ,"-t" , "-i:" +str(port)])
      for l in out.splitlines():
        pids.append(int(l))
    except Exception:
      pass

  return pids


