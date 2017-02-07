
import signal
import subprocess
import os


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


