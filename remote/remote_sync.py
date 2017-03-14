#!/usr/bin/env python3
import chi


chi.set_loglevel('debug')


@chi.experiment
def remote_sync(self: chi.Experiment, address, dir, remote_home="", delay=5):
  import subprocess
  import os
  from os.path import join
  src = os.path.join(os.path.expanduser('~'), dir)

  self.config.name = 'sync_' + dir + '@' + address.split('@')[1]

  target = join(remote_home, dir) if remote_home else join('/home', address.split('@')[0], dir)
  import tempfile
  with tempfile.NamedTemporaryFile('wt') as config:
    c = f'''
    settings{{
      nodaemon = true,
      log = "all",
    }}
    sync{{
      default.rsyncssh,
      source="{src}",
      host="{address}",
      targetdir="{target}",
      delay={delay},
      exclude = {{ '.svn', '.cvs', '.idea', '.DS_Store', '.git', '.hg'}},
      rsync = {{ }},
    }}
    '''
    print(c)
    config.write(c)
    config.flush()
    cmd = ["lsyncd", "-log", "all", config.name]
    print(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    try:

      while True:
        line = p.stdout.readline()
        if not line:
          break
        print(line[:-1])
      retval = p.wait()
      return retval

    finally:
      print('terminate process')
      p.kill()


