import logging
import sys


# create logger with 'spam_application'
from contextlib import contextmanager

logger = logging.getLogger('chi')
logger.propagate = False
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('|%(asctime)s| chi/%(module)s: %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
# logger.addHandler(fh)
logger.addHandler(ch)


def set_loglevel(level: str):
  l = getattr(logging, level.upper())
  logger.setLevel(l)


class StdLogger:
  def __init__(self, file, terminal):
    self.terminal = terminal
    self.log = file

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.log.flush()
    pass


@contextmanager
def capture_std(filename):
  logfile = open(filename, "a")
  a = StdLogger(logfile, sys.stdout)
  b = StdLogger(logfile, sys.stderr)
  sys.stdout = a
  sys.stderr = b

  yield

  sys.stderr = b.terminal
  sys.stdout = a.terminal
  logfile.close()





if __name__ == "__main__":
  logger.info('info')
  logger.debug('debug')
