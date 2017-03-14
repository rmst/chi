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
ch = logging.StreamHandler(stream=sys.stderr)
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

    self.log.write(message)
    self.terminal.write(message)
    self.flush()  # TODO: threaded flushing better?

  def flush(self):
    self.log.flush()
    pass


# @contextmanager
# def capture_std(filename, append=False):
#   logfile = open(filename, "a" if append else "w+")
#   ch = logging.StreamHandler(stream=logfile)
#   logger.addHandler(ch)
#
#   a = StdLogger(logfile, sys.stdout)
#   b = StdLogger(logfile, sys.stderr)
#   sys.stdout = a
#   sys.stderr = b
#
#   yield
#
#   sys.stderr = b.terminal
#   sys.stdout = a.terminal
#   logfile.close()

@contextmanager
def capture_std(fn, append=False):
  logfile = open(fn, "a" if append else "w+")

  wo = sys.stderr.write
  fo = sys.stderr.flush

  def write(m):
    logfile.write(m)
    logfile.flush()
    wo(m)

  def flush():
    logfile.flush()
    fo()

  sys.stdout.write = write
  sys.stderr.write = write
  sys.stdout.flush = flush
  sys.stderr.flush = flush

  yield

  logfile.close()

if __name__ == "__main__":
  logger.info('info')
  logger.debug('debug')
