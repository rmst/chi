import inspect
import threading
import traceback
from collections import OrderedDict
from contextlib import contextmanager
import signal

import sys

from inspect import Parameter


def app(f):
  a = App(f)
  if sys.modules[f.__module__].__name__ == '__main__':
    a.parse_args_and_run()

  return a.run


class SigtermException(BaseException):
  pass


@contextmanager
def sigint_exception():
  original_handler = signal.getsignal(signal.SIGINT)

  def handle_sigint(signum, frame):
    print('- sigint -')
    sys.exit()

  try:
    signal.signal(signal.SIGINT, handle_sigint)
    yield
  finally:
    signal.signal(signal.SIGINT, original_handler)


@contextmanager
def sigterm_exception():
  original_handler = signal.getsignal(signal.SIGTERM)

  def handle_sigterm(signum, frame):
    print('- sigterm -')
    sys.exit()

  try:
    signal.signal(signal.SIGTERM, handle_sigterm)
    yield
  finally:
    signal.signal(signal.SIGTERM, original_handler)


class App:
  def __init__(self, f: callable):
    self.f = f
    self.argv = sys.argv
    sig = inspect.signature(f)
    self.native_params = sig.parameters
    self.has_self = list(sig.parameters.keys())[0] == 'self'
    if self.has_self:
      # remove "self"
      self.params = OrderedDict(p for i, p in enumerate(sig.parameters.items()) if i > 0)
    else:
      self.params = OrderedDict(sig.parameters)

  def run(self, *args, **kwargs):
    # make all args kwargs
    args = {n: a for n, a in zip(self.native_params, args)}
    assert not set(args.keys()).intersection(kwargs.keys())  # conflicting args and kwargs
    kwargs.update(args)
    with sigterm_exception(), sigint_exception():
      return self._run(**kwargs)

  def _run(self, **kwargs):
    # only use valid args
    kwargs = {n: v for n, v in kwargs.items() if n in self.native_params}
    if self.has_self:
      return self.f(self, **kwargs)
    else:
      return self.f(**kwargs)

  def parse_args_and_run(self, args=None):
    args = args or sys.argv
    import argparse
    parser = argparse.ArgumentParser(description=self.f.__name__)

    for n, p in self.params.items():
      d = p.default == Parameter.empty
      t = str if d else type(p.default)
      if t is bool:
        g = parser.add_mutually_exclusive_group(required=False)
        g.add_argument('--' + n, dest=n, action='store_true')
        g.add_argument('--no-' + n, dest=n, action='store_false')
        parser.set_defaults(**{n: Parameter.empty})
      elif p.kind == Parameter.POSITIONAL_OR_KEYWORD:
        g = parser.add_mutually_exclusive_group(required=d)
        g.add_argument(n, nargs='?', type=t, default=Parameter.empty)
        g.add_argument('--' + n, dest='--' + n, type=t, default=Parameter.empty, help=argparse.SUPPRESS)
      elif p.kind == Parameter.KEYWORD_ONLY:
        parser.add_argument('--' + n, type=type(p.default), default=p.default)

    ag = vars(parser.parse_args())
    parsed = {}
    for n, p in self.params.items():
      a, b, c = ag[n], ag.get('--' + n, Parameter.empty), p.default
      v = a if a != Parameter.empty else b if b != Parameter.empty else c
      parsed.update({n: v})

    result = self.run(**parsed)

    from chi.logger import logger
    logger.info('Finishing with ' + str(result))

    sys.exit(result)
