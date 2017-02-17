import inspect
from collections import OrderedDict
from contextlib import contextmanager
import signal

import sys

import chi


def app(f=None):
  return App(f)


class App:
  def __init__(self, f: callable, extra_args=None):

    self.f = f
    extra_args = extra_args or {}

    self.args = OrderedDict(extra_args)
    sig = inspect.signature(f)
    self.native_args = []
    for n, v in sig.parameters.items():
      d = None if v.default == inspect.Parameter.empty else v.default
      a = "" if v.annotation == inspect.Parameter.empty else v.annotation
      self.args.update({n: (d, a)})
      self.native_args.append(n)

    if sys.modules[f.__module__].__name__ == '__main__':
      import argparse
      parser = argparse.ArgumentParser(description=f.__name__)

      for n, (v, a) in self.args.items():
        parser.add_argument('--{}'.format(n), default=v, help=a)

      ag = parser.parse_args()

      kwargs = {n: v for n, (v, _) in self.args.items()}
      kwargs.update(ag.__dict__)

      result = self.__call__(**kwargs)

      sys.exit(result)

  def __call__(self, *args, **kwargs):
    # make all args kwargs
    args = {n: v for (n, _), v in zip(self.args, args)}
    assert not set(args.keys()).intersection(kwargs.keys())  # conflicting args and kwargs
    kwargs.update(args)

    result = self.run(**kwargs)

    return result

  def run(self, **kwargs):
    # only use valid args
    kwargs = {n: v for n, v in kwargs.items() if n in self.native_args}

    result = self.f(**kwargs)

    return result


