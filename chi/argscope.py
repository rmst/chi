from contextlib import contextmanager
import types

argstack = []


def argscope(f):
  """
  Annotating functions or classes with this decorator allows to create scopes
  that contain partial specification of some of their arguments. E.g.:

  with MyClass(a=3, b=5):
    ...
    a = MyClass(14)  # which is equivalent to MyClass(14, a=3, b=5)
    ...

  For more examples see tests below.

  It can only be used with functions and classes that take args as well as kwargs
  and only kwargs can be specified in the scope.

  :param f:
  :return:
  """
  if isinstance(f, types.FunctionType):
    def wrapper(*args, **kwargs):
      kw = {}
      if args:  # call function
        for _f, _kw in argstack:
          if _f is f:
            kw.update(_kw)

        kw.update(kwargs)
        return f(*args, **kw)
      else:
        @contextmanager
        def cm():
          argstack.append((f, kwargs))
          yield
          assert argstack.pop()[0] is f
        return cm()

  else:
    fin = f.__init__
    fen = getattr(f, '__enter__', False)
    fex = getattr(f, '__exit__', False)

    def __init__(self, *args, **kwargs):
      kw = {}
      if args:  # call function
        for _f, _kw in argstack:
          if _f is f:
            kw.update(_kw)

        kw.update(kwargs)
        fin(self, *args, **kw)
      else:
        self._argscope_kwargs = kwargs

    def __enter__(self):
      if self._argscope_kwargs:
        argstack.append((f, self._argscope_kwargs))
      elif fen:
        fen(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
      if self._argscope_kwargs:
        assert argstack.pop()[0] is f
      elif fex:
        fex(self, exc_type, exc_val, exc_tb)

    wrapper = type(f.__name__, (f,), {'__init__': __init__,
                                      '__enter__': __enter__,
                                      '__exit__': __exit__})

  return wrapper


def test_argscope():

  @argscope
  def bla(a, b, c=None, d=None):
    return a, b, c, d

  with bla(c=5):
    assert bla(3, 4) == (3, 4, 5, None)

    with bla(d=6):
      assert bla(3, 4) == (3, 4, 5, 6)

  @argscope
  class Bla:
    def __init__(self, a, b, c=None, d=None):
      self.data = (a, b, c, d)

  with Bla(c=5):
    assert Bla(3, 4).data == (3, 4, 5, None)

    with Bla(d=6):
      assert Bla(3, 4).data == (3, 4, 5, 6)

  assert isinstance(Bla, type)


if __name__ == "__main__":
  test_argscope()