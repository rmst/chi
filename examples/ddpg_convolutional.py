class dkd:
  def __init__(self, f):
    self.f = f
    self.blibla = 4

  def __call__(self):
    return self.f() * 3

  def __enter__(self):
    pass
  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

a = dkd(3)
with a:
  pass


import contextlib
@contextlib.contextmanager
class bla():

  def __call__(self):
    yield 3


with bla() as bl:
  print(bl)


# h = dkd(
#   def hallo():
#     return 3
# )
# a = hallo  # type: dkd


#
# print hallo()