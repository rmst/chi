import tensorflow as tf
from .logger import logger
from .subgraph import SubGraph
from . import util
import inspect


def model(*a, **kw):
  if len(a) == 1 and not kw:
    return Model(*a)
  else:
    return lambda f: Model(f, *a, **kw)


class Model(SubGraph):  # TODO: make this always represent the latest subgraph
  """produces SubGraphs
  """

  def __init__(self, f, reuse=None, optimizer=None, tracker=None):
    self._f = f
    self.after_update = []  # list of functions
    self.optimizer = optimizer or tf.train.AdamOptimizer()
    self.tracker = tracker
    self._tracker_active = False
    self._reuse = reuse or [tf.GraphKeys.GLOBAL_VARIABLES]
    self.name = f.__name__ if f.__name__ != '<lambda>' else 'lambda'
    # self.built = False
    self._first_graph = None # subgraph
    self._last_graph = None  # subgraph

  def __call__(self, *args, **kwargs):
    sg = SubGraph(lambda: self.build(*args, **kwargs), default_name=self.name, getter=self._getter)
    if not self._last_graph:
      self._first_graph = sg
    self._last_graph = sg
    self.output = sg.output
    return sg.output

  def build(self, *args, **kwargs):
    # process args
    sig = parse_signature(self._f)
    args = [self.filter_args(a, n, t, s, d) for a, (n, t, s, d) in zip(args, sig)]
    kwargs = {n: self.filter_args(kwargs[n], n, t, s, d) for n, t, s, d in sig if n in kwargs}
    return self._f(*args, **kwargs)

  def filter_args(self, a, n, t, s, d):
    if s:
      a = tf.convert_to_tensor(a, t, n)
      a.set_shape(s)
    return a

  def _getter(self, relative_name, replacer=None):
    if not self._first_graph:
      return
    cs = [self._first_graph.get_collection(c) for c in self._reuse]
    vs = sum(cs, [])
    vs = {util.relpath(v.name, self._first_graph.name): v for v in vs}
    v = vs.get(relative_name)
    if v and replacer:
      v = replacer(v)
    return v

  def tracked(self, *args, **kwargs):
    assert self.tracker and self._last_graph
    if not self._tracker_active:
      self._tracker_active = True
      self.after_update.append(lambda: self.tracker.apply(self.trainable_variables()))

    replacer = lambda v: self.tracker.average(v)
    getter = lambda name: self._getter(name, replacer)
    sg = SubGraph(default_name=self.name+'_tracked', getter=getter)
    output = self.build(sg, args, kwargs)

    return output

  def minimize(self, losses, name=None, collections=None, **kwargs):
    assert isinstance(self.optimizer, tf.train.Optimizer)
    if not isinstance(losses, (list, tuple)):
      losses = [losses]
    if not collections:
      collections = [tf.GraphKeys.LOSSES, tf.GraphKeys.REGULARIZATION_LOSSES]
    losses += sum([self.get_collection(c) for c in collections], [])
    loss = tf.add_n([tf.reduce_sum(l) for l in losses], 'loss')
    tf.summary.scalar(name or "loss", loss)
    minimize = self.optimizer.minimize(loss, var_list=self.trainable_variables(), **kwargs)
    with tf.control_dependencies([minimize]):
      [f() for f in self.after_update]
      return tf.identity(loss)

  def get_collection(self, name):
    assert self._last_graph
    items = self._last_graph.get_collection(name)

    reused = self._last_graph._get_reused_variables()
    fcol = self._first_graph.get_collection(name)
    # print(reused, fcol)
    items += set(reused) & set(fcol)

    return items


def parse_signature(f):
  sig = inspect.signature(f)
  out = []
  for n, v in sig.parameters.items():
    if not v.annotation == inspect.Parameter.empty:
      out.append((n, *util.type_and_shape_from_annotation(v.annotation), None))
    else:
      out.append((n, None, None, None))

  return out


def test_model():
  pass

