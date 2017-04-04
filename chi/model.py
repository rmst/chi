import tensorflow as tf

import chi
from .logger import logger
from .subgraph import SubGraph
from .function import Function
from . import util
import inspect
from .util import in_collections, apply_to_leaves


def model(f=None, reuse=None, optimizer=None, tracker=None, initializer=None, regularizer=None):
  """
  Decorator that transforms the decorated function into a chi.Model
  :param f:
  :param reuse:
  :param optimizer:
  :param tracker:
  :return:
  """
  deco = lambda f: Model(f, reuse, optimizer, tracker, initializer=initializer, regularizer=regularizer)
  return deco(f) if f else deco


class Model(SubGraph):
  """produces SubGraphs
  """

  def __init__(self, f, reuse=None, optimizer=None, tracker=None, logdir=None,
               initializer=None, regularizer=None):
    self._f = f
    self.after_update = []
    self.optimizer = optimizer or tf.train.AdamOptimizer()
    self.tracker = tracker
    self.logdir = logdir
    self._tracker_active = False
    self.tracker_variables = []
    self._reuse = reuse or [tf.GraphKeys.GLOBAL_VARIABLES]
    self.name = f.__name__ if f.__name__ != '<lambda>' else 'lambda'
    self._first_graph = None  # subgraph
    self._last_graph = None  # subgraph
    with tf.variable_scope(f'{self.name}', initializer=initializer, regularizer=regularizer) as sc:
      self._scope = sc

  def __call__(self, *args, **kwargs):
    # TODO: namescope != varscope in this case, fix!
    if not self._last_graph:
      # self.run = Function(self._f, logdir=self.logdir, scope=self._scope, _arg_spec=self._arg_spec(args, kwargs))
      # self._first_graph = self.run
      self._first_graph = SubGraph(lambda: self.build(*args, **kwargs), scope=self._scope)
      self._first_graph.initialize()
      sg = self._first_graph
    else:
      sg = SubGraph(lambda: self.build(*args, **kwargs), default_name=self.name, getter=self._getter)

    self._last_graph = sg
    self.output = sg.output
    return sg.output

  def build(self, *args, **kwargs):
    # process args
    sig = parse_signature(self._f)

    def filter_args(a, n, t, s, d):
      if s:
        a = tf.convert_to_tensor(a, t, n)
        a.set_shape(s)
      return a
    kw = {n: filter_args(a, n, t, s, d) for a, (n, t, s, d) in zip(args, sig)}
    kw.update({n: filter_args(kwargs[n], n, t, s, d) for n, t, s, d in sig if n in kwargs})
    out = self._f(**kw)

    if self.tracker and self._first_graph is None:
      self.after_update.append(self.tracker.apply(chi.trainable_variables()))

    return out


  def _arg_spec(self, args, kwargs):
    sig = parse_signature(self._f)

    def filter_args(a, n, t, s, d):
      ten = tf.convert_to_tensor(a, t, n)
      return ten.get_shape(), ten.dtype
    kw = {n: filter_args(a, n, t, s, d) for a, (n, t, s, d) in zip(args, sig)}
    kw.update({n: filter_args(kwargs[n], n, t, s, d) for n, t, s, d in sig if n in kwargs})
    return kw

  def _getter(self, relative_name, replacer=None):
    cs = [self._first_graph.get_collection(c) for c in self._reuse]
    vs = sum(cs, [])
    vs = {util.relpath(v.name, self._first_graph.name): v for v in vs}
    v = vs.get(relative_name)
    if v and replacer:
      vn = replacer(v)
      if vn:
        logger.debug(f'insert {vn.name} {in_collections(vn)} for {v.name} {in_collections(v)}')
        v = vn
    return v

  def tracked(self, *args, **kwargs):
    assert self.tracker and self._last_graph

    def replacer(v):
      av = self.tracker.average(v)
      # tf.train.ExponentialMovingAverage
      self.tracker_variables.append(av)
      return av
    getter = lambda name: self._getter(name, replacer)
    sg = SubGraph(lambda: self.build(*args, **kwargs), default_name=self.name+'_tracked', getter=getter)
    out = apply_to_leaves(sg.output, lambda x: tf.stop_gradient(x) if isinstance(x, tf.Tensor) else x)
    return out

  def minimize(self, losses, name=None, collections=None, **kwargs):
    with tf.name_scope('minimize'):
      assert isinstance(self.optimizer, tf.train.Optimizer)
      if not isinstance(losses, (list, tuple)):
        losses = [losses]
      if not collections:
        collections = [tf.GraphKeys.LOSSES, tf.GraphKeys.REGULARIZATION_LOSSES]
      losses += sum([self.get_collection(c) for c in collections], [])

      logger.info(f'"{self.name}" minimizing losses:\n' +
                  '\n'.join([f'{l.name}' for l in losses]) + '\n' +
                  f'... with respect to:\n' +
                  '\n'.join([f'  {v.name} - {in_collections(v)}' for v in self.trainable_variables()]) + '\n')

      with tf.name_scope('losses'):
        loss = tf.add_n([tf.reduce_sum(l) for l in losses], 'loss')
        tf.summary.scalar(name or self.name+"_loss", loss)

      var_list = self.trainable_variables()
      if var_list:
        gav = self.optimizer.compute_gradients(loss, var_list=var_list)
        for g, v in gav:
          tf.add_to_collection('chi_gradients', g)

        with tf.variable_scope(self._scope):
          sg = SubGraph(lambda: self.optimizer.apply_gradients(gav, name='apply_gradients'), default_name='minimize')
          sg.initialize()
          minimize = sg.output
      else:
        minimize = tf.no_op()

      # self.after_update += self.update_ops()

      logger.info(f'"{self.name}" updating after optimization step:\n' +
                  '\n'.join([f'  {v.name} - {in_collections(v)}' for v in self.after_update]) + '\n')

      with tf.control_dependencies([minimize]):
        assert isinstance(self._first_graph, SubGraph)
        out = tf.tuple([loss], control_inputs=self.after_update, name='after_update')
        return out[0]

  def get_collection(self, name):
    assert self._last_graph
    items = self._last_graph.get_collection(name)

    reused = (
      self._last_graph._get_reused_variables() +
      self._first_graph.regularization_losses()
    )
    fcol = self._first_graph.get_collection(name)
    # print(reused, fcol)
    items += set(reused) & set(fcol)

    return items

  def regularization_losses(self):
    items = super().regularization_losses()
    items += self._first_graph.regularization_losses()
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

