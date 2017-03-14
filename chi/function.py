import tensorflow as tf
from .subgraph import SubGraph
from . import util
import chi
import inspect
import numpy as np
from chi.logger import logger
import os
from .argscope import argscope
import random
from functools import wraps


@argscope
class Function(SubGraph):
  """

  """
  def __init__(self, f, logdir=None, logging_policy=None, scope=None,
               _experiment=None, _arg_spec=None):
    """

    :param f:
    :param logdir:
    :param args:
    :param kwargs:
    """
    self.logging_policy = logging_policy
    self.logdir = logdir
    self._experiment = _experiment
    self._f = f

    # process inputs
    import collections
    self.inputs = collections.OrderedDict()
    self.auto_wrap = collections.OrderedDict()
    for name, dtype, shape, default in parse_signature(f):
      if _arg_spec and name in _arg_spec:
        sh, dtype = _arg_spec[name]
        assert isinstance(sh, tf.TensorShape)
        shape = sh.merge_with(shape) if shape is None else shape
      if default:
        p = tf.placeholder_with_default(default, shape)
      else:
        p = tf.placeholder(dtype, shape, name)
      self.auto_wrap.update({name: isinstance(shape, list)})
      self.inputs.update({name: p})
    self.use_wrap = any(self.auto_wrap.values())

    self._iscope = scope
    # if not self._scope:
    #   with tf.variable_scope(f.__name__) as sc:
    #     self._scope = sc

    self._build()

  def _build(self):
    # build graph
    self._built = True

    def scoped():
      self._step = tf.get_variable('step', initializer=0, trainable=False)
      self._advance = self._step.assign_add(1)

      out = self._f(**self.inputs)

      # collect summaries
      summaries = super(Function, self).summaries()
      self._summary_op = tf.summary.merge(summaries) if summaries else None

      self._output = tf.no_op(name='run') if out is None else out

      return out

    super().__init__(scoped, self._iscope, self._f.__name__)
    self.output = self._output
    self.step = 0

    if self.logdir:
      writer = self._experiment.writers.get(self.logdir) if self._experiment else None
      if not writer:
        writer = tf.summary.FileWriter(self.logdir)
        self._experiment.writers.update({self.logdir: writer})

      writer.add_graph(chi.chi.get_session().graph)
      self.writer = writer

      self.logging_policy = self.logging_policy or decaying_logging_policy
    else:
      self.writer = None

    super().initialize()

  # def __getattribute__(self, *args, **kwargs):
  #   ga = object.__getattribute__
  #   try:
  #     if not ga(self, '_built'):
  #       ga(self, '_build')()
  #   except AttributeError:
  #     pass
  #
  #   return ga(self, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    if not self._built:
      self._build()

    feeds = {}
    use_wrap = False  # automatically add batch dimension true and necessary
    for p, auto_wrap, arg in zip(self.inputs.values(), self.auto_wrap.values(), args):
      if auto_wrap:
        arg = np.array(arg)
        if p.get_shape().is_compatible_with(arg.shape):
          assert not use_wrap
        else:
          wrapped = arg[np.newaxis, ...]  # add batch dimension
          if p.get_shape().is_compatible_with(wrapped.shape):
            arg = wrapped
            use_wrap = True
          else:
            pass  # let TF throw the error
      feeds[p] = arg

    feeds.update(kwargs)  # TODO: process kwargs correctly

    results = self.run_log(self.output, feeds)

    if use_wrap:
      # remove batch dimension
      if isinstance(results, (tuple, list)):
        results = [r[0, ...] if r.shape[0] == 1 else r for r in results if isinstance(r, np.ndarray)]
      else:
        results = results[0, ...] if isinstance(results, np.ndarray) and results.shape[0] == 1 else results

    return results

  def run_log(self, fetches, feeds):
    log = self._summary_op is not None and self.writer and self.logging_policy(self.step)
    fetches = (fetches, self._advance)
    if log:
      (results, step), summary = chi.chi.get_session().run((fetches, self._summary_op), feeds)
      global_step = self._experiment.global_step if self._experiment else None
      self.writer.add_summary(summary, global_step=global_step or step)
    else:
      results, step = chi.chi.get_session().run(fetches, feeds)

    self.step = step
    return results

  def reset(self):
    local_inits = [v.initializer for v in self.local_variables()]
    chi.chi.get_session().run(local_inits)

  # TODO:
  def save(self):
    # save parameters etc.
    # if (self.t+45000) % 50000 == 0: # TODO: correct
    #   s = self.saver.save(self.sess,FLAGS.outdir+"f/tf/c",self.t)
    #   print("DDPG Checkpoint: " + s)
    pass


  # TODO: SubGraph as class method
  # def __get__(self, obj, objtype):
  #   """
  #   In case the SubGraph is a class method we need to instantiate it for every instance.
  #   By implementing __get__ we make it a property which allows us to instantiate a new SubGraph
  #   the first time it is used.
  #   """
  #   if obj:
  #     setattr(obj, self.f.__name__, SubGraph(self.f, parent=obj))
  #   else:
  #     # if we are called directly from the class (not the instance) TODO: error?
  #     pass


def function(f=None, *args, **kwargs) -> Function:
  """
  Decorator that transforms the decorated function into a chi.Function
  """
  deco = lambda f: Function(f, *args, **kwargs)
  return deco(f) if f else deco


def parse_signature(f):
  sig = inspect.signature(f)
  out = []
  for n, v in sig.parameters.items():
    if not v.default == inspect.Parameter.empty:
      t = tf.convert_to_tensor(v.default)
      out.append((n, t.dtype, t.get_shape(), v.default))
    elif not v.annotation == inspect.Parameter.empty:
      out.append((n, *util.type_and_shape_from_annotation(v.annotation), None))
    else:
      out.append((n, tf.float32, None, None))

  return out


def decaying_logging_policy(step):
  return random.random() < max(.01, 1000 / (step+1))


def test_runnable():
  @function
  def f(a, b):
    return a * b

  assert f(3, 2) == 6


