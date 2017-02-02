import tensorflow as tf
import numpy as np
# import types
from contextlib import contextmanager
import os.path as path
import random
from ..logger import logger, logging
from .util import signature
import inspect

from tensorflow.contrib import layers
from tensorflow.contrib import framework as fw
# fw.arg_scope
logger.setLevel(logging.DEBUG)
from . import chi
from . import util


class SubGraph:
  """
  Basically just a wrapper around a variable scope
  with some convenience functions
  """
  stack = []

  def __init__(self, model, args, kwargs, custom_getter=None):  # TODO: make SubGraph independent of Model!
    self._reused_variables = []

    self._children = []
    if SubGraph.stack:
      SubGraph.stack[-1]._children.append(self)

    def cg(getter, name, *args, **kwargs):
      if model.built:
        cs = [model.get_collection(c) for c in model.reuse]
        vs = sum(cs, [])
        vs = {util.relpath(v.name, model.variable_scope.name): v for v in vs}
        relative_name = util.relpath(name + ':0', self.variable_scope.name)
        v = vs.get(relative_name)
        if v:
          if custom_getter:
            v = custom_getter(v)
          self._reused_variables.append(v)
          logger.debug('reuse {}'.format(name))
          return v

      logger.debug('create {}'.format(name))
      v = getter(name, *args, **kwargs)
      return v

    with tf.variable_scope(None, model.name, custom_getter=cg) as sc:
      self.variable_scope = sc
      f = model.f

      # process args
      sig = signature(f)

      def filter(a, n, t, s, d):
        if s:
          a = tf.convert_to_tensor(a, t)
          a.set_shape(s)
        return a
      args = [filter(a, n, t, s, d) for a, (n, t, s, d) in zip(args, sig)]
      kwargs = {n: filter(kwargs[n], n, t, s, d) for n, t, s, d in sig if n in kwargs}

      # build
      SubGraph.stack.append(self)
      self.output = f(*args, **kwargs)
      SubGraph.stack.pop()

  def initialize(self):  # TODO: init from checkpoint
    l = self.local_variables()
    g = self.global_variables()
    r = self.reused_variables()
    vs = l+g+r
    names = chi.get_session().run(tf.report_uninitialized_variables(vs))
    initvs = [v for v in vs if v.name[:-2].encode() in names]
    chi.get_session().run(tf.variables_initializer(initvs))

  def get_ops(self):
    all_ops = tf.get_default_graph().get_operations()
    scope_ops = [x for x in all_ops if x.name.startswith(self.variable_scope.name)]
    return scope_ops

  def reused_variables(self):
    vs = self._reused_variables
    for c in self._children:
      vs += c.reused_variables()
    return vs

  def get_ops_by_type(self, type_name):
    return [op for op in self.get_ops() if op.type == type_name]

  def get_tensors_by_optype(self, type_name):
    return [op.outputs[0] for op in self.get_ops_by_type(type_name)]

  def get_collection(self, name):
    return tf.get_collection(name, self.variable_scope.name)

  def global_variables(self):
    return self.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  def trainable_variables(self):
    return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  def model_variables(self):
    return self.get_collection(tf.GraphKeys.MODEL_VARIABLES)

  def local_variables(self):
    return self.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

  def losses(self):
    return self.get_collection(tf.GraphKeys.LOSSES)

  def regularization_losses(self):
    self.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  def summaries(self):
    return tf.get_collection(tf.GraphKeys.SUMMARIES, self.variable_scope.name)

  def update_ops(self):
    return tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.variable_scope.name)

  def histogram_summaries(self, tensors, prefix=""):
    summaries = []
    for t in tensors:
      assert t.name.startswith(self.variable_scope.name)
      name = prefix + '/' + t.name[len(self.variable_scope.name):]
      summaries.append(tf.histogram_summary(name, t))

    return summaries


