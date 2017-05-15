from queue import Queue
from threading import Thread

import tensorflow as tf

from chi.util import in_collections, apply_to_leaves
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
               prefetch_fctn=None, prefetch_capacity=1, async=False, prefetch_threads=None,
               _experiment=None, _arg_spec=None):
    """

    :param f:
    :param logdir:
    :param args:
    :param kwargs:
    """
    self.prefetch_threads = prefetch_threads or prefetch_capacity
    self._arg_spec = _arg_spec
    self.prefetch_capacity = prefetch_capacity
    self.prefetch_fctn = prefetch_fctn
    self.logging_policy = logging_policy
    self.logdir = logdir
    self._experiment = _experiment
    self._f = f
    self._iscope = scope
    self.session = chi.chi.get_session()

    self.async = async
    if async:
      self.queue = Queue(1)
      self.thread = Thread(target=self.run_async, daemon=True)
      self.thread_started = False

    # if not self._scope:
    #   with tf.variable_scope(f.__name__) as sc:
    #     self._scope = sc

    self._build()

  def _build(self):
    # build graph
    self._built = True

    def scoped():

      sig = parse_signature(self._f)

      if self.prefetch_fctn:
        # Create prefetching queue
        with tf.name_scope('prefetch_inputs'):
          dtypes = [p[1] for p in sig]

          def pf():
            outs = self.prefetch_fctn()
            outs = outs if isinstance(outs, tuple) else (outs,)
            outs = tuple(np.asarray(out, dtype.as_numpy_dtype) for out, dtype in zip(outs, dtypes))
            return outs

          # shapes = [p[2] for p in sig]
          queue = tf.FIFOQueue(capacity=self.prefetch_capacity, dtypes=dtypes)

          qs = queue.size()  # log queue size
          tf.summary.scalar(qs.name, qs)

          enqueue_op = queue.enqueue(tf.py_func(pf, [], dtypes))
          qr = tf.train.QueueRunner(queue, [enqueue_op] * self.prefetch_threads)
          tf.train.add_queue_runner(qr)

          dequeued_tensors = queue.dequeue()
          dequeued_tensors = dequeued_tensors if isinstance(dequeued_tensors, (list, tuple)) else (dequeued_tensors,)

          sig = [(n, t, s, d) for (n, t, s, _), d in zip(sig, dequeued_tensors)]

      # create placeholders
      import collections
      self.inputs = collections.OrderedDict()
      self.auto_wrap = collections.OrderedDict()
      for name, dtype, shape, default in sig:
        if self._arg_spec and name in self._arg_spec:
          sh, dtype = self._arg_spec[name]
          assert isinstance(sh, tf.TensorShape)
          shape = sh.merge_with(shape) if shape is None else shape
        if default is not None:
          p = tf.placeholder_with_default(default, shape, name)
        else:
          p = tf.placeholder(dtype, shape, name)
        self.auto_wrap.update({name: isinstance(shape, list)})
        self.inputs.update({name: p})
      self.use_wrap = any(self.auto_wrap.values())

      # local step
      self._step = tf.get_variable('step', initializer=0, trainable=False, dtype=tf.int32)
      self._advance = self._step.assign_add(1)

      # build function
      out = self._f(**self.inputs)

      # collect summaries
      summaries = super(Function, self).summaries()
      self._summary_op = tf.summary.merge(summaries) if summaries else None

      output = tf.no_op(name='run') if out is None else out

      logger.debug(f'Function {self.name} updating:\n' +
                   '\n'.join([f'  {v.name} - {in_collections(v)}' for v in self.update_ops()]) + '\n')
      out = chi.util.after(output, self.update_ops(), out)
      self._output = out

      return out

    super().__init__(scoped, self._iscope, self._f.__name__)
    self.output = self._output
    self.step = 0

    if self.logdir:
      writer = self._experiment.writers.get(self.logdir) if self._experiment else None
      if not writer:
        writer = tf.summary.FileWriter(self.logdir)
        self._experiment.writers.update({self.logdir: writer})

      writer.add_graph(self.session.graph)
      self.writer = writer

      self.logging_policy = self.logging_policy or decaying_logging_policy
    else:
      self.writer = None

    super().initialize()

    self.coordinator = tf.train.Coordinator()
    self.queue_collection_key = self.name + 'queue_runners'
    qr = self.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    if qr:
      logger.debug(f'Collecting queue runners {qr}')
    for q in qr:
      tf.add_to_collection(self.queue_collection_key, q)

    self.queue_runners = None

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

    if not self.queue_runners:
      self.queue_runners = tf.train.start_queue_runners(self.session,
                                                        self.coordinator,
                                                        collection=self.queue_collection_key)
    if self.async:
      if not self.thread_started:
        self.thread.setName(self.name)
        self.thread.start()
        self.thread_started = True
      self.queue.put((args, kwargs))
      return
    else:
      return self.run(*args, **kwargs)

  def run_async(self):
    while True:
      args, kwargs = self.queue.get()
      self.run(*args, **kwargs)

  def run(self, *args, **kwargs):
    feeds = {}
    use_wrap = False  # automatically add batch dimension true and necessary
    for p, auto_wrap, arg in zip(self.inputs.values(), self.auto_wrap.values(), args):
      if auto_wrap:
        arg = np.asarray(arg)
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
      def unwrap(r: np.ndarray):
        return r[0, ...] if r.shape[0] == 1 else r

      results = apply_to_leaves(results, unwrap)

    return results

  def run_log(self, fetches, feeds):
    log = self._summary_op is not None and self.writer and self.logging_policy(self.step)
    fetches = (fetches, self._advance)
    try:
      if log:
        (results, step), summary = self.session.run((fetches, self._summary_op), feeds)
        global_step = self._experiment.global_step if self._experiment else None
        self.writer.add_summary(summary, global_step=global_step or step)
      else:
        results, step = self.session.run(fetches, feeds)
    except tf.errors.OutOfRangeError:
      results = None
      self.coordinator.request_stop()
      self.coordinator.join(self.queue_runners)

    self.step = step
    return results

  def reset(self):
    local_inits = [v.initializer for v in self.local_variables()]
    self.session.run(local_inits)

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


class FunctionBuilder:
  def __call__(self, f=None, *args, **kwargs) -> Function:
    """
    Decorator that transforms the decorated function into a chi.Function
    """
    deco = lambda f: Function(f, *args, **kwargs)
    return deco(f) if f else deco

  def stack(self):
    return [sg for sg in SubGraph.stack if isinstance(sg, Function)]

  def current(self) -> Function:
    return self.stack()[-1]

  def step(self):
    return self.current()._step


function = FunctionBuilder()


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
  return random.random() < max(.01, 1000 / (step + 1))


def test_runnable():
  @function
  def f(a, b):
    return a * b

  assert f(3, 2) == 6
