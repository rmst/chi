import tensorflow as tf
from .model import Model
from . import util
from . import chi


def function(f=None, logdir=None, smart_squeeze=True, *args, **kwargs):
  if f:  # use as function
    return Runnable(f, logdir, smart_squeeze, *args, **kwargs)
  else:  # use with @ as decorator
    return lambda f: Runnable(f, logdir, smart_squeeze, *args, **kwargs)


class Runnable(Model):

  def __init__(self, f, logdir=None, smart_squeeze=True, *args, **kwargs):
    self._step = 0
    Model.__init__(self, f, *args, **kwargs)
    self.smart_squeeze = smart_squeeze
    import collections
    self.inputs = collections.OrderedDict()
    for name, dtype, shape, default in util.signature(f):
      if default:
        p = tf.placeholder_with_default(default, shape)
      else:
        p = tf.placeholder(dtype, shape, name)
      self.inputs.update({name: p})

    self.output = super().__call__(**self.inputs) # get subgraph

    if self.output is None:
      self.output = tf.no_op()

    # self.inputs = self.get_tensors_by_optype("Placeholder")

    self.writer = tf.summary.FileWriter(logdir, graph=chi.get_session().graph) if logdir else None
    # collect summaries
    activations = self.get_tensors_by_optype('Relu')  # TODO: generalize to non-Relu
    # activations = self.subgraph.histogram_summaries(activations, 'activations')
    summaries = self.summaries()
    if summaries and self.writer:
      self._summary_op = tf.summary.merge(summaries)

    super().initialize()

  def __call__(self, *args, **kwargs):

    log = self.writer

    feeds = {}
    squeeze = False
    for i, arg in zip(self.inputs.values(), args):
      feeds[i], s = util.smart_expand(arg, i) if self.smart_squeeze else (arg, False)
      squeeze = squeeze or s

    feeds.update(kwargs)

    out = self.output if isinstance(self.output, (tuple, list)) else [self.output]
    out += [self._summary_op] if log else []

    res = chi.get_session().run(out, feeds)

    if log:  # TODO: good default logging policy
      # i = kwargs['global_step']
      self.writer.add_summary(res[-1], global_step=self._step)
      res = res[:-1]

    if squeeze:
      res = [util.smart_squeeze(r) for r in res] if isinstance(res, (tuple, list)) else util.smart_squeeze(res)

    if not isinstance(self.output, (tuple, list)):
      res = res[0]

    self._step += 1
    return res

  def reset(self):
    local_inits = [v.initializer for v in self.local_variables()]
    chi.get_session().run(local_inits)

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


def test_runnable():
  @function
  def f(a, b):
    return a * b

  assert f(3, 2) == 6


