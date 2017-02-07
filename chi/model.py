import tensorflow as tf
from .logger import logger
from .subgraph import SubGraph


def model(*a, **kw):
  if len(a) == 1 and not kw:
    return Model(*a)
  else:
    return lambda f: Model(f, *a, **kw)


class Model(SubGraph):  # TODO: make this always represent the latest subgraph
  """produces SubGraphs
  """

  def __init__(self, f, reuse=None, optimizer=None, tracker=None):
    # chi._reg.append(self)  # register
    self.f = f
    self.logger = logger
    self.after_update = []
    self.optimizer = optimizer or tf.train.AdamOptimizer()
    self.tracker = tracker
    self.tracker_active = False
    self.reuse = reuse or [tf.GraphKeys.GLOBAL_VARIABLES]
    self.name = f.__name__ if f.__name__ != '<lambda>' else 'lambda'
    self.built = False

    with tf.variable_scope(None, self.name) as sc:
      pass

    self.variable_scope = sc

    # init SubGraph properties  # TODO: unhardcode this
    def exc(): raise Exception("{} has not yet been built".format(self.__class__.name))
    self.output = property(exc)

  def __call__(self, *args, **kwargs):
    if not self.built:
      super().__init__(self, args, kwargs)  # TODO: update subgaph on every call!
      self.built = True
      return self.output
    else:
      sg = SubGraph(self, args, kwargs)  # TODO: only forward valid args
      return sg.output

  def tracked(self, *args, **kwargs):
    assert self.tracker and self.built
    if not self.tracker_active:
      self.tracker_active = True
      self.after_update.append(self.tracker.apply(self.trainable_variables()))

    gc = lambda v: self.tracker.average(v)
    sg = SubGraph(self, args, kwargs, gc)
    return sg.output

  def minimize(self, loss, name=None, collections=None):
    assert isinstance(self.optimizer, tf.train.Optimizer)
    if not isinstance(loss, (list, tuple)):
      loss = [loss]
    if not collections:
      collections = [tf.GraphKeys.LOSSES, tf.GraphKeys.REGULARIZATION_LOSSES]
    loss += sum([self.get_collection(c) for c in collections], [])
    lossop = tf.add_n([tf.reduce_sum(l) for l in loss], 'loss')
    tf.summary.scalar(name or "loss", lossop)
    # tf.summary.scalar('{}/loss'.format(self.variable_scope.name), lossop)
    min = self.optimizer.minimize(lossop, var_list=self.trainable_variables())
    with tf.control_dependencies([min]):
      return tf.identity(lossop)


def test_model():
  pass

