import tensorflow as tf
import numpy as np
# import types
from contextlib import contextmanager
import os.path as path
import random
from .logger import logger, logging


class GraphKeys:
  MODULE_PROPERTIES = "module_properties"
  RESET_OPS = "reset_ops"


def in_collections(var):
  return [k for k in tf.get_default_graph().get_all_collection_keys() if var in tf.get_collection(k)]


def basename(name):
  return name.split('/')[-1].split(':')[0]


def relpath(name, module):
  m = module + '/'
  end = ':0'
  if not (name.startswith(m) and name.endswith(end)):
    raise Exception("'{}' should start with '{}' and end with {}.".format(name, m, end))

  return name[len(m):-len(end)]


flags = {}


def configure(**kwargs):
  """

  """
  flags.update(kwargs)


def get_value(var):
  return var.eval(session=flags.get('session'))


def set_value(var, value):
  var.initializer.run({var.initial_value: value}, flags.get('session'))


@contextmanager
def module(mod, reuse_vars=None):
  '''
  Args:
    mod: name string or Module object
    reuse_vars: a dict {relative_name1 : var1, ...} of vars to be reused

  Yields: a Module object which mainly consists of a name and variable scope
    and helper functions for collecting operations and variables from that scope.
    After the context closes the Module object create properties for each
    variable in the "module_properties" collection which can be used to
    directly feed and fetch these variables. (:rtype: Module)
  '''

  # with tf.name_scope(name) as ns: # make unique scope
  m = mod if isinstance(mod, Module) else Module(mod, reuse_vars)
  m  # type: Module
  with tf.variable_scope(m._scope):
    yield m
  m.finalize()



class Module:
  """
  Args:
    name:
    reuse_vars:
    session:
    logging_path:

  """

  def __init__(self, name, reuse_vars=None, session=None, logging_path=None, **kwargs):
    self._finalized = False
    self.output = None
    self.inputs = None
    self.reuse_vars = reuse_vars or {}

    with tf.variable_scope(name, reuse=False) as self._scope:
      pass

    logger.debug("module: " + self._scope.name)
    # print("variables: ", variables)

    self._scope.set_custom_getter(self.custom_getter)
    self._session = session or tf.get_default_session()
    self._logging_path = logging_path
    self._writer = None

  def custom_getter(self, getter, name, *args, **kwargs):
    # print(kwargs)
    # assert not getter(name, *args, **kwargs) # Variable should not exist in scope yet

    # if not kwargs.get("reuse"):
    #   # return getter(name,*args,**kwargs)
    #   raise Exception()

    n = relpath(name + ':0', self._scope.name)
    # v = self.reuse_vars.get(n + ':0')
    v = self.reuse_vars.get(n)

    # if v: logger.debug("reuse " + n + ':0' + " - " + v.name)
    if v: logger.debug("reuse " + n + " - " + v.name)

    if not v:
      v = getter(name, *args, **kwargs)

      logger.debug("create {} - {}".format(n, v.name))

      if True:  # logger.level == logging.DEBUG:
        col = in_collections(v)
        if not any([tf.GraphKeys.VARIABLES in col, tf.GraphKeys.LOCAL_VARIABLES in col]):
          raise Exception("Error: collections have to contain tf.GraphKeys.VARIABLES or tf.GraphKeys.LOCAL_VARIABLES")

    return v

  def get_ops(self):
    all_ops = tf.get_default_graph().get_operations()
    scope_ops = [x for x in all_ops if x.name.startswith(self._scope.name)]
    return scope_ops

  def get_ops_by_type(self, type_name):
    return [op for op in self.get_ops() if op.type == type_name]

  def get_tensors_by_optype(self, type_name):
    return [op.outputs[0] for op in self.get_ops_by_type(type_name)]

  def get_collection(self, name):
    return tf.get_collection(name, self._scope.name)

  def collectionToDict(self, collection):
    return {relpath(e.name, self._scope.name): e for e in collection}

  def get_var(self, name):
    name = name + ':0'
    v = self.local_variables_dict.get(name) or self.variables_dict[name]
    return v

  @property
  def variables(self):
    return self.get_collection(tf.GraphKeys.VARIABLES)

  @property
  def trainable_variables(self):
    return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  @property
  def model_variables(self):
    return self.get_collection(tf.GraphKeys.MODEL_VARIABLES)

  @property
  def local_variables(self):
    return self.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

  @property
  def summaries(self):
    return tf.get_collection(tf.GraphKeys.SUMMARIES, self._scope.name)

  @property
  def update_ops(self):
    return tf.get_collection(tf.GraphKeys.UPDATE_OPS, self._scope.name)

  @property
  def reset_ops(self):
    """ get ops in collection "reset_ops"
    """
    return tf.get_collection(GraphKeys.RESET_OPS, self._scope.name)

  @property
  def module_properties(self):
    """ get variables in collection "module_properties"
    """
    return tf.get_collection(GraphKeys.MODULE_PROPERTIES, self._scope.name)

  @property
  def variables_dict(self):
    return self.collectionToDict(self.variables)

  @property
  def trainable_variables_dict(self):
    return self.collectionToDict(self.trainable_variables)

  @property
  def model_variables_dict(self):
    return self.collectionToDict(self.model_variables)

  @property
  def local_variables_dict(self):
    return self.collectionToDict(self.local_variables)

  def histogram_summaries(self, tensors, prefix=""):
    summaries = []
    for t in tensors:
      assert t.name.startswith(self._scope.name)
      name = prefix + '/' + t.name[len(self._scope.name):]
      summaries.append(tf.histogram_summary(name, t))

    return summaries

  def finalize(self):
    self._finalized = True

    if self.output is not None:
      # make this module callable
      self.inputs = self.inputs or self.get_tensors_by_optype("Placeholder")

      # logging
      self._step = 0
      self.log = 0.  # logging probability
      # random.seed(Module._session.graph.seed)
      activations = self.get_tensors_by_optype('Relu')  # TODO: generalize to non-Relu
      activations = self.histogram_summaries(activations, 'activations')
      summaries = self.summaries
      summaries += activations
      if summaries and self._logging_path:
        self._summary_op = tf.merge_summary(summaries)
        self._writer = tf.train.SummaryWriter(
          self._logging_path + path.basename(self._scope.name),
          self._session.graph)

  def reset(self):
    local_inits = [v.initializer for v in self.local_variables]
    self._session.run(self.reset_ops + local_inits)

  def __call__(self, *args, **kwargs):
    return self.run(*args, **kwargs)

  def run(self, *args, **kwargs):
    assert self._finalized

    # log = kwargs.get('log',False)
    log = self._writer is not None and random.random < self.log

    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self.inputs[argpos]] = arg

    out = self.output + [self._summary_op] if log else self.output
    res = self._session.run(out, feeds)

    if log:
      # i = kwargs['global_step']
      self._writer.add_summary(res[-1], global_step=self._step)
      res = res[:-1]

    return res

  # TODO:
  def save(self):
    # save parameters etc.
    # if (self.t+45000) % 50000 == 0: # TODO: correct
    #   s = self.saver.save(self.sess,FLAGS.outdir+"f/tf/c",self.t)
    #   print("DDPG Checkpoint: " + s)
    pass


class SummaryWriter:
  def __init__(self, tf_summary_writer):
    self.writer = tf_summary_writer

  def write_scalar(self, tag, val):
    s = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
    self.writer.add_summary(s, self.t)

    # TODO: more functions


class ExponentialMovingAverage:
  """
  Creates shadow EMA variables colocated with variables.
  The shadow variables have differnet names than variables.
  The update op is added to the tf.GraphKEys.UPDATE_OPS collection.

  parameters:
    variables_dict: {name1: var1, ...}

  returns:
    averages: {name1: avg1, ...}
  """

  def __init__(self, variables_dict, tau=0.001):
    vs = tf.get_variable_scope().name
    with tf.name_scope(""):
      self.vars = variables_dict
      name = "{}/ExponentialMovingAverage".format(vs) if vs else "ExponentialMovingAverage"
      self.ema = tf.train.ExponentialMovingAverage(decay=1 - tau, name=name)

      self.upd = self.ema.apply(self.vars.values())  # also creates shadow vars

      self.averages = {n: self.ema.average(v) for n, v in variables_dict.items()}
      logger.debug(self.averages.values()[0].name)

  def update(self):
    # with tf.control_dependencies([self.upd]):
    #   update = tf.no_op()
    vs = tf.get_variable_scope().name
    with tf.control_dependencies([self.upd]):
      with tf.name_scope(""):
        # update = tf.group(self.upd, name = )
        name = "{}/ema_update".format(vs) if vs else "ema_update"
        update = tf.no_op(name=name)
    # print(update.name)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
    return update


def test_get_set_value():
  with tf.Session().as_default():

    b = tf.get_variable('bla', initializer=0)

    tf.initialize_variables([b]).run()

    assert get_value(b) == 0
    set_value(b, 3)
    assert get_value(b) == 3


def test_module():
  # collect variables and ops in a scope
  with module('foo') as foo:
    a = tf.get_variable('a', [0], trainable=False)
    b = tf.get_variable('b', [0])
    c = tf.get_variable('c', [0], trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

  assert foo.variables_dict == {'a': a, 'b': b}
  assert foo.variables == [a, b]
  assert foo.trainable_variables_dict == {'b': b}

  # inject variables into a scope
  with module('bar', reuse_vars={'b': b}) as bar:
    b2 = tf.get_variable('b', [0])
  assert b2 is b

  bla = Module('bla', reuse_vars={'x': a})  # use this to make autocomplete work
  with module(bla):
    x = tf.get_variable('x', [0])
  assert x is a

  # use module as function
  with tf.Session().as_default():
    with module('fun') as fun:
      x = tf.placeholder(tf.float32)
      fun.output = 2 * x

    assert fun(3.) == 6.


def test_reset():
  with tf.Session().as_default():
    with module('fun') as fun:
      x = tf.get_variable('x', initializer=0., collections=[tf.GraphKeys.LOCAL_VARIABLES])

    tf.initialize_all_variables().run()  # this does not initialize local variables
    fun.reset()  # reset / initialize local variables in fun
    assert x.eval() == 0.
    set_value(x, 3.)
    assert x.eval() == 3.
    fun.reset()
    assert x.eval() == 0.


def test_get_ops_by_type():
  # a = tf.zeros(2) + 3
  a = tf.placeholder(tf.float32)

  with module("bla") as bla:
    b = tf.placeholder(tf.float32)
    b + a

  # which tf function creates which ops:
  # {tf.placeholder: "Placeholder", tf.nn.relu: "Relu", tf.nn.zeros: "Const"} ...
  assert bla.get_ops_by_type("Placeholder") == [b.op]
  assert bla.get_tensors_by_optype("Placeholder") == [b]


if __name__ == "__main__":
  pass