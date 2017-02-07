
import tensorflow as tf
from contextlib import contextmanager

import inspect
import collections

import numpy as np


def ClippingOptimizer(opt: tf.train.Optimizer, low, high):
  original = opt.apply_gradients

  def apply_gradients(grads_and_vars, *a, **kw):
    app = original(grads_and_vars, *a, **kw)
    asg = [v.assign_add(tf.maximum(high-v, 0)+tf.minimum(low-v, 0)) for g, v in grads_and_vars]
    return tf.group(app, *asg)  # note that clipping is asynchronous here
  opt.apply_gradients = apply_gradients
  return opt



def smart_expand(x, p: tf.Tensor):
  x = np.array(x)
  if p.get_shape().is_compatible_with(x.shape):
    return x, False
  xe = x[np.newaxis, ...]  # add batch dimension
  if p.get_shape().is_compatible_with(xe.shape):
    return xe, True
  else:
    return x, False  # let tf raise the exception


def smart_squeeze(x: np.ndarray):
  if len(x.shape) > 0 and x.shape[0] == 1:
    return x[0, ...]  # remove batch dimension
  else:
    return x


def type_and_shape_from_annotation(an):
  if isinstance(an, type):
    return tf.as_dtype(an), None
  elif isinstance(an, collections.Iterable):
    if isinstance(an[0], type):
      return an[0], an[1]
    else:
      return tf.float32, an  # just shape
  else:
    raise ValueError('Can not interpret annotation')


def signature(f):
  sig = inspect.signature(f)
  out = []
  for n, v in sig.parameters.items():
    if not v.default == inspect.Parameter.empty:
      t = tf.convert_to_tensor(v.default)
      out.append((n, t.dtype, t.get_shape(), v.default))
    elif not v.annotation == inspect.Parameter.empty:
      out.append((n, *type_and_shape_from_annotation(v.annotation), None))
    else:
      out.append((n, tf.float32, None, None))

  return out


# LEGACY:

@contextmanager
def hook(cl, callback):
  cl = cl
  orig = cl.__init__

  def patch(*args, **kwargs):
    callback(*args, **kwargs)
    return orig(*args, **kwargs)

  cl.__init__ = patch
  yield

  cl.__init__ = orig


class GraphKeys:
  MODULE_PROPERTIES = "module_properties"
  RESET_OPS = "reset_ops"


def in_collections(var):
  return [k for k in tf.get_default_graph().get_all_collection_keys() if var in tf.get_collection(k)]


def basename(name):
  return name.split('/')[-1].split(':')[0]


def relpath(name, scope_name):
  m = scope_name + '/'
  end = ':0'
  if not (name.startswith(m) and name.endswith(end)):
    raise Exception("'{}' should start with '{}' and end with {}.".format(name, m, end))

  return name[len(m):-len(end)]


def get_value(var):
  return var.eval(session=fu.get_session())


def set_value(var, value):
  var.initializer.run({var.initial_value: value}, fu.get_session())


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
