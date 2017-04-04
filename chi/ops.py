import tensorflow as tf


def after(deps, ops, outputs=None, name=None):
  # from collections import Iterable
  if not ops:
    return outputs if outputs is not None else tf.no_op()

  wrap = not isinstance(outputs, (tuple, list))

  if wrap and outputs is not None:
    outputs = (outputs,)

  if not isinstance(deps, (tuple, list)):
    deps = (deps, )

  with tf.control_dependencies(deps):
    out = tf.tuple(outputs if outputs is not None else ops, control_inputs=ops, name=name)

  if wrap:
    out = out[0]

  return out