import tensorflow as tf


def after(deps, ops, outputs=None, name=None):
  with tf.control_dependencies(deps):
    return tf.tuple(outputs or ops, control_inputs=ops, name=name)
