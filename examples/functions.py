""" Tutorial for chi.function
    -------------------------
    This is how we can use python functions with the chi.function decorator to build and execute TensorFlow graphs.
"""
import chi
import tensorflow as tf
import numpy as np


@chi.function
def my_tf_fun(x, y):
  z = tf.nn.relu(x) * y
  return z

assert my_tf_fun(3, 5) == 15.


# we can also specify shapes (using python3's annotation syntax)
@chi.function
def my_tf_fun(x: (2, 3), y):
  z = tf.nn.relu(x) * y
  return z

z = my_tf_fun(np.ones((2, 3)), -8)
print(z)


# the first dimension is often the batch dimension and is required
# for the Keras-style tf.contrib.layers
# With a special syntax, chi.function can automatically add that dimension and remove it
# from the result if it is == 1
@chi.function
def my_tf_fun(x: [[3]], y):  # [shape] activates auto wrap/unwrap
  z = tf.nn.relu(x) * y
  return z

assert np.all(my_tf_fun(np.zeros([32, 3]), 5) == np.zeros([32, 3]))  # with batch dimension
assert np.all(my_tf_fun(np.zeros([3]), 5) == np.zeros([3]))  # without batch dimension

# chi.function also helps with logging see experiments.py for that


# Btw.: these @ decorators are just a shortcut for:
def my_tf_fun(x, y):
  z = tf.nn.relu(x) * y
  return z
my_tf_fun = chi.function(my_tf_fun)

assert my_tf_fun(3, 5) == 15.

# other than decorators, this does not break type inference and auto complete
# actually i've filed an issue for that in PyCharm. It shouldn't be very hard to make that work:
# https://youtrack.jetbrains.com/issue/PY-23060 (you can upvote it)


# MORE ADVANCED STUFF

# decaying the learning rate (or anything)
@chi.function
def g():
  step = chi.function.step()  # returns a tf.Variable that counts how often g has been called
  return tf.train.exponential_decay(1000., step, 10000, .001)

for i in range(10000):
  print(g())
