""" demo of chi.function
This is how we can use python functions with the
chi.function decorator to build and execute TensorFlow graphs.
"""
import tensorflow as tf
from chi import function
import numpy as np


@function
def my_tf_fun(x, y):
  z = tf.nn.relu(x) * y
  return z

assert my_tf_fun(3, 5) == 15.


# we can also specify shapes (using python3's annotation syntax)
@function
def my_tf_fun(x: (2, 3), y):
  z = tf.nn.relu(x) * y
  return z

z = my_tf_fun(np.ones((2, 3)), -8)
print(z)


# chi.function also helps with logging see 3-experiments.py for that


# these @ decorators are btw just a shortcut for:
def my_tf_function(x, y):
  z = tf.nn.relu(x) * y
  return z

my_tf_fun = function(my_tf_fun)

assert my_tf_function(3, 5) == 15.

# other than decorators, this does not break type inference and auto complete
