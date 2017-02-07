""" demo of chi.model
This is how we can use python functions to define and
scope TensorFlow models
"""
import tensorflow as tf
from chi import function, model
import numpy as np

from tensorflow.contrib import layers  # Keras-style layers


@model
def my_model(x: (32, 2)):
  z = layers.fully_connected(x, 100, tf.nn.relu)
  y = layers.fully_connected(z, 1, None)
  tf.summary.histogram('outputs', y)
  return y


@function
def compute_forward(x):
  y = my_model(x)
  return y

# now that the model has been used once and
# its internal variables hav been created we can get

print(my_model.trainable_variables())
print(my_model.summaries())  # TODO: this should output something :P


# we can also use it again, parameters are automatically shared
@function
def train(x, targets):
  y = my_model(x)
  loss = tf.square(y-targets)  # Mean Squared Error
  return my_model.minimize(loss)


# now we can use that to approximate a function

# ...
