""" Tutorial for chi.model
    ----------------------
    This is how we can use python functions to define models
"""
import os
import chi
import tensorflow as tf
from tensorflow.contrib import layers  # Keras-style layers
from tensorflow.contrib import learn

chi.set_loglevel('debug')  # log whenever variables are created or shared


@chi.model
def my_digit_classifier(x: (None, 28 * 28)):  # specify shape as (None, 28*28)
  y = layers.fully_connected(x, 100)
  z = layers.fully_connected(y, 10, None)
  p = layers.softmax(z)
  return x, y, z, p


@chi.function
def train(x, labels: tf.int32):
  x, y, z, p = my_digit_classifier(x)  # create model parameters (first usage of my_model)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=z)

  gav = my_digit_classifier.compute_gradients(loss)

  th = my_digit_classifier.apply_gradients(gav)
  return th


# now that the model has been used once and
# its internal variables hav been created we can get

print('\n'.join([v.name for v in my_digit_classifier.trainable_variables()]))


@chi.function
def test(x, labels: tf.int64):
  z, p = my_digit_classifier(x)  # reuse model parameters
  correct_prediction = tf.equal(tf.argmax(p, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

# now we can train that to classify handwritten digits
datapath = os.path.join(os.path.expanduser('~'), '.chi', 'datasets', 'mnist')
dataset = learn.datasets.mnist.read_data_sets(datapath)

for i in range(10000):
  images, labels = dataset.train.next_batch(64)
  loss = train(images, labels)
  if i % 100 == 0:
    accuracy = test(*dataset.test.next_batch(1024))
    print('accuracy =', accuracy, 'after', i, 'minibatches')
