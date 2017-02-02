# experiment #
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from flow.chi import model, runnable
from flow.chi.util import ClippingOptimizer


from tensorflow.tensorboard import tensorboard

# tf.flags.DEFINE_string('logdir', '/tmp/wgan', '')
FLAGS = tf.flags.FLAGS

alpha = .00005
c = .01
m = 2  # 64
n_critic = 5
activation = tf.nn.relu


@model(optimizer=ClippingOptimizer(tf.train.RMSPropOptimizer(alpha), -c, c))
def critic(x: (None, 64, 64, 3)):
  h0 = activation(layers.conv2d(x, 128, 5, 2))
  h1 = activation(layers.conv2d(h0, 256, 5, 2))
  h2 = activation(layers.conv2d(h1, 512, 5, 2))
  h3 = activation(layers.conv2d(h2, 1024, 5, 2))
  y = layers.fully_connected(h3, 1, None)
  return y


@model(optimizer=tf.train.RMSPropOptimizer(alpha))
def generator(z):
  zp = layers.fully_connected(z, 1024*4*4, None)
  h0 = tf.reshape(zp, [m, 4, 4, 1024])
  h1 = activation(layers.conv2d_transpose(h0, 512, 5, 2))
  h2 = activation(layers.conv2d_transpose(h1, 256, 5, 2))
  h3 = activation(layers.conv2d_transpose(h2, 128, 5, 2))
  h4 = activation(layers.conv2d_transpose(h3, 3, 5, 2))
  return tf.nn.tanh(h4)  # 64x64x3


@runnable(logdir='/tmp/wgan/critic')
def train_critic(x):
  z = tf.random_normal([m, 100])
  loss = critic(x) - critic(generator(z))
  return critic.minimize(loss)


@runnable(logdir='/tmp/wgan/generator')
def train_generator():
  z = tf.random_normal([m, 100])
  x = generator(z)
  tf.summary.image('x', x)
  loss = - critic(x)
  return generator.minimize(loss)


@runnable
def resize_mnist(data):
  x = tf.reshape(data, [-1, 28, 28, 1])
  images = tf.tile(x, [1, 1, 1, 3])
  return tf.image.resize_images(images, [64, 64])

dataset = learn.datasets.mnist.load_mnist('/tmp/wgan')
data = resize_mnist(dataset.test.images)

for i in range(100000):
  indices = np.random.randint(0, data.shape[0], m)
  batch = data[indices, ...]
  for j in range(n_critic):
    train_critic(batch)

  train_generator()