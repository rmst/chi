""" Wasserstein GANs (https://arxiv.org/abs/1701.07875)

This runs but is not tested yet
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import learn
import chi
from chi.util import ClippingOptimizer

chi.setLogLevel('debug')


@chi.experiment(local_dependencies=[chi])
def wgan_conv(alpha=.00005, c=.01, m=64, n_critic=5, logdir="~/chi-results/+"):

  @chi.model(optimizer=ClippingOptimizer(tf.train.RMSPropOptimizer(alpha), -c, c))
  def critic(x: (None, 64, 64, 3)):
    x = layers.conv2d(x,  128, 5, 2)
    x = layers.conv2d(x,  256, 5, 2)
    x = layers.conv2d(x,  512, 5, 2)
    x = layers.conv2d(x, 1024, 5, 2)
    y = layers.fully_connected(x, 1, None)  # linear
    return y

  @chi.model(optimizer=tf.train.RMSPropOptimizer(alpha))
  def generator(z):
    zp = layers.fully_connected(z, 1024*4*4, None)
    x = tf.reshape(zp, [m, 4, 4, 1024])
    x = layers.conv2d_transpose(x, 512, 5, 2)
    x = layers.conv2d_transpose(x, 256, 5, 2)
    x = layers.conv2d_transpose(x, 128, 5, 2)
    x = layers.conv2d_transpose(x,   3, 5, 2, activation_fn=tf.nn.tanh)
    return x  # 64x64x3

  @chi.function(logdir=logdir + '/cri')
  def train_critic(x):
    z = tf.random_normal([m, 100])
    loss = critic(x) - critic(generator(z))
    return critic.minimize(loss)

  @chi.function(logdir=logdir + '/gen')
  def train_generator():
    z = tf.random_normal([m, 100])
    x = generator(z)
    tf.summary.image('x', x)
    loss = - critic(x)
    return generator.minimize(loss)

  @chi.function
  def resize_mnist(data):
    x = tf.reshape(data, [-1, 28, 28, 1])
    images = tf.tile(x, [1, 1, 1, 3])
    return tf.image.resize_images(images, [64, 64])

  dataset = learn.datasets.mnist.load_mnist(logdir+'data')
  data = resize_mnist(dataset.test.images)

  for i in range(100000):
    indices = np.random.randint(0, data.shape[0], m)
    batch = data[indices, ...]
    for j in range(n_critic):
      train_critic(batch)

    train_generator()
