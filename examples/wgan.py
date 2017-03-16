#!/usr/bin/env python
""" Wasserstein GAN with convolutional nets generating MNIST
    --------------------------------------------------------
    Paper: https://arxiv.org/abs/1701.07875
    Snippets from: https://github.com/jiamings/wgan/blob/master/mnist/dcgan.py
"""
import chi
import os
chi.set_loglevel('debug')


@chi.experiment
def wgan_conv(self: chi.Experiment, alpha=5e-5, c=.01, m=64, n_critic=5, logdir=None, curbed=True):
  import tensorflow as tf
  import numpy as np
  from tensorflow.contrib import layers
  from tensorflow.contrib import learn
  from tensorflow.contrib.framework import arg_scope

  if curbed:
    tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                                intra_op_parallelism_threads=2))

  def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

  @chi.model(optimizer=tf.train.RMSPropOptimizer(alpha))
  def generator(z):
    # arg_scope set default arguments for certain layers inside that scope
    with arg_scope([layers.fully_connected, layers.convolution2d_transpose],
                   weights_initializer=layers.xavier_initializer(),
                   weights_regularizer=layers.l2_regularizer(2.5e-5)):
      x = layers.fully_connected(z, 1024, normalizer_fn=layers.batch_norm)
      x = layers.fully_connected(x, 7 * 7 * 128, normalizer_fn=layers.batch_norm)
      x = tf.reshape(x, [-1, 7, 7, 128])
      x = layers.convolution2d_transpose(x, 64, [4, 4], [2, 2], normalizer_fn=layers.batch_norm)
      x = layers.convolution2d_transpose(x, 1, [4, 4], [2, 2], activation_fn=tf.sigmoid)
      return x

  @chi.model(optimizer=tf.train.RMSPropOptimizer(alpha))
  def critic(x: [[28, 28, 1]]):
    with arg_scope([layers.convolution2d, layers.fully_connected],
                   weights_initializer=layers.xavier_initializer()):
      x = layers.convolution2d(x, 64, [4, 4], [2, 2], activation_fn=leaky_relu)
      x = layers.convolution2d(x, 128, [4, 4], [2, 2], normalizer_fn=layers.batch_norm, activation_fn=leaky_relu)
      x = layers.fully_connected(x, 1024, normalizer_fn=layers.batch_norm, activation_fn=leaky_relu)
      x = layers.fully_connected(x, 1, activation_fn=tf.identity)
      return x

  @chi.function
  def train_critic(x):
    z = tf.random_normal([m, 100])
    loss = critic(generator(z)) - critic(x)
    loss = critic.minimize(loss)

    # clip parameters
    params = critic.trainable_variables()
    loss = chi.ops.after([loss], [v.assign(tf.clip_by_value(v, -c, c)) for v in params], [loss])
    return loss

  @chi.function
  def train_generator():
    z = tf.random_normal([m, 100])
    x = generator(z)
    loss = - critic(x)
    loss = generator.minimize(loss)

    # logging
    tf.summary.image('x', x, max_outputs=16)
    layers.summarize_tensors(chi.activations() +
                             generator.trainable_variables() +
                             critic.trainable_variables())
    return loss

  # Download MNIST dataset from Yann LeCun's website
  datapath = os.path.join(os.path.expanduser('~'), '.chi', 'datasets', 'mnist')
  dataset = learn.datasets.mnist.read_data_sets(datapath, reshape=False, validation_size=0)

  print('Starting training')
  gl = []
  dl = []
  for t in range(1000000):
    di = 5
    # if t % 500 == 0 or t < 25:
    #   di = 100
    for _ in range(di):
      images, _ = dataset.train.next_batch(m)
      dl.append(train_critic(images))

    gl.append(train_generator())

    if t % 100 == 0:
      print(f'Loss generator = {np.mean(gl)}; discriminator = {np.mean(dl)}')
      gl = []
      dl = []

