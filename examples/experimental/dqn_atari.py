"""
Paper:
http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
"""

import chi
from chi.rl.dqn import DqnAgent
from chi.rl.util import print_env


@chi.experiment
def dqn_atari(env='Pong-v0', logdir=""):
  import numpy as np
  import gym
  import tensorflow as tf
  from gym import wrappers
  from tensorflow.contrib import layers
  from tensorflow.contrib.framework import arg_scope
  from chi.util import in_collections

  chi.set_loglevel('debug')

  env = gym.make(env)
  env = wrappers.Monitor(env, logdir+'/monitor')
  env = chi.rl.util.AtariWrapper(env)
  env = chi.rl.util.StackFrames(env, 4)
  env = wrappers.SkipWrapper(4)(env)

  print_env(env)

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1-.0005),
             optimizer=tf.train.AdamOptimizer(.0025))
  def q_network(x):
    x /= 256
    x = layers.conv2d(x, 32, 8, 4)
    x = layers.conv2d(x, 64, 4, 2)
    x = layers.conv2d(x, 64, 3, 1)
    x = layers.flatten(x)
    x = layers.fully_connected(x, 512)
    x = layers.fully_connected(x, env.action_space.n, activation_fn=None)
    x = tf.identity(x, name='Q')
    return x

  memory = chi.rl.ReplayMemory(1000000)

  agent = DqnAgent(env, q_network, memory)

  for ep in range(100000):
    agent.play_episode()
