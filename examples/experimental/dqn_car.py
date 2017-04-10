""" NOT FUNCTIONIONAL YET
"""
import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

# chi.chi.tf_debug = True
from chi.rl.ddpg import DdpgAgent
from chi.rl.util import print_env, PenalizeAction


@experiment
def ddpg(self: Experiment, logdir=None, env=0):
  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope
  from chi.rl import ReplayMemory


  import rlunity
  # print(rlunity.__file__)
  env = gym.make('UnityCar-v0')
  env = wrappers.SkipWrapper(1)(env)
  env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)
  env = PenalizeAction(env)

  print_env(env)

  m = ReplayMemory(100000)

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
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
    ret, _ = agent.play_episode()

  getattr(getattr(env, 'unwrapped', env), 'report', lambda: None)()

  print(f'Episode {ep}: R={ret}, t={agent.t} -- (R_threshold = {threshold})')
