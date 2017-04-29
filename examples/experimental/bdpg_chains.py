""" Bayesian Determinisitc Policy Gradient evaluated on th
didactic "chain" environment
"""

import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import Experiment
from chi import experiment, model
from chi.rl import ReplayMemory


# chi.chi.tf_debug = True
from chi.rl.bdpg import BdpgAgent
from chi.rl.ddpg import DdpgAgent
from chi.rl.util import print_env


@experiment
def bdpg_chains(self: Experiment, logdir=None, env=3, heads=3, n=10, bootstrap=True):
  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope

  def gym_make(id) -> gym.Env:
    return gym.make(id)

  chi.set_loglevel('debug')

  import gym_mix
  from chi.rl.wrappers import PenalizeAction
  env = gym_mix.envs.ChainEnv(n)
  env = PenalizeAction(env, .001, 1)

  print_env(env)

  def ac(x):
    with tf.name_scope('actor_head'):
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      # a = layers.fully_connected(x, env.action_space.shape[0], None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
      a = layers.fully_connected(x, env.action_space.shape[0], None)
      return a

  def cr(x, a):
    with tf.name_scope('critic_head'):
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      x = tf.concat([x, a], axis=1)
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      # q = layers.fully_connected(x, 1, None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
      q = layers.fully_connected(x, 1, None)
      return tf.squeeze(q, 1)

  if bootstrap:
    agent = BdpgAgent(env, ac, cr, heads=heads, replay_start=5000)
  else:
    agent = DdpgAgent(env, ac, cr, replay_start=5000)
  threshold = getattr(getattr(env, 'spec', None), 'reward_threshold', None)

  for ep in range(100000):

    R, info = agent.play_episode()

    if ep % 20 == 0:
      head = info.get('head')
      print(f'Return of episode {ep} after timestep {agent.t}: {R} (head = {head}, threshold = {threshold})')
  #
  # @chi.function(logging_policy=lambda _: True)
  # def plot():
  #   # obsp = env.observation_space
  #   # h = obsp.high
  #   # l = obsp.low
  #   # x, y = tf.meshgrid(tf.linspace(l[0], h[0], 100), tf.linspace(l[1], h[1], 100))
  #   # x = tf.reshape(x, [-1])
  #   # y = tf.reshape(y, [-1])
  #   # inp = tf.stack(x, y, axis=1)
  #
  #   x = tf.linspace(0, 30, 100)
  #   x = tf.py_func(env.batch_features, x, tf.float32, stateful=False)
  #   s = pp(x)
  #   a0 = actor(s)
  #   tf.image

