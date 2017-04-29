""" This script implements the DDPG algorithm
"""
import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

# chi.chi.tf_debug = True
from chi.rl.ddpg import DdpgAgent
from chi.rl.util import print_env
from chi.rl.wrappers import PenalizeAction


@experiment
def ddpg_car(self: Experiment, logdir=None, env=3):
  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope
  from chi.rl import ReplayMemory

  if env == 0:
    import gym_mix
    env = gym.make('ContinuousCopyRand-v0')

  elif env == 1:
    env = gym.make('Pendulum-v0')

    class P(gym.Wrapper):
      def _step(self, a):
        observation, reward, done, info = self.env.step(a)
        # observation = observation * np.array([1, 1, 1 / 8])
        reward = reward - .01 * a**2
        reward *= .1
        return observation, reward, done, info
    env = P(env)

  elif env == 2:
    env = gym.make('MountainCarContinuous-v0')
    env = wrappers.Monitor(env, logdir + '/monitor')

  elif env == 3:
    import rlunity
    # print(rlunity.__file__)
    env = gym.make('UnityCar-v0')
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)
    env = wrappers.SkipWrapper(1)(env)
    env = PenalizeAction(env)

  print_env(env)

  m = ReplayMemory(100000)

  @model(optimizer=tf.train.AdamOptimizer(.00005),
         tracker=tf.train.ExponentialMovingAverage(1-0.0005))
  def pp(x):
    # x = layers.batch_norm(x, trainable=False)
    print(x.shape)

    # x = tf.Print(x, [x], summarize=20)
    x = tf.concat([tf.maximum(x, 0), -tf.minimum(x, 0)], 1)

    # x = tf.reshape(o, [tf.shape(o)[0], 32*32*3])
    x = layers.fully_connected(x, 300)
    x = layers.fully_connected(x, 300)
    return x

  @model(optimizer=tf.train.AdamOptimizer(0.0001),
         tracker=tf.train.ExponentialMovingAverage(1-0.001))
  def actor(x, noise=False):
    # x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    a = layers.fully_connected(x, env.action_space.shape[0], None,
                               weights_initializer=tf.random_normal_initializer(0, 1e-4))
    return a

  @model(optimizer=tf.train.AdamOptimizer(.001),
         tracker=tf.train.ExponentialMovingAverage(1-0.001))
  def critic(x, a):
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    x = tf.concat([x, a], axis=1)
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    q = layers.fully_connected(x, 1, None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
    return tf.squeeze(q, 1)


  # @chi.function
  # def plot():
  #   obsp = env.observation_space
  #   h = obsp.high
  #   l = obsp.low
  #   x, y = tf.meshgrid(tf.linspace(l[0], h[0], 100), tf.linspace(l[1], h[1], 100))
  #   x = tf.reshape(x, [-1])
  #   y = tf.reshape(y, [-1])
  #   inp = tf.stack(x, y, axis=1)
  #
  #   s = pp(inp)
  #   a0 = actor(s)

  agent = DdpgAgent(env, actor, critic, pp, m, training_repeats=5)

  for ep in range(100000):
    ret, _ = agent.play_episode()

    if ep % 100 == 0:
      print(f'Episode {ep}: R={ret}, t={agent.t})')
      getattr(getattr(env, 'unwrapped', env), 'report', lambda: None)()


