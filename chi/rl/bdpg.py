""" This script implements the DDPG algorithm
"""
import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import Experiment
from chi import Function
from chi import experiment, model, Model
from chi.rl import ReplayMemory

from tensorflow.contrib import layers
import gym
from gym import spaces
import numpy as np
from tensorflow.contrib.framework import arg_scope


class BdpgAgent:
  def __init__(self, env: gym.Env, actor: callable, critic: callable, preprocess: Model = None, memory=None, heads=2,
               replay_start=2000):
    self.replay_start = replay_start

    assert isinstance(env.observation_space, spaces.Box), 'action space has to be continuous'
    assert isinstance(env.action_space, spaces.Box), 'observation space has to be continuous'

    so = env.observation_space.shape
    sa = env.action_space.shape

    self.env = env
    self.memory = memory or ReplayMemory(1000000)
    self.heads = heads
    self.t = 0

    preprocess = preprocess or Model(lambda x: x,
                                     optimizer=tf.train.AdamOptimizer(.001),
                                     tracker=tf.train.ExponentialMovingAverage(1 - 0.001))

    def actors(x, noise=False):
      actions = [actor(x) for i in range(heads)]
      return actions

    actors = Model(actors,
                   optimizer=tf.train.AdamOptimizer(0.0001),
                   tracker=tf.train.ExponentialMovingAverage(1 - 0.001))

    def critics(x, actions):
      qs = [critic(x, a) for a in actions]
      return qs

    critics = Model(critics,
                    optimizer=tf.train.AdamOptimizer(.001),
                    tracker=tf.train.ExponentialMovingAverage(1 - 0.001))

    def act(o: [so], noise=True):
      with arg_scope([layers.batch_norm], is_training=False):
        s = preprocess(o)
        a = actors(s, noise=noise)
        q = critics(s, a)
        layers.summarize_tensors([s, *a, *q])
        return a

    self.act = Function(act)

    def train_actor(o: [so]):
      s = preprocess(o)
      a0 = actors(s)
      q = critics(tf.stop_gradient(s), a0)
      loss = sum((- tf.reduce_mean(_, axis=0) for _ in q))/heads
      return loss

    bootstrap = False

    def train_critic(o: [so], a: [sa], r, t: tf.bool, o2: [so], i: tf.int32):
      s = preprocess(o)
      q2 = critics(s, [a for _ in range(heads)])
      s2 = preprocess.tracked(o2)
      qt = critics.tracked(s2, actors.tracked(s2))
      qtt = [tf.where(t, r, r + 0.99 * tf.stop_gradient(_)) for _ in qt]

      # def loss(_i, _q2, _qtt):
      #   sel = tf.equal(i, _i) if bootstrap else tf.fill(tf.shape(i), True)
      #   e = tf.where(sel, tf.square(_q2 - _qtt), tf.zeros_like(_q2))
      #   mse = tf.reduce_sum(e, axis=0) / tf.reduce_sum(tf.cast(sel, tf.float32), axis=0)
      #   return mse

      mse = sum([tf.reduce_mean(tf.square(_q2 - _qtt)) for _q2, _qtt in zip(q2, qtt)])/heads
      return mse

    def train(o: [so], a: [sa], r, t: tf.bool, o2: [so], i: tf.int32):
      al = train_actor(o)
      mse = train_critic(o, a, r, t, o2, i)
      return actors.minimize(al), critics.minimize(mse), preprocess.minimize([mse, al])

    self.train = Function(train,
                          prefetch_fctn=lambda: self.memory.sample_batch(),
                          prefetch_capacity=3,
                          async=True)

    def log_return(r: []):
      layers.summarize_tensor(r, 'Return')

    self.log_return = Function(log_return)

  def play_episode(self):
    ob = self.env.reset()
    done = False
    R = 0
    self.act.initialize_local()
    idx = np.random.randint(0, self.heads)
    while not done:
      a = self.act(ob)
      a = a[idx]
      a = a if np.random.rand() > .1 else self.env.action_space.sample()
      ob2, r, done, info = self.env.step(a)
      self.memory.enqueue(ob, a, r, done, idx)

      ob = ob2
      R += info.get('unwrapped_reward', r)

      debug_training = self.t == 100
      if self.t > self.replay_start or debug_training:
        self.train()

      self.t += 1

    self.log_return(R)
    return R, {'head': idx}
