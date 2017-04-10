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
  def __init__(self, env: gym.Env, actor: callable, critic: callable, preprocess: Model = None, memory=None, heads=2):
    obsp = env.observation_space
    acsp = env.action_space
    assert isinstance(acsp, spaces.Box), 'action space has to be continuous'
    assert isinstance(obsp, spaces.Box), 'observation space has to be continuous'

    self.env = env
    self.memory = memory or ReplayMemory(1000000)
    self.heads = heads
    self.t = 0

    pp = preprocess or Model(lambda x: x,
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

    def act(o: [obsp.shape], noise=True):
      with arg_scope([layers.batch_norm], is_training=False):
        s = pp(o)
        a = actors(s, noise=noise)
        q = critics(s, a)
        layers.summarize_tensors([s, *a, *q])
        return a

    self.act = Function(act)

    def train_actor(o: [obsp.shape]):
      s = pp(o)
      a0 = actors(s)
      q = critics(tf.stop_gradient(s), a0)
      loss = [- tf.reduce_mean(_, axis=0) for _ in q]
      return actors.minimize(loss), pp.minimize(loss)

    self.train_actor = Function(train_actor)

    bootstrap = False

    def train_critic(o: [obsp.shape], a: [acsp.shape], r, t: tf.bool, o2: [obsp.shape], i: tf.int32):
      s = pp(o)
      q2 = critics(s, [a for _ in range(heads)])
      s2 = pp.tracked(o2)
      qt = critics.tracked(s2, actors.tracked(s2))
      qtt = [tf.where(t, r, r + 0.99 * _) for _ in qt]

      def loss(_i, _q2, _qtt):
        sel = tf.equal(i, _i) if bootstrap else tf.fill(tf.shape(i), True)
        e = tf.where(sel, tf.square(_q2 - _qtt), tf.zeros_like(_q2))
        mse = tf.reduce_sum(e, axis=0) / tf.reduce_sum(tf.cast(sel, tf.float32), axis=0)
        return mse

      mse = [loss(_i, _q2, _qtt) for _i, (_q2, _qtt) in enumerate(zip(q2, qtt))]

      return critics.minimize(mse), pp.minimize(mse)

    self.train_critic = Function(train_critic)

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
      ob2, r, done, _ = self.env.step(a)
      self.memory.enqueue(ob, a, r, done, idx)

      ob = ob2
      R += r

      debug_training = self.t == 100
      if self.t > 2000 or debug_training:
        mbs = 64

        for i in range(1):
          mb = self.memory.sample_batch(mbs)
          self.train_critic(*mb)

        mb = self.memory.sample_batch(mbs)
        self.train_actor(mb[0])

      self.t += 1

    self.log_return(R)
    return R, {'head': idx}

