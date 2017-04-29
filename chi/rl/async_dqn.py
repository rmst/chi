from time import sleep, time

import chi
import chi.rl.wrappers
import gym
import numpy as np
import tensorflow as tf
from chi import Function
from chi.rl.core import Agent
from chi.rl.wrappers import get_wrapper
from gym import wrappers
from gym.wrappers import Monitor
from tensorflow.contrib import layers


class DQN:
  """
  An implementation of
    Human Level Control through Deep Reinforcement Learning
    http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
  and
    Deep Reinforcement Learning with Double Q-learning
    https://arxiv.org/abs/1509.06461
  """

  def __init__(self, n_actions, observation_shape, q_network: chi.Model, memory=None, double_dqn=True,
               replay_start=50000, logdir=""):
    self.logdir = logdir
    self.replay_start = replay_start
    self.n_actions = n_actions
    self.observation_shape = observation_shape
    self.memory = memory or chi.rl.ReplayMemory(1000000)

    def act(x: [observation_shape]):
      qs = q_network(x)
      a = tf.argmax(qs, axis=1)
      # qm = tf.reduce_max(qs, axis=1)
      return a, qs

    self.act = Function(act)

    def train_step(o: [observation_shape], a: (tf.int32, [[]]), r, t: tf.bool, o2: [observation_shape]):
      q = q_network(o)
      # ac = tf.argmax(q, axis=1)

      # compute targets
      q2 = q_network.tracked(o2)

      if double_dqn:
        a2 = tf.argmax(q_network(o2), axis=1)  # yep, that's really the only difference
      else:
        a2 = tf.argmax(q2, axis=1)

      mask2 = tf.one_hot(a2, n_actions, 1.0, 0.0, axis=1)
      q_target = tf.where(t, r, r + 0.99 * tf.reduce_sum(q2 * mask2, axis=1))
      q_target = tf.stop_gradient(q_target)

      # compute loss
      mask = tf.one_hot(a, n_actions, 1.0, 0.0, axis=1)
      qs = tf.reduce_sum(q * mask, axis=1, name='q_max')
      td = tf.subtract(q_target, qs, name='td')
      # td = tf.clip_by_value(td, -10, 10)
      # loss = tf.reduce_mean(tf.abs(td), axis=0, name='mae')
      # loss = tf.where(tf.abs(td) < 1.0, 0.5 * tf.square(td), tf.abs(td) - 0.5, name='mse_huber')
      loss = tf.reduce_mean(tf.square(td), axis=0, name='mse')

      loss = q_network.minimize(loss)

      # logging
      layers.summarize_tensors([td, loss, r, o, a,
                                tf.subtract(o2, o, name='state_dif'),
                                tf.reduce_mean(tf.cast(t, tf.float32), name='frac_terminal'),
                                tf.subtract(tf.reduce_max(q, 1, True), q, name='av_advantage')])
      # layers.summarize_tensors(chi.activations())
      # layers.summarize_tensors(chi.gradients())
      return loss

    self.train_step = Function(train_step,
                               prefetch_fctn=lambda: self.memory.sample_batch()[:-1],
                               prefetch_capacity=10)

    def log_weigths():
      v = q_network.trainable_variables()
      # print(f'log weights {v}')

      f = q_network.tracker_variables
      # print(f'log weights EMA {f}')

      difs = []
      for g in v:
        a = q_network.tracker.average(g)
        difs.append(tf.subtract(g, a, name=f'ema/dif{g.name[:-2]}'))

      layers.summarize_tensors(v + f + difs)

    self.log_weights = Function(log_weigths, async=True)

  def train(self, timesteps=10000000, tter=.25):
    debugged = False
    t = 0
    wt = 0
    while self.memory.t < timesteps:
      train_debug = not debugged and self.memory.t > 512  # it is assumed the batch size is smaller than that
      debugged = debugged or train_debug
      curb = t > self.memory.t * tter
      if (self.memory.t > self.replay_start and not curb) or train_debug:

        if t % 5000 == 0:
          print(f"{t} steps of training after {self.memory.t} steps of experience (idle for {wt*.1} s)")
          wt = 0

        self.train_step()

        if t % 50000 == 0:
          self.log_weights()
        t += 1
      else:
        sleep(.1)
        wt += 1


class DQNAgent(Agent):
  def __init__(self, dqn: DQN, env: gym.Env, episodes=1000000000, test=False, name=None, logdir=None):
    super().__init__(env, episodes, name, logdir)
    self.test = test
    self.dqn = dqn

    if test:
      def log_returns(real_return: [], ret: [], qs):
        layers.summarize_tensors([real_return, ret, qs, tf.subtract(ret, qs, name='R-Q')])

      self.log_returns = Function(log_returns, async=True)

  def action_generator(self):
    monitor = get_wrapper(self.env, wrappers.Monitor)
    dqn = self.dqn
    t = 0
    ti = time()
    for ep in range(10000000000000):
      done = False
      R = 0
      ret = 0
      annealing_time = 1000000
      value_estimates = []
      ob = yield
      while not done:
        # select actions according to epsilon-greedy policy
        anneal = max(0, 1 - dqn.memory.t / annealing_time)
        if not self.test and (dqn.memory.t < dqn.replay_start or np.random.rand() < 1 * anneal + .1):
          a = np.random.randint(0, dqn.n_actions)
          q = None
        else:
          a, q = dqn.act(ob)
          value_estimates.append(np.max(q))

        meta = {'action_values': q}
        ob2, r, done, info = yield (a, meta)

        if not self.test:
          dqn.memory.enqueue(ob, a, r, done, info)

        ob = ob2

        ret += r
        R += info.get('unwrapped_reward', r)
        t += 1

      if self.test:
        self.log_returns(R, ret, value_estimates)

      logi = 5
      if ep % logi == 0 and monitor:
        assert isinstance(monitor, Monitor)
        at = np.mean(monitor.get_episode_rewards()[-logi:])
        ds = sum(monitor.get_episode_lengths()[-logi:])
        dt = time() - ti

        ti = time()
        self.logger.info(f'av. return {at}, av. fps {ds/dt}')
