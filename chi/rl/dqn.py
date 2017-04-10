from time import sleep

import chi
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib import layers
from chi import Function


class DqnAgent:
  """
  An implementation of
    Human Level Control through Deep Reinforcement Learning
    http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
  and
    Deep Reinforcement Learning with Double Q-learning
    https://arxiv.org/abs/1509.06461
  """
  def __init__(self, env: gym.Env, q_network: chi.Model, memory=None, double_dqn=True):
    so = env.observation_space.shape

    self.env = env
    self.memory = memory or chi.rl.ReplayMemory(1000000)

    def act(x: [so]):
      qs = q_network(x)
      a = tf.argmax(qs, axis=1)
      # qm = tf.reduce_max(qs, axis=1)
      layers.summarize_tensor(a)
      return a, qs

    self.act = Function(act)

    def train(o: [so], a: (tf.int32, [[]]), r, t: tf.bool, o2: [so]):
      q = q_network(o)
      # ac = tf.argmax(q, axis=1)

      # compute targets
      q2 = q_network.tracked(o2)

      if double_dqn:
        a2 = tf.argmax(q_network(o2), axis=1)  # yep, that's really the only difference
      else:
        a2 = tf.argmax(q2, axis=1)

      mask2 = tf.one_hot(a2, env.action_space.n, 1.0, 0.0, axis=1)
      q_target = tf.where(t, r, r + 0.99 * tf.reduce_sum(q2 * mask2, axis=1))
      q_target = tf.stop_gradient(q_target)

      # compute loss
      mask = tf.one_hot(a, env.action_space.n, 1.0, 0.0, axis=1)
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
      layers.summarize_tensors(chi.activations())
      layers.summarize_tensors(chi.gradients())
      return loss

    self.train = Function(train,
                          prefetch_fctn=lambda: self.memory.sample_batch()[:-1],
                          prefetch_capacity=1,
                          async=True)

    def log_weigths():
      v = q_network.trainable_variables()
      # print(f'log weights {v}')

      f = q_network.tracker_variables
      # print(f'log weights EMA {f}')

      difs = []
      for g in v:
        a = q_network.tracker.average(g)
        difs.append(tf.subtract(g, a, name=f'ema/dif{g.name[:-2]}'))

      layers.summarize_tensors(v+f+difs)

    self.log_weights = Function(log_weigths, async=True)

    def log_returns(ret: [], qs):
      layers.summarize_tensors([ret, qs, tf.subtract(ret, qs, name='R-Q')])

    self.log_returns = Function(log_returns, async=True)

    self.t = 0

  def play_episode(self):
    ob = self.env.reset()
    done = False
    R = 0
    annealing_time = 1000000
    value_estimates = []

    while not done:
      # select actions according to epsilon-greedy policy
      anneal = max(0, 1 - self.t / annealing_time)
      if np.random.rand() < .1 * anneal:
        a = np.random.randint(0, self.env.action_space.n)
      else:
        a, q = self.act(ob)
        value_estimates.append(np.max(q))

      ob2, r, done, info = self.env.step(a)
      self.memory.enqueue(ob, a, r, done, info)

      ob = ob2
      R += info.get('unwrapped_reward', r)

      train_debug = self.t == 512  # it is assumed the batch size is smaller than that
      if self.t > 1000 or train_debug:
        self.train()

      if self.t % 20000 == 0:
        self.log_weights()

      self.t += 1

    self.log_returns(R, value_estimates)
    return R, {}


# Tests

def dqn_test(env='OneRoundDeterministicReward-v0'):
  env = gym.make(env)
  env = ObservationShapeWrapper(env)

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1-.01),
             optimizer=tf.train.AdamOptimizer(.001))
  def q_network(x):
    x = layers.fully_connected(x, 32)
    x = layers.fully_connected(x, env.action_space.n, activation_fn=None,
                               weights_initializer=tf.random_normal_initializer(0, 1e-4))
    return x

  agent = DqnAgent(env, q_network)

  for ep in range(4000):
    R, _ = agent.play_episode()

    if ep % 100 == 0:
      print(f'Return after episode {ep} is {R}')


def test_dqn():
  with tf.Graph().as_default(), tf.Session().as_default():
    dqn_test()  # optimal return = 1

  with tf.Graph().as_default(), tf.Session().as_default():
    dqn_test('OneRoundNondeterministicReward-v0')  # optimal return = 1

  with tf.Graph().as_default(), tf.Session().as_default():
    dqn_test('TwoRoundDeterministicReward-v0')  # optimal return = 3


# Test Utils
class ObservationShapeWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    from gym.spaces import Box
    super().__init__(env)
    self.observation_space = Box(1, 1, [1])

  def _observation(self, observation):
    return [observation]

if __name__ == '__main__':
  chi.chi.tf_debug = True
  test_dqn()