from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import chi
from chi import Function, Model
from chi.rl.core import Agent
from chi.rl.memory import ShardedMemory, ReplayMemory
from chi.util import scalar_summaries


class BootstrappedDQN:
  """

  """
  def __init__(self, n_actions, observation_shape, pp: Model, heads: Model, double_dqn=True, replay_start=50000, logdir=None, clip_gradients=10.):
    self.logdir = logdir
    self.observation_shape = observation_shape
    self.n_actions = n_actions
    self.replay_start = replay_start
    self.n_heads = None

    self.memory = ShardedMemory()

    self.discount = .99
    self.step = 0
    self.n_state = None

    def act(x: [observation_shape]):
      s = pp(x)

      self.n_state = int(s.shape[1])

      qs = heads(s)

      self.n_heads = len(qs)
      return qs, s

    self.act = Function(act)

    @chi.model(optimizer=tf.train.RMSPropOptimizer(6.25e-5, .95, .95, .01))
    def pred(x, a: tf.int32):
      x = tf.concat((x, layers.one_hot_encoding(a, self.n_actions)), axis=1)
      x = layers.fully_connected(x, 100)
      x = layers.fully_connected(x, 50)
      x = layers.fully_connected(x, 50)
      x = layers.fully_connected(x, 100)
      x = layers.fully_connected(x, self.n_state, None)
      return x

    def train_step(o: [observation_shape], a: (tf.int32, [[]]), r, t: tf.bool, o2: [observation_shape]):
      s = pp(o)
      qs = heads(s)

      self.n_heads = len(qs)

      # compute targets
      q2s = heads.tracked(pp.tracked(o2))

      s2 = pp(o2)

      # transition model
      sp = pred(s, a)
      loss_pred = tf.reduce_mean(tf.square(sp-s2))

      if double_dqn:
        a2s = [tf.argmax(_, axis=1) for _ in heads(s2)]
      else:
        a2s = [tf.argmax(_, axis=1) for _ in q2s]

      losses = []
      for a2, q2, q in zip(a2s, q2s, qs):
        mask2 = tf.one_hot(a2, n_actions, 1.0, 0.0, axis=1)
        q_target = tf.stop_gradient(tf.where(t, r, r + self.discount * tf.reduce_sum(q2 * mask2, axis=1)))

        # compute loss
        mask = tf.one_hot(a, n_actions, 1.0, 0.0, axis=1)
        q = tf.reduce_sum(q * mask, axis=1, name='q_max')
        td = tf.subtract(q_target, q, name='td')
        # td = tf.clip_by_value(td, -10, 10)
        # loss = tf.reduce_mean(tf.abs(td), axis=0, name='mae')
        # loss = tf.where(tf.abs(td) < 1.0, 0.5 * tf.square(td), tf.abs(td) - 0.5, name='mse_huber')
        losses.append(tf.reduce_mean(tf.square(td), axis=0, name='mse'))

      loss = tf.add_n(losses)

      gav = heads.compute_gradients(loss)
      if clip_gradients:
        gav = [(tf.clip_by_norm(g, clip_gradients), v) for g, v in gav]
      th = heads.apply_gradients(gav)

      gav = pp.compute_gradients(loss)
      if clip_gradients:
        gav = [(tf.clip_by_norm(g / self.n_heads, clip_gradients), v) for g, v in gav]
      tp = pp.apply_gradients(gav)

      gav = pred.compute_gradients(loss_pred)
      if clip_gradients:
        gav = [(tf.clip_by_norm(g, clip_gradients), v) for g, v in gav]
      tpred = pred.apply_gradients(gav)

      return th, tp, tpred

    self.train_step = Function(train_step,
                               prefetch_fctn=lambda: self.memory.sample_batch()[:-1],
                               prefetch_capacity=5,
                               prefetch_threads=3,
                               async=False)

    assert self.n_state

    def predict(s: [[self.n_state]], a: (tf.int32, [[]])):
      sp = pred(s, a)
      return sp
    self.predict = Function(predict)

  def train(self, timesteps=10000000, tter=.25):
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
    # saver.restore()
    debugged = False
    wt = 0.
    while self.step < timesteps:

      if self.step % 50000 == 0:
        saver.save(chi.chi.get_session(), self.logdir + '/dqn_checkpoint', global_step=self.step)

      train_debug = not debugged and self.memory.t > 512  # it is assumed the batch size is smaller than that
      debugged = debugged or train_debug
      curb = self.step > self.memory.t * tter
      if (self.memory.t > self.replay_start and not curb) or train_debug:

        if self.step % 500 == 0:
          print(f"{self.step} steps of training after {self.memory.t} steps of experience (idle for {wt} s)")
          wt = 0.

        self.train_step()

        self.step += 1
      else:
        sleep(.1)
        wt += .1

  def make_agent(self, test=False, train=True, memory_size=50000, name=None, logdir=None):
    return Agent(self.agent(test, train, memory_size, logdir + '/' + name), name, logdir)

  def agent(self, test=False, train=True, memory_size=50000, logdir=None):
    if test:
      def log_returns(rret: [], ret: [], qs, q_minus_ret, duration: []):
        layers.summarize_tensors([rret, ret, qs, q_minus_ret, duration])

      log_returns = Function(log_returns, async=True)
      memory = None
      writer = tf.summary.FileWriter(logdir)
    else:
      writer = None

    if train:
      memory = ReplayMemory(memory_size, batch_size=None)
      self.memory.children.append(memory)

    t = 0
    for ep in range(10000000000000):
      done = False
      annealing_time = 1000000
      qs = []
      unwrapped_rewards = []
      rewards = []

      ob = yield  # get initial observation
      tt = 0
      head = np.random.randint(0, self.n_heads)
      sp = np.zeros(self.n_state)
      while not done:
        q, s = self.act(ob)

        action = np.argmax(q[head])

        qs.append(q[head][action])

        smse = np.mean(np.square(sp-s))
        sp = self.predict(s, action)

        aqs = [q[h][action] for h in range(self.n_heads)]
        mq = np.mean(aqs)
        vq = np.mean(np.square(aqs - mq))
        td = qs[-2] - (rewards[-1] - self.discount * qs[-1]) if len(qs) > 1 else 0
        meta = {'action_values': q[head], 'mq': mq, 'vq': vq, 'smse': smse, 'td': td}

        if writer:
          writer.add_summary(scalar_summaries(prefix='stats', mq=mq, vq=vq, smse=smse, td=td), self.memory.t)

        ob2, r, done, info = yield action, meta  # return action and meta information and receive environment outputs

        if train:
          memory.enqueue(ob, action, r, done, info)

        ob = ob2

        rewards.append(r)
        unwrapped_rewards.append(info.get('unwrapped_reward', r))

        t += 1
        tt += 1

      if test:
        wrapped_return = sum(rewards)
        unwrapped_return = sum(unwrapped_rewards)
        discounted_returns = [sum(rewards[i:] * self.discount ** np.arange(len(rewards)-i)) for i, _ in enumerate(rewards)]
        q_minus_ret = np.subtract(qs, discounted_returns)
        log_returns(unwrapped_return, wrapped_return, qs, q_minus_ret, tt)

# Tests

# def test_dqn(env='Chain-v0'):
#   import gym_mix
#   env = gym.make(env)
#
#   def pp(x):
#     x = layers.fully_connected(x, 32)
#     x = layers.fully_connected(x, 32)
#     return x
#
#   def head(x):
#     x = layers.fully_connected(x, 32)
#     x = layers.fully_connected(x, env.action_space.n, activation_fn=None,
#                                weights_initializer=tf.random_normal_initializer(0, 1e-4))
#     return x
#
#   agent = BootstrappedDQNAg(env, pp, head, replay_start=64)
#
#   for ep in range(100000):
#     R, _ = agent.play_episode()
#
#     if ep % 100 == 0:
#       print(f'Return after episode {ep} is {R}')
