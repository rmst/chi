""" This script implements the DDPG algorithm
"""
import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import Experiment
from chi import experiment, model
from chi.rl import ReplayMemory


# chi.chi.tf_debug = True


@experiment
def bdpg(self: Experiment, logdir=None, env=1, heads=1, bootstrap=False):
  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope

  def gym_make(id) -> gym.Env:
    return gym.make(id)

  chi.set_loglevel('debug')

  if env == 0:
    import gym_mix
    env = gym.make('ContinuousCopyRand-v0')
    env = wrappers.TimeLimit(env, max_episode_steps=0)
  elif env == 1:
    env = gym.make('Pendulum-v0')

    class P(gym.Wrapper):
      def _step(self, a):
        observation, reward, done, info = self.env.step(a)
        # observation = observation * np.array([1, 1, 1 / 8])
        reward = reward - .01 * a ** 2
        reward *= .1
        return observation, reward, done, info

    env = P(env)
    # env = wrappers.Monitor(env, logdir + '/monitor', video_callable=lambda i: i % 20 == 0)
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)

  elif env == 2:
    env = gym_make('MountainCarContinuous-v0')

    class Obw(gym.ObservationWrapper):
      def _observation(self, o):
        return

    env = wrappers.Monitor(env, logdir + '/monitor')
  elif env == 3:
    import rlunity
    # print(rlunity.__file__)
    env = gym_make('UnityCar-v0')
    # env = wrappers.SkipWrapper(10)(env)
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=lambda i: True)

    env.configure(batchmode=False)

  assert isinstance(env, gym.Env)
  assert isinstance(env.action_space, spaces.Box)

  spec = getattr(env, 'spec', False)
  if spec:
    from gym.envs.registration import EnvSpec
    assert isinstance(spec, EnvSpec)
    threshold = spec.reward_threshold
    self.config.name = self.config.name + '_' + spec.id
    print(f'Env: {vars(spec)}')
  else:
    threshold = None

  acsp = env.action_space
  obsp = env.observation_space

  print(f'Continuous action space = [{acsp.high}, {acsp.low}]')
  print(f'Continuous observation space = [{obsp.high}, {obsp.low}]')

  m = ReplayMemory(100000, obsp.shape, acsp.shape)

  @model(optimizer=tf.train.AdamOptimizer(.001),
         tracker=tf.train.ExponentialMovingAverage(1 - 0.001))
  def pp(x: [obsp.shape]):
    x = layers.batch_norm(x, trainable=False)

    # x = tf.reshape(o, [tf.shape(o)[0], 32*32*3])
    # x = layers.fully_connected(x, 300)
    # x = layers.fully_connected(x, 300)
    return x

  def ac(x):
    with tf.name_scope('actor_head'):
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      a = layers.fully_connected(x, acsp.shape[0], None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
      # a = layers.fully_connected(x, acsp.shape[0], None, weights_initializer=layers.xavier_initializer())
      return a

  @model(optimizer=tf.train.AdamOptimizer(0.0001),
         tracker=tf.train.ExponentialMovingAverage(1 - 0.001))
  def actor(x, noise=False):
    actions = [ac(x) for i in range(heads)]
    # actions = tf.stack([ac(x) for i in heads], axis=1)
    # batch_size = tf.shape(actions)[0]
    # ids = tf.random_uniform(batch_size, 0, heads)
    # gids = tf.stack([tf.range(batch_size), ids])
    # action = tf.gather(actions, gids)
    # action = tf.Print(action, [actions, gids, action], 'actions, gids, action: ', 1)
    # return action, actions
    return actions

  def cr(x, a):
    with tf.name_scope('critic_head'):
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      x = tf.concat([x, a], axis=1)
      x = layers.fully_connected(x, 50, biases_initializer=layers.xavier_initializer())
      q = layers.fully_connected(x, 1, None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
      return tf.squeeze(q, 1)

  @model(optimizer=tf.train.AdamOptimizer(.001),
         tracker=tf.train.ExponentialMovingAverage(1 - 0.001))
  def critic(x, actions):
    qs = [cr(x, a) for a in actions]
    return qs

  def act(o: [obsp.shape], noise=True):
    with arg_scope([layers.batch_norm], is_training=False):
      s = pp(o)
      a = actor(s, noise=noise)
      q = critic(s, a)
      # layers.summarize_tensors([s, a, q])
      return a

  act = chi.function(act)

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

  @chi.function
  def train_actor(o):
    s = pp(o)
    a0 = actor(s)
    q = critic(s, a0)
    loss = [- tf.reduce_mean(_, axis=0) for _ in q]
    return actor.minimize(loss), pp.minimize(loss)

  @chi.function
  def train_critic(o, a: [acsp.shape], r, t: tf.bool, o2, i: tf.int32):
    s = pp(o)
    q2 = critic(s, [a for _ in range(heads)])
    s2 = pp.tracked(o2)
    qt = critic.tracked(s2, actor.tracked(s2))
    qtt = [tf.where(t, r, r + 0.99 * _) for _ in qt]

    def loss(_i, _q2, _qtt):
      sel = tf.equal(i, _i) if bootstrap else tf.fill(tf.shape(i), True)
      e = tf.where(sel, tf.square(_q2 - _qtt), tf.zeros_like(_q2))
      mse = tf.reduce_sum(e, axis=0) / tf.reduce_sum(tf.cast(sel, tf.float32), axis=0)
      return mse

    mse = [loss(_i, _q2, _qtt) for _i, (_q2, _qtt) in enumerate(zip(q2, qtt))]

    return critic.minimize(mse), pp.minimize(mse)

  @chi.function
  def log_return(r):
    layers.summarize_tensor(r, 'Return')

  # @chi.function
  # def train(o, a, r, t: tf.bool, o2):
  #   return train

  t = 0
  for ep in range(1000000):
    ob = env.reset()
    done = False
    R = 0
    act.initialize_local()
    # print('reset')
    idx = np.random.randint(0, heads)
    while not done:
      a = act(ob)
      a = a[idx]
      a = a if np.random.rand() > .1 else acsp.sample()

      ob2, r, done, _ = env.step(a)
      m.enqueue(ob, a, r, done, idx)

      ob = ob2
      R += r

      debug_training = t == 100
      if t > 2000 or debug_training:
        mbs = 64

        for i in range(1):
          mb = m.minibatch(mbs)
          train_critic(*mb)

        mb = m.minibatch(mbs)
        train_actor(mb[0])

      t += 1

    log_return(R)

    if ep % 20 == 0:
      print(f'Return of episode {ep}: {R} (head = {idx}, threshold = {threshold})')
