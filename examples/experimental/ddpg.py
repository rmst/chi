""" This script implements the DDPG algorithm
"""
import tensorflow as tf
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

# chi.chi.tf_debug = True


@experiment
def ddpg(self: Experiment, logdir=None, env=3):
  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope
  from chi.rl import ReplayMemory

  def gym_make(id) -> gym.Env:
    return gym.make(id)

  chi.set_loglevel('debug')

  if env == 0:
    import gym_mix
    env = gym.make('ContinuousCopyRand-v0')
    # env = envs.ContinuousCopyEnv()

    class P(gym.Wrapper):
      def _step(self, a):
        observation, reward, done, info = self.env.step(a)
        # observation = observation * np.array([1, 1, 1 / 8])
        reward *= 10
        return observation, reward, done, info
    env = P(env)
    env = wrappers.TimeLimit(env, max_episode_steps=0)
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
    env = gym_make('MountainCarContinuous-v0')
    class Obw(gym.ObservationWrapper):
      def _observation(self, o):
        return

    env = wrappers.Monitor(env, logdir + '/monitor')
  elif env == 3:
    import rlunity
    # print(rlunity.__file__)
    env = gym_make('UnityCar-v0')
    env = wrappers.SkipWrapper(1)(env)
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)

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

  def ou_noise(a, t_decay=100000):
    noise_var = get_local_variable("nm", initializer=tf.zeros(a.get_shape()[1:]))
    ou_theta = get_local_variable("ou_theta", initializer=0.2)
    ou_sigma = get_local_variable("ou_sigma", initializer=0.15)
    # ou_theta = tf.Print(ou_theta, [noise_var], 'noise: ', first_n=2000)
    ou_sigma = tf.train.exponential_decay(ou_sigma, chi.function.step(), t_decay, 1e-6)
    n = noise_var.assign_sub(ou_theta * noise_var - tf.random_normal(a.get_shape()[1:], stddev=ou_sigma))
    return a + n

  @model(optimizer=tf.train.AdamOptimizer(.001),
         tracker=tf.train.ExponentialMovingAverage(1-0.001))
  def pp(x: [obsp.shape]):
    # x = layers.batch_norm(x, trainable=False)
    x = tf.concat([tf.maximum(x, 0), -tf.minimum(x, 0)], 1)
    # x = tf.Print(x, [x], summarize=20)

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
    a = layers.fully_connected(x, acsp.shape[0], None, weights_initializer=tf.random_normal_initializer(0, 1e-4))

    def n():
      return a + tf.random_normal(tf.shape(a), mean=0, stddev=.15)

    a = smart_cond(noise, lambda: ou_noise(a), lambda: a)
    return a

  @model(optimizer=tf.train.AdamOptimizer(.001),
         tracker=tf.train.ExponentialMovingAverage(1-0.001))
  def critic(x, a: [acsp.shape]):
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    x = tf.concat([x, a], axis=1)
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
    q = layers.fully_connected(x, 1, None, weights_initializer=tf.random_normal_initializer(0, 1e-4))
    return tf.squeeze(q, 1)

  def act(o: [obsp.shape], noise=True):
    with arg_scope([layers.batch_norm], is_training=False):
      s = pp(o)
      a = actor(s, noise=noise)
      q = critic(s, a)
      layers.summarize_tensors([s, a, q])
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
    loss = - tf.reduce_mean(q, axis=0)
    return actor.minimize(loss), pp.minimize(loss)

  @chi.function
  def train_critic(o, a, r, t: tf.bool, o2):
    s = pp(o)
    q2 = critic(s, a)
    s2 = pp.tracked(o2)
    qt = critic.tracked(s2, actor.tracked(s2))
    qtt = tf.where(t, r, r + 0.99 * qt)
    mse = tf.reduce_mean(tf.square(q2 - qtt), axis=0)
    return critic.minimize(mse), pp.minimize(mse)

  @chi.function
  def log_return(r: []):
    layers.summarize_tensor(r, 'Return')

  # @chi.function
  # def train(o, a, r, t: tf.bool, o2):
  #   return train

  from itertools import count

  t = 0
  for ep in range(1000000):
    ob = env.reset()
    done = False
    R = 0
    act.initialize_local()
    # print('reset')
    for et in count():
      # a = act(ob, False) if np.random.rand() > .1 else acsp.sample()
      a = act(ob)
      ob2, r, done, _ = env.step(a)
      m.enqueue(ob, a, r, done)

      ob = ob2
      R += r

      debug_training = t == 100
      if t > 2000 or debug_training:
        mbs = 64
        for i in range(1):
          mb = m.minibatch(mbs)[:-1]
          train_critic(*mb)

        mb = m.minibatch(mbs)[:-1]
        train_actor(mb[0])

      t += 1

      if done:
        break

    log_return(R)

    if ep % 1 == 0:

      getattr(getattr(env, 'unwrapped', env), 'report', lambda: None)()

      print(f'Episode {ep}: R={R}, et={et}, t={t} -- (R_threshold = {threshold})')
