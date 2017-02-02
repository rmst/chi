from .. import chi
from ..memory import ReplayMemory
import tensorflow as tf
from tensorflow.python.ops.variable_scope import get_local_variable


class Ddpg:
  warmup = 10000

  def __init__(self, pp, ac, cr, m):
    # pp = chi.Model(pp)
    # ac = chi.Model(ac)
    # cr = chi.Model(cr)

    self.m = m

    @chi.runnable
    def act(o):
      s = pp(o)
      a = ac(s, is_training=True)
      return a

    self.act = lambda o: act.run(o)

    @chi.runnable(logdir='')
    def train(o, a, r, t: bool, o2):
      s = pp(o)
      a0 = ac(s, noisevar=0.)
      q = cr(s, a0)

      q2 = cr(s, a)
      s2 = pp.tracked(o2)
      qt = cr.tracked(s2, ac.tracked(s))
      qtt = tf.cond(t, lambda: r, lambda: r + 0.99 * qt)
      mse = tf.square(q2 - qtt)

      return ac.minimize(-q), cr.minimize(mse), pp.minimize([-q, mse])

    self.step = lambda _, *args: train.run(*args)


def test_ddpg():

  import tensorflow.contrib as contrib
  layers = contrib.layers
  import gym
  from gym import spaces

  env = gym.make('MountainCarContinuous-v0')
  assert isinstance(env, gym.Env)
  assert isinstance(env.action_space, spaces.Box)
  dima = env.action_space.shape[0]
  dimo = env.observation_space.shape[0]

  m = ReplayMemory(100000, dimo, dima)

  def noise(a):
    noise_var = get_local_variable("nm", initializer=tf.zeros(a.get_shape()[1:]))
    ou_theta = get_local_variable("ou_theta", initializer=0.2)
    ou_sigma = get_local_variable("ou_sigma", initializer=0.15)
    n = noise_var.assign_sub(ou_theta * noise_var - tf.random_normal(a.get_shape()[1:], stddev=ou_sigma))
    return a + n

  @chi.model(optimizer=tf.train.AdamOptimizer(.001), tracker=tf.train.ExponentialMovingAverage(0.001))
  def pp(o: (None, dimo)):
    h1 = layers.fully_connected(o, 300)
    h2 = layers.fully_connected(h1, 300)
    h3 = layers.fully_connected(h2, 300)
    return h3

  @chi.model(optimizer=tf.train.AdamOptimizer(.001), tracker=tf.train.ExponentialMovingAverage(0.001))
  def actor(s: (None, 300), is_training=False):
    a = layers.fully_connected(s, dima)
    an = layers.utils.smart_cond(is_training, lambda: noise(a), lambda: a)
    return an

  @chi.model(optimizer=tf.train.AdamOptimizer(.0001),
             tracker=tf.train.ExponentialMovingAverage(0.001))
  def critic(s, a: (None, dima)):
    q = layers.fully_connected(s, 1, tf.identity)
    q = q + layers.fully_connected(a, 1, tf.identity)
    return q

  agent = Ddpg(pp, actor, critic, m)

  from tensorflow.contrib import framework


  tf.Graph.finalize()

  ob = env.reset()
  for _ in range(10000):
    a = agent.act(ob)
    ob, r, t, _ = env.step(a)


  #agent.train(env, timesteps=1000)

