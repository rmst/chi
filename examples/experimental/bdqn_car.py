"""
"""

import chi
import tensorflow as tf
from chi import experiment, Experiment
from chi.rl.async_dqn import DQN
from chi.rl.bdqn import BootstrappedDQN
from chi.rl.util import print_env, Plotter, draw
from chi.rl.wrappers import DiscretizeActions
from chi.util import log_top, log_nvidia_smi
from matplotlib import pyplot as plt
import numpy as np


@experiment
def dqn_car(self: Experiment, logdir=None, frameskip=5,
            timesteps=100000000, memory_size=100000,
            agents=3,
            replay_start=50000,
            tter=.25,
            n_heads=3):
  from tensorflow.contrib import layers
  import gym
  from gym import wrappers
  import numpy as np

  chi.set_loglevel('debug')
  log_top(logdir + '/logs/top')
  log_nvidia_smi(logdir + '/logs/nvidia-smi')

  actions = [[0, 1], [0, -1],
             [1, 1], [1, -1],
             [-1, 1], [-1, -1]]
  action_names = ['fw', 'bw', 'fw_r', 'bw_r', 'fw_l', 'bw_l']

  class RenderMeta(chi.rl.Wrapper):
    def __init__(self, env, limits=None, mod=False):
      super().__init__(env)
      self.mod = mod
      self.an = env.action_space.n
      # self.q_plotters = [Plotter(limits=None, title=f'A({n})') for n in action_names]
      self.f, ax = plt.subplots(2, 4, figsize=(4 * 3, 2 * 2), dpi=64)
      self.f.set_tight_layout(True)
      ax = iter(np.reshape(ax, -1))
      self.q = Plotter(next(ax), title='Q - mean Q', legend=action_names)
      self.mq = Plotter(next(ax), title='mean Q')
      self.r = Plotter(next(ax), limits=None, title='reward')
      self.s = Plotter(next(ax), title='speed')
      self.a = Plotter(next(ax), title='av_speed')
      self.d = Plotter(next(ax), title='distance_from_road')
      self.td = Plotter(next(ax), title='td')

      if mod:
        self.mask = np.asarray(np.random.normal(0, 30, size=self.observation_space.shape), dtype=np.uint8)

    def _step(self, action):
      ob, r, done, info = super()._step(action)
      qs = self.meta.get('action_values', np.full(self.an, np.nan))
      mq = np.mean(qs)
      qs -= mq
      # [qp.append(qs[i, ...]) for i, qp in enumerate(self.q_plotters)]

      self.q.append(qs)
      self.mq.append(mq)
      self.r.append(r)
      self.s.append(info.get('speed', np.nan))
      self.a.append(info.get('average_speed', np.nan))
      self.d.append(info.get('distance_from_road', np.nan))
      self.td.append(self.meta.get('td', 0))

      return ob, r, done, info

    def _render(self, mode='human', close=False):
      f = super()._render(mode, close)
      # fs = [qp.draw() for qp in self.q_plotters]
      f2 = draw(self.f)
      obs = np.tile(self.obs[:, :, np.newaxis], (1, 1, 3))
      return chi.rl.util.concat_frames(f, obs, f2)

    def _observation(self, observation):
      if self.mod:
        np.clip(observation + self.mask, 0, 255, observation)
      self.obs = observation
      return observation

  class ScaleRewards(gym.Wrapper):
    def _step(self, a):
      ob, r, d, i = super()._step(a)
      i.setdefault('unwrapped_reward', r)
      r /= frameskip
      return ob, r, d, i

  def make_env(i):
    import rlunity
    env = gym.make('UnityCarPixels-v0')
    r = 100
    env.unwrapped.conf(loglevel='info', log_unity=True, logfile=logdir + f'/logs/unity_{i}', w=r, h=r)

    env = DiscretizeActions(env, actions)

    if i in (0, 1):
      env = RenderMeta(env, mod=i == 1)
    env = wrappers.Monitor(env, self.logdir + '/monitor_' + str(i),
                           video_callable=lambda j: j % (20 if i in (0, 1) else 200) == 0)

    # env = wrappers.SkipWrapper(frameskip)(env)
    # env = ScaleRewards(env)
    env = chi.rl.StackFrames(env, 4)
    return env

  envs = [make_env(i) for i in range(agents)]
  env = envs[0]
  print_env(env)

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
             optimizer=tf.train.RMSPropOptimizer(6.25e-5, .95, .95, .01))
  def pp(x):
    x /= 255
    x = layers.conv2d(x, 32, 8, 4)
    x = layers.conv2d(x, 64, 4, 2)
    x = layers.conv2d(x, 64, 3, 1)
    x = layers.flatten(x)
    return x

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
             optimizer=tf.train.RMSPropOptimizer(6.25e-5, .95, .95, .01))
  def heads(x):
    qs = []
    for _ in range(n_heads):
      xv = layers.fully_connected(x, 512)
      val = layers.fully_connected(xv, 1, activation_fn=None)
      # val = tf.squeeze(val, 1)

      xa = layers.fully_connected(x, 512)
      adv = layers.fully_connected(xa, env.action_space.n, activation_fn=None)

      q = val + adv - tf.reduce_mean(adv, axis=1, keep_dims=True)
      q = tf.identity(q, name='Q')
      qs.append(q)

    return qs

  dqn = BootstrappedDQN(env.action_space.n,
                        env.observation_space.shape,
                        pp,
                        heads,
                        replay_start=replay_start,
                        logdir=logdir)

  for i, env in enumerate(envs):
    agent = dqn.make_agent(test=i in (0, 1), train=i != 1, memory_size=memory_size // (agents-1), logdir=logdir, name=f'Agent_{i}')
    agent.run(env, async=True)

  dqn.train(timesteps, tter)