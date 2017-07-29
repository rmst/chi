"""
"""

import chi
import tensorflow as tf

from chi import Model
from chi import experiment, Experiment
from chi.rl.bdqn import BootstrappedDQN
from chi.rl.util import print_env, Plotter, draw
from chi.util import log_top, log_nvidia_smi
from matplotlib import pyplot as plt


@experiment
def dqn_atari(self: Experiment, logdir=None, env='Pong', frameskip=4,
              timesteps=100000000, memory_size=100000,
              agents=2,
              replay_start=50000,
              tter=.25,
              n_heads=3):
  from tensorflow.contrib import layers
  import gym
  from gym import wrappers
  import numpy as np

  chi.set_loglevel('info')
  log_top(logdir + '/logs/top')
  log_nvidia_smi(logdir + '/logs/nvidia-smi')

  memory = chi.rl.ReplayMemory(memory_size, 32)

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
      self.f, ax = plt.subplots(2, 3, figsize=(3 * 3, 2 * 2), dpi=64)
      self.f.set_tight_layout(True)
      ax = iter(np.reshape(ax, -1))
      self.q = Plotter(next(ax), title='A', legend=action_names)
      self.r = Plotter(next(ax), limits=None, title='reward')
      self.mq = Plotter(next(ax), title='mq')
      self.vq = Plotter(next(ax), title='vq')
      self.td = Plotter(next(ax), title='td', auto_limit=12)
      self.pe = Plotter(next(ax), title='pred.err.', auto_limit=12)

    def _step(self, action):
      ob, r, done, info = super()._step(action)
      qs = self.meta.get('action_values', np.full(self.an, np.nan))
      qs -= np.mean(qs)
      # [qp.append(qs[i, ...]) for i, qp in enumerate(self.q_plotters)]

      self.q.append(qs)
      self.r.append(r)
      self.mq.append(self.meta.get('mq', 0))
      self.vq.append(self.meta.get('vq', 0))
      self.td.append(self.meta.get('td', 0))
      self.pe.append(self.meta.get('smse', 0))

      return ob, r, done, info

    def _render(self, mode='human', close=False):
      f = super()._render(mode, close)
      # fs = [qp.draw() for qp in self.q_plotters]
      f2 = draw(self.f)
      return chi.rl.util.concat_frames(f, self.obs, f2)

    def _observation(self, observation):
      self.obs = np.tile(observation[:, :, -1:], (1, 1, 3))
      return observation

  class NoiseWrapper(chi.rl.Wrapper):
    def _reset(self):
      self.mask = np.asarray(np.random.normal(0, 20, size=self.observation_space.shape), dtype=np.uint8)

      return super()._reset()

    def _observation(self, observation):
      np.clip(observation + self.mask, 0, 255, observation)
      return observation

  env_name = env  # no clue why this is necessary

  def make_env(i):
    env = env_name + 'NoFrameskip-v3'
    env = gym.make(env)
    env = chi.rl.wrappers.AtariWrapper(env)

    if i == 1:
      env = NoiseWrapper(env)

    env = chi.rl.wrappers.StackFrames(env, 4)
    env = wrappers.SkipWrapper(4)(env)

    if i in (0, 1):
      env = RenderMeta(env)

    env = wrappers.Monitor(env, self.logdir + '/monitor_' + str(i),
                           video_callable=lambda j: j % (20 if i in (0, 1) else 200) == 0)

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
