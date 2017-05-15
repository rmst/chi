"""
"""

import chi
import tensorflow as tf
from chi import experiment, Experiment
from chi.rl.async_dqn import DQN
from chi.rl.util import print_env, Plotter, draw
from chi.util import log_top, log_nvidia_smi
from matplotlib import pyplot as plt


@experiment
def dqn_atari(self: Experiment, logdir=None, env='Pong', frameskip=4,
              timesteps=100000000, memory_size=100000,
              agents=2,
              replay_start=50000,
              tter=.25,
              duelling=True):
  from tensorflow.contrib import layers
  import gym
  from gym import wrappers
  import numpy as np

  chi.set_loglevel('debug')
  log_nvidia_smi(logdir + '/logs/nvidia-smi')

  class RenderMeta(chi.rl.Wrapper):
    def __init__(self, env, limits=None):
      super().__init__(env)
      self.an = env.action_space.n
      # self.q_plotters = [Plotter(limits=None, title=f'A({n})') for n in action_names]
      self.f, ax = plt.subplots(2, 2, figsize=(2 * 3, 2 * 2), dpi=64)
      self.f.set_tight_layout(True)
      ax = iter(np.reshape(ax, -1))
      self.q = Plotter(next(ax), legend=[str(i) for i in range(self.an)], title='Q - mean Q')
      self.qm = Plotter(next(ax), title='mean Q')
      self.r = Plotter(next(ax), legend=['wrapped', 'unwrapped'], title='reward')

    def _reset(self):
      obs = super()._reset()
      self.last_frame = np.tile(obs[:, :, -1:], (1, 1, 3))

      self.last_q = None
      return obs

    def _step(self, action):
      ob, r, done, info = super()._step(action)
      self.last_frame = np.tile(ob[:, :, -1:], (1, 1, 3))
      qs = self.meta.get('action_values', np.full(self.an, np.nan))
      qm = np.mean(qs)
      qs -= qm
      # [qp.append(qs[i, ...]) for i, qp in enumerate(self.q_plotters)]
      self.qm.append(qm)
      self.q.append(qs)
      self.r.append((r, info.get('unwrapped_reward', r), self.meta.get('td', np.nan)))
      return ob, r, done, info

    def _render(self, mode='human', close=False):
      f = super()._render(mode, close)
      # fs = [qp.draw() for qp in self.q_plotters]
      f2 = draw(self.f)
      return chi.rl.util.concat_frames(f, self.last_frame, f2)

  def make_env(i):
    e = env + 'NoFrameskip-v3'
    e = gym.make(e)
    e = chi.rl.wrappers.AtariWrapper(e)
    e = chi.rl.wrappers.StackFrames(e, 4)
    e = chi.rl.wrappers.SkipWrapper(e, 4)

    if i == 0:
      e = RenderMeta(e)
    e = wrappers.Monitor(e, self.logdir + '/monitor_' + str(i),
                         video_callable=lambda j: j % (20 if i == 0 else 200) == 0 if i < 4 else False)

    return e

  envs = [make_env(i) for i in range(agents)]
  env = envs[0]
  print_env(env)

  if duelling:
    # https://arxiv.org/abs/1511.06581

    @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),  # TODO: replace with original weight freeze
               optimizer=tf.train.RMSPropOptimizer(6.25e-5, .95, .95, .01))
    def q_network(x):
      x /= 255
      x = layers.conv2d(x, 32, 8, 4)
      x = layers.conv2d(x, 64, 4, 2)
      x = layers.conv2d(x, 64, 3, 1)
      x = layers.flatten(x)

      xv = layers.fully_connected(x, 512)
      val = layers.fully_connected(xv, 1, activation_fn=None)
      # val = tf.squeeze(val, 1)

      xa = layers.fully_connected(x, 512)
      adv = layers.fully_connected(xa, env.action_space.n, activation_fn=None)

      q = val + adv - tf.reduce_mean(adv, axis=1, keep_dims=True)
      q = tf.identity(q, name='Q')
      return q
  else:
    @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),  # TODO: replace with original weight freeze
               optimizer=tf.train.RMSPropOptimizer(.00025, .95, .95, .01))
    def q_network(x):
      x /= 255
      x = layers.conv2d(x, 32, 8, 4)
      x = layers.conv2d(x, 64, 4, 2)
      x = layers.conv2d(x, 64, 3, 1)
      x = layers.flatten(x)
      x = layers.fully_connected(x, 512)
      x = layers.fully_connected(x, env.action_space.n, activation_fn=None)
      x = tf.identity(x, name='Q')
      return x

  dqn = DQN(env.action_space.n,
            env.observation_space.shape,
            q_network,
            replay_start=replay_start,
            logdir=logdir)

  for i, env in enumerate(envs):
    agent = dqn.make_agent(test=i in (0, 1), memory_size=memory_size//agents, logdir=logdir, name=f'Agent_{i}')
    agent.run(env, async=True)

  log_top(logdir + '/logs/top')

  dqn.train(timesteps, tter)
