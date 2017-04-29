"""
"""
from time import sleep
import typing

from chi.rl.util import print_env, Plotter, draw

typing.Tuple
import tensorflow as tf
from chi.rl.async_dqn import DQN, DQNAgent
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

# chi.chi.tf_debug = True
from chi.rl.ddpg import DdpgAgent
from chi.rl.dqn import DqnAgent
from chi.rl.wrappers import get_wrapper, list_wrappers, DiscretizeActions, PenalizeAction
from chi.util import log_top, log_nvidia_smi, output_redirect
import logging
import threading
from time import time
from matplotlib import pyplot as plt

@experiment
def dqn_car(self: Experiment, logdir=None, frameskip=5,
            timesteps=100000000, memory_size=100000,
            agents=3,
            replay_start=50000,
            tter=.25):

  from tensorflow.contrib import layers
  import gym
  from gym import spaces
  from gym import wrappers
  import numpy as np
  from tensorflow.contrib.framework import arg_scope
  from chi.rl import ReplayMemory

  chi.set_loglevel('debug')
  log_top(logdir + '/logs/top')
  log_nvidia_smi(logdir + '/logs/nvidia-smi')

  memory = chi.rl.ReplayMemory(memory_size, 32)

  actions = [[0, 1], [0, -1],
             [1, 1], [1, -1],
             [-1, 1], [-1, -1]]
  action_names = ['fw', 'bw', 'fw_r', 'bw_r', 'fw_l', 'bw_l']

  class RenderMeta(chi.rl.Wrapper):
    def __init__(self, env, limits=None):
      super().__init__(env)
      self.an = env.action_space.n
      # self.q_plotters = [Plotter(limits=None, title=f'A({n})') for n in action_names]
      self.f, ax = plt.subplots(2, 3, figsize=(3 * 3, 2 * 2), dpi=64)
      self.f.set_tight_layout(True)
      ax = iter(np.reshape(ax, -1))
      self.q = Plotter(next(ax), title='A', legend=action_names)
      self.r = Plotter(next(ax), limits=None, title='reward')
      self.s = Plotter(next(ax), title='speed')
      self.a = Plotter(next(ax), title='av_speed')
      self.d = Plotter(next(ax), title='distance')

    def _step(self, action):
      ob, r, done, info = super()._step(action)
      qs = self.meta.get('action_values', np.full(self.an, np.nan))
      qs -= np.mean(qs)
      # [qp.append(qs[i, ...]) for i, qp in enumerate(self.q_plotters)]

      self.q.append(qs)
      self.r.append(r)
      self.s.append(info.get('speed', np.nan))
      self.a.append(info.get('average_speed', np.nan))
      self.d.append(info.get('distance_from_road', np.nan))

      return ob, r, done, info

    def _render(self, mode='human', close=False):
      f = super()._render(mode, close)
      # fs = [qp.draw() for qp in self.q_plotters]
      f2 = draw(self.f)
      return chi.rl.util.concat_frames(f, f2)

  class ScaleRewards(gym.Wrapper):
    def _step(self, a):
      ob, r, d, i = super()._step(a)
      i.setdefault('unwrapped_reward', r)
      r /= frameskip
      return ob, r, d, i

  def make_env(i):
    import rlunity
    # print(rlunity.__file__)
    env = gym.make('UnityCarPixels-v0')
    r = 256 if i == 0 else 100
    env.unwrapped.conf(loglevel='info', log_unity=True, logfile=logdir + f'/logs/unity_{i}', w=r, h=r)

    env = DiscretizeActions(env, actions)

    if i == 0:
      env = RenderMeta(env)
    env = wrappers.Monitor(env, self.logdir + '/monitor_' + str(i),
                           video_callable=lambda j: j % (20 if i == 0 else 200) == 0)

    # env = wrappers.SkipWrapper(frameskip)(env)
    # env = ScaleRewards(env)
    env = chi.rl.StackFrames(env, 4)
    return env

  envs = [make_env(i) for i in range(agents)]
  env = envs[0]
  print_env(env)

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
            q_network, memory,
            replay_start=replay_start,
            logdir=logdir)

  agents = [DQNAgent(dqn, env, test=i==0, logdir=logdir, name=f'Agent_{i}') for i, env in enumerate(envs)]

  for a in agents:
    a.start()

  dqn.train(timesteps, tter)

