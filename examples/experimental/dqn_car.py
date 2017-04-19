""" NOT FUNCTIONIONAL YET
"""
import tensorflow as tf
from chi.rl.async_dqn import AsyncDQNAgent
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

# chi.chi.tf_debug = True
from chi.rl.ddpg import DdpgAgent
from chi.rl.dqn import DqnAgent
from chi.rl.util import print_env, PenalizeAction, DiscretizeActions
from chi.util import log_top, log_nvidia_smi


@experiment
def dqn_car(self: Experiment, logdir=None, frameskip=1, T=10000000, memory_size=400000, agents=2, replay_start=50000):
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

  import rlunity
  # print(rlunity.__file__)

  def make_env(i):
    env = gym.make('UnityCarPixels-v0')
    env.unwrapped.conf(loglevel='info', log_unity=True, logfile=logdir + f'/logs/unity_{i}')
    env = DiscretizeActions(env, [[0, 1], [0, -1],
                                  [1, 1], [1, -1],
                                  [-1, 1], [-1, -1]])

    env = wrappers.SkipWrapper(frameskip)(env)
    env = wrappers.Monitor(env, logdir + '/monitor_' + str(i),
                           video_callable=lambda j: j % (50 if i == 0 else 200) == 0)

    return env

  envs = [make_env(i) for i in range(agents)]
  monitor = envs[0]

  print_env(envs[0])

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),  # TODO: replace with original weight freeze
             optimizer=tf.train.RMSPropOptimizer(.00025, .95, .95, .01))
  def q_network(x):
    x /= 255
    x = layers.conv2d(x, 32, 8, 4)
    x = layers.conv2d(x, 64, 4, 2)
    x = layers.conv2d(x, 64, 3, 1)
    x = layers.flatten(x)
    x = layers.fully_connected(x, 512)
    x = layers.fully_connected(x, envs[0].action_space.n, activation_fn=None)
    x = tf.identity(x, name='Q')
    return x

  agent = AsyncDQNAgent(envs, q_network, memory, replay_start=replay_start, logdir=logdir)

  from time import time

  agent.run_training(T)
