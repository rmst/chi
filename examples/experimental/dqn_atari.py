"""
Paper:
http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
"""

import chi
import chi.rl.wrappers
from chi.rl.dqn import DqnAgent
from chi.rl.util import print_env
from chi.util import run_parallel, log_top, log_nvidia_smi


@chi.experiment
def dqn_atari(logdir, env='Pong', memory_size=100000):
  import numpy as np
  import gym
  import tensorflow as tf
  from gym import wrappers
  from tensorflow.contrib import layers
  from tensorflow.contrib.framework import arg_scope
  from chi.util import in_collections

  chi.set_loglevel('debug')
  log_top(logdir+'/logs/top')
  log_nvidia_smi(logdir+'/logs/nvidia-smi')


  env += 'NoFrameskip-v3'
  env = gym.make(env)
  env = chi.rl.wrappers.AtariWrapper(env)
  env = chi.rl.wrappers.StackFrames(env, 4)
  env = wrappers.SkipWrapper(4)(env)

  test = 10
  train = 40
  env = monitor = wrappers.Monitor(env, logdir+'/monitor', video_callable=lambda i: i % (test+train) == 0 or i % (test+train) == train)

  print_env(env)

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1-.0005),  # TODO: replace with original weight freeze
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

  memory = chi.rl.ReplayMemory(memory_size, 32)

  agent = DqnAgent(env, q_network, memory)

  from time import time
  step = monitor.get_total_steps()
  t = time()
  for ep in range(100000):
    for _ in range(train):
      agent.play_episode()

    for _ in range(test):
      agent.play_episode(test=True)

    ar = np.mean(monitor.get_episode_rewards()[-(train+test):-test])
    at = np.mean(monitor.get_episode_rewards()[-test:])
    ds = monitor.get_total_steps() - step
    step = monitor.get_total_steps()
    dt = time() - t
    t = time()
    print(f'av. test return {at}, av. train return {ar}, av. fps {ds/dt}')