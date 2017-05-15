from typing import Union

import gym
import numpy as np
from PIL import Image
from gym import Env, ObservationWrapper
from gym.spaces import Box, Discrete


def get_wrapper(env, wrapper_type):
  w = next(filter(lambda w: isinstance(w, wrapper_type), list_wrappers(env)), None)
  return w


def list_wrappers(env: Union[Env, gym.Wrapper]):
  while isinstance(env, gym.Wrapper):
    yield env
    env = env.env


class Wrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.meta = {}

  def set_meta(self, meta):
    self.meta = meta
    w = get_wrapper(self.env, Wrapper)
    if w:
      w.set_meta(meta)

  def _reset(self):
      observation = self.env.reset()
      return self._observation(observation)

  def observation(self, observation):
      return self._observation(observation)

  def _observation(self, observation):
      return observation

  def _step(self, action):
    action = self.action(action)
    observation, reward, done, info = self.env.step(action)
    return self.observation(observation), self.reward(reward), done, info

  def reward(self, reward):
    return self._reward(reward)

  def _reward(self, reward):
      return reward

  def action(self, action):
      return self._action(action)

  def _action(self, action):
    return action

  def reverse_action(self, action):
      return self._reverse_action(action)

  def _reverse_action(self, action):
    return action


class SkipWrapper(Wrapper):
  def __init__(self, env, repeat_count):
    super().__init__(env)
    self.repeat_count = repeat_count
    self.stepcount = 0

  def _step(self, action):
    done = False
    total_reward = 0
    total_unwrapped_reward = 0
    current_step = 0
    while current_step < (self.repeat_count + 1) and not done:
      self.stepcount += 1
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      total_unwrapped_reward += info.get('unwrapped_reward', reward)
      current_step += 1

    if 'skip.stepcount' in info:
      raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking '
                            'the SkipWrapper wrappers.')
    info['skip.stepcount'] = self.stepcount

    if 'unwrapped_reward' in info:
      info['unwrapped_reward'] = total_unwrapped_reward

    return obs, total_reward, done, info

  def _reset(self):
    self.stepcount = 0
    return self.env.reset()


class DiscretizeActions(gym.Wrapper):
  def __init__(self, env, actions):
    super().__init__(env)
    acsp = self.env.action_space
    assert isinstance(acsp, Box), "action space not continuous"
    self.actions = np.array(actions)
    assert self.actions.shape[1:] == acsp.shape, "shape of actions does not match action space"
    self.action_space = Discrete(self.actions.shape[0])

  def _step(self, action):
    a = self.actions[action]
    return super()._step(a)


class AtariWrapper(ObservationWrapper):
  """
  Pre-processing according to the following paper:
  http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
  """
  def __init__(self, env):
    super().__init__(env)
    lo = self.env.observation_space.low
    hi = self.env.observation_space.high
    w, h, c = self.env.observation_space.shape
    self.w = w
    self.h = h
    self.observation_space = Box(0, 255, [84, 84])

  def _reset(self):
    self.previous_frame = np.zeros([self.w, self.h, 3], dtype=np.uint8)
    o = super()._reset()
    return o

  def _step(self, action):
    s, r, t, i = super()._step(action)
    i.setdefault('unwrapped_reward', r)
    r = np.clip(r, -1, 1)
    return s, r, t, i

  def _observation(self, observation):
    """ Paper: First, to encode a single frame we take the maximum value for each pixel colour
        value over the frame being encoded and the previous frame. This was necessary to
        remove flickering that is present in games where some objects appear only in even
        frames while other objects appear only in odd frames, an artefact caused by the
        limited number of sprites Atari 2600 can display at once. """

    obs = np.maximum(observation, self.previous_frame)
    self.previous_frame = observation

    """ Paper: Second, we then extract
    the Y channel, also known as luminance, from the RGB frame and rescale it to
    84 x 84 """
    img = Image.fromarray(obs)
    obs = img.resize([84, 84]).convert('L')

    obs = np.asarray(obs, dtype=np.uint8)

    return obs


class StackFrames(ObservationWrapper):

  def __init__(self, env, n, dtype=np.uint8):
    super().__init__(env)
    lo = self.env.observation_space.low
    hi = self.env.observation_space.high
    self.so = self.env.observation_space.shape
    self.observation_space = Box(0, 255, [*self.so, n])
    self.n = n
    self.dtype = dtype

  def _reset(self):
    self.obs = tuple(np.zeros(self.so, dtype=self.dtype) for _ in range(self.n))
    s = super()._reset()
    return s

  def _observation(self, observation):
    self.obs = (*self.obs[1:], observation)
    return np.stack(self.obs, axis=-1)


class PenalizeAction(gym.Wrapper):
  def __init__(self, env, alpha=.01, slack=.5):
    super().__init__(env)
    self.alpha = alpha
    self.slack = slack

  def _step(self, action):
    s, r, t, i = super()._step(action)
    assert isinstance(self.env, Env)
    assert isinstance(self.env.action_space, Box)
    l = self.env.action_space.low
    h = self.env.action_space.high
    m = h - l
    dif = (action - np.clip(action, l - self.slack * m, h + self.slack * m))
    i.setdefault('unwrapped_reward', r)
    r -= self.alpha * np.mean(np.square(dif / m))
    return s, r, t, i
