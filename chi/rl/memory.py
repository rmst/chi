from threading import Lock

import numpy as np


class ReplayMemory:
  def __init__(self, size, batch_size=64):
    self.lock = Lock()
    self.size = size
    self.init = False
    self.rewards = np.empty(size, dtype=np.float32)
    self.terminals = np.empty(size, dtype=np.bool)
    self.info = np.empty(size, dtype=np.object)
    self.batch_size = batch_size

    self.n = 0
    self.i = -1
    self.t = -1

  def reset(self):
    self.n = 0
    self.i = -1
    self.t = -1

  def enqueue(self, observation, action, reward, terminal, info=None):
    self.lock.acquire()
    observation = np.asarray(observation)
    aa = np.asarray(action)
    if not self.init:
      self.observations = np.empty((self.size,)+observation.shape, dtype=observation.dtype)
      self.actions = np.empty((self.size,) + aa.shape, dtype=aa.dtype)
      self.init = True

    self.i = (self.i + 1) % self.size
    self.t = self.t + 1

    self.observations[self.i, ...] = observation
    self.terminals[self.i] = terminal  # tells whether this observation is the last

    self.actions[self.i, ...] = action
    self.rewards[self.i] = reward

    self.info[self.i] = info

    self.n = min(self.size, self.n + 1)
    self.lock.release()

  def sample_batch(self, size=None):
    self.lock.acquire()
    size = size or self.batch_size
    assert self.n-1 > size
    indices = np.random.randint(0, self.n - 1, size)

    o = self.observations[indices, ...]
    a = self.actions[indices]
    r = self.rewards[indices]
    t = self.terminals[indices]
    o2 = self.observations[indices + 1, ...]
    info = self.info[indices, ...]
    self.lock.release()

    return o, a, r, t, o2, info

  def __repr__(self):
    indices = range(0, self.n)
    o = self.observations[indices, ...]
    a = self.actions[indices]
    r = self.rewards[indices]
    t = self.terminals[indices]
    info = self.info[indices, ...]

    s = f"Memory with n={self.n}, i={self.i}\nOBSERVATIONS\n{o}\n\nACTIONS\n{a}\n\nREWARDS\n{r}\n\nTERMINALS\n{t}\n"

    return s


def test_sample():
  s = 100
  rm = ReplayMemory(s)

  for i in range(0, 100, 1):
    rm.enqueue(i, i, i, i % 3 == 0, i)

  for i in range(1000):
    o, a, r, t, o2, info = rm.sample_batch(10)
    assert np.all(o == o2 - 1), "error: o and o2"
    assert np.all(o != s - 1), "error: o wrap over rm. o = " + str(o)
    assert np.all(o2 != 0), "error: o2 wrap over rm"


def test_full_capacity():
  s = 5
  rm = ReplayMemory(s)

  for i in range(0, 8, 1):
    rm.enqueue(i, i, i, i % 3 == 0, i)

  assert np.all(rm.observations.flatten() == [5., 6, 7, 3, 4])


