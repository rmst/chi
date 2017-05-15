import threading
from contextlib import contextmanager

import numpy as np
from chi.rl.wrappers import Wrapper
from gym import Env
from gym.spaces import Box


def print_env(env: Env):
  spec = getattr(env, 'spec', False)
  if spec:
    print(f'Env spec: {vars(spec)}')

  acsp = env.action_space
  obsp = env.observation_space

  print(f'Observation space {obsp}')
  if isinstance(obsp, Box) and len(obsp.high) < 20:
      print(f'low = {obsp.low}\nhigh = {obsp.high}')

  print(f'Action space {acsp}')
  if isinstance(acsp, Box) and len(acsp.high) < 20:
      print(f'low = {acsp.low}\nhigh = {acsp.high}')

  print("")


from matplotlib import pyplot as plt
plt.switch_backend('agg')


class Plotter:
  def __init__(self, axes: plt.Axes, timesteps=20, limits=None, auto_limit=6, title="", legend=()):
    self.auto_limit = auto_limit
    self.legend = legend
    self.x = []
    self.timesteps = timesteps
    self.limits = limits
    self.lines = []
    self.ax = axes
    self.ax.set_title(title)
    self.ax.set_xlim(-self.timesteps, 0)
    self.mean = 0
    self.var = 1
    self.y = []
    self.reset()

  def reset(self):
    self.y = []

  def append(self, data):
    self.y.append(data)
    if len(self.y) > self.timesteps:
      self.y = self.y[1:]
    self.x = range(-len(self.y), 0)

    alpha = .0001
    beta = .001
    y = np.asarray(self.y)
    mean = np.mean(y)
    self.mean = (1-alpha) * self.mean + alpha * mean
    self.var = (1-beta) * self.var + beta * np.mean(np.square(y - self.mean))

    if self.y:
      y = np.asarray(self.y)
      if not self.lines:
        self.lines = self.ax.plot(self.x, y)
        if self.legend:
          self.ax.legend(self.legend, loc='upper left')
      elif np.ndim(y) == 1:
        self.lines[0].set_data(self.x, y)
      else:
        for i, line in enumerate(self.lines):
          line.set_data(self.x, y[:, i])

      limits = self.limits or (self.mean - self.auto_limit * np.sqrt(self.var), self.mean + self.auto_limit * np.sqrt(self.var))
      self.ax.set_ylim(*limits)
      # self.ax.relim()
      # self.ax.autoscale_view()


def draw(figure):
  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  frame = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
  return frame


def concat_frames(*frames):
  h = max(f.shape[0] for f in frames)
  w = sum(f.shape[1] for f in frames)
  canvas = np.zeros((h, w, 3), dtype=np.uint8)
  wi = 0
  for f in frames:
    canvas[0: f.shape[0], wi: wi+f.shape[1], :] = f
    wi += f.shape[1]

  return canvas


class ReadWriteLock:
  """ Lock excluding writes and reads while allowing multiple reads at the same time
  Parts from: https://majid.info/blog/a-reader-writer-lock-for-python/
  """
  def __init__(self):
    self.count = 0
    self.writers_waiting = 0
    self.count_lock = threading.Lock()
    self.readers_ok = threading.Condition(self.count_lock)
    self.writers_ok = threading.Condition(self.count_lock)

  @contextmanager
  def read(self):
    self.count_lock.acquire()
    while self.count < 0 or self.writers_waiting:
      self.readers_ok.wait()
    self.count += 1
    self.count_lock.release()
    yield
    self.release()

  @contextmanager
  def write(self):
    self.count_lock.acquire()
    while self.count != 0:
      self.writers_waiting += 1
      self.writers_ok.wait()
      self.writers_waiting -= 1
    self.count = -1
    self.count_lock.release()
    yield
    self.release()

  def release(self):
    self.count_lock.acquire()
    if self.count < 0:
      self.count = 0
    else:
      self.count -= 1
    wake_writers = self.writers_waiting and self.count == 0
    wake_readers = self.writers_waiting == 0
    self.count_lock.release()
    if wake_writers:
      self.writers_ok.acquire()
      self.writers_ok.notify()
      self.writers_ok.release()
    elif wake_readers:
      self.readers_ok.acquire()
      self.readers_ok.notifyAll()
      self.readers_ok.release()