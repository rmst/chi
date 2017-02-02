import gym
import universe  # register the universe environments
import numpy as np
from universe import wrappers

def test_u():
  env = gym.make('flashgames.DuskDrive-v0')
  env = wrappers.BlockingReset(env=env)
  env.configure(remotes=1)  # automatically creates a local docker container
  observation_n = env.reset()

  s = None
  for _ in range(10000):
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
    o = observation_n[0]['vision']
    assert isinstance(o, np.ndarray)
    assert not s or s == o.shape
    s = o.shape