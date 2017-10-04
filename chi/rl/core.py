import logging
from threading import Thread

import gym
from gym.wrappers import Monitor

import chi
import tensortools as tt
from chi_rl import get_wrapper
from tensortools.util import output_redirect

from time import time
import numpy as np

class Agent:
    def __init__(self, generator, name=None, logdir=None):
        self.logdir = logdir
        self.name = name
        self.logger = logging.getLogger(self.name)
        handler = logging.FileHandler(logdir + '/logs/' + self.name) if logdir else logging.StreamHandler()
        self.logger.addHandler(handler)
        self.gen = generator
        self.gen.send(None)

    def run(self, env, episodes=100000000, async=False):
        if async:
            t = Thread(target=self.run, args=(env, episodes),
                    daemon=True, name=self.name)
            t.start()
            return

        with output_redirect(self.logger.info, self.logger.error):
            monitor = get_wrapper(env, gym.wrappers.Monitor)
            tick = time()
            for ep in range(episodes):
                self.run_episode(env)

                logi = 5
                if ep % logi == 0 and monitor:
                    assert isinstance(monitor, Monitor)
                    at = np.mean(monitor.get_episode_rewards()[-logi:])
                    ds = sum(monitor.get_episode_lengths()[-logi:])
                    dt = time() - tick
                    tick = time()

                    self.logger.info(f'av. return {at}, av. fps {ds/dt}')

    def run_episode(self, env: gym.Env):

        meta_wrapper = get_wrapper(env, .wrappers.Wrapper)

        done = False
        ob = env.reset()
        a, meta = self.act(ob)

        rs = []
        while not done:
            if meta_wrapper:
                meta_wrapper.set_meta(meta)  # send meta information to wrappers
            ob, r, done, info = env.step(a)
            a, meta = self.act(ob, r, done, info)
            rs.append(r)

        return sum(rs)

    def act(self, *args) -> tuple:
        if not self.gen:
            self.gen = self.action_generator()
            self.gen.send(None)
        args = args[0] if len(args) == 1 else args
        r = self.gen.send(args)
        r = r or (None, {})
        return r

    def action_generator(self):
        pass
