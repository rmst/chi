from time import sleep, time

import chi
import tensortools as tt
import chi.rl.wrappers
import gym
import numpy as np
import tensorflow as tf
from tensortools import Function
from chi.rl.memory import ReplayMemory
from chi.rl.core import Agent
from chi.rl.memory import ShardedMemory
from chi.rl.wrappers import get_wrapper
from gym import wrappers
from gym.wrappers import Monitor
from tensorflow.contrib import layers


class DQN:
    """
    An implementation of
        Human Level Control through Deep Reinforcement Learning
        http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
    and
        Deep Reinforcement Learning with Double Q-learning
        https://arxiv.org/abs/1509.06461
    """

    def __init__(self, n_actions, observation_shape, q_network: tt.Model, double_dqn=True,
                             replay_start=50000, clip_td=False, logdir="", clip_gradients=10):
        self.logdir = logdir
        self.replay_start = replay_start
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.memory = ShardedMemory()
        self.discount = .99
        self.step = 0

        @tt.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),    # TODO: replace with original weight freeze
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
            return q, x


        def act(x: [observation_shape]):
            qs = q_network(x)
            a = tf.argmax(qs, axis=1)
            # qm = tf.reduce_max(qs, axis=1)
            return a, qs

        self.act = Function(act)

        def train_step(o: [observation_shape], a: (tf.int32, [[]]), r, t: tf.bool, o2: [observation_shape]):
            q = q_network(o)
            # ac = tf.argmax(q, axis=1)

            # compute targets
            q2 = q_network.tracked(o2)

            if double_dqn:
                a2 = tf.argmax(q_network(o2), axis=1)  # yep, that's really the only difference
            else:
                a2 = tf.argmax(q2, axis=1)

            mask2 = tf.one_hot(a2, n_actions, 1.0, 0.0, axis=1)
            q_target = tf.where(t, r, r + self.discount * tf.reduce_sum(q2 * mask2, axis=1))
            q_target = tf.stop_gradient(q_target)

            # compute loss
            mask = tf.one_hot(a, n_actions, 1.0, 0.0, axis=1)
            qs = tf.reduce_sum(q * mask, axis=1, name='q_max')
            td = tf.subtract(q_target, qs, name='td')
            if clip_td:
                td = tf.clip_by_value(td, -.5, .5, name='clipped_td')
            # loss = tf.reduce_mean(tf.abs(td), axis=0, name='mae')
            # loss = tf.where(tf.abs(td) < 1.0, 0.5 * tf.square(td), tf.abs(td) - 0.5, name='mse_huber')
            loss = tf.reduce_mean(tf.square(td), axis=0, name='mse')

            gav = q_network.compute_gradients(loss)
            if clip_gradients:
                gav = [(tf.clip_by_norm(g, clip_gradients), v) for g, v in gav]
            loss_update = q_network.apply_gradients(gav)

            # logging
            layers.summarize_tensors([td, loss, r, o, a,
                                                                tf.subtract(o2, o, name='state_dif'),
                                                                tf.reduce_mean(tf.cast(t, tf.float32), name='frac_terminal'),
                                                                tf.subtract(tf.reduce_max(q, 1, True), q, name='av_advantage')])
            # layers.summarize_tensors(chi.activations())
            # layers.summarize_tensors(chi.gradients())
            return loss_update

        self.train_step = Function(train_step,
                                                             prefetch_fctn=lambda: self.memory.sample_batch()[:-1],
                                                             prefetch_capacity=10,
                                                             prefetch_threads=3)

        def log_weigths():
            v = q_network.trainable_variables()
            # print(f'log weights {v}')

            f = q_network.tracker_variables
            # print(f'log weights EMA {f}')

            difs = []
            for g in v:
                a = q_network.tracker.average(g)
                difs.append(tf.subtract(g, a, name=f'ema/dif{g.name[:-2]}'))

            layers.summarize_tensors(v + f + difs)

        self.log_weights = Function(log_weigths, async=True)

    def train(self, timesteps=10000000, tter=.25):
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
        # saver.restore()
        debugged = False
        wt = 0.
        while self.step < timesteps:

            if self.step % 50000 == 0:
                saver.save(tt.get_session(), self.logdir + '/dqn_checkpoint', global_step=self.step)

            train_debug = not debugged and self.memory.t > 512  # it is assumed the batch size is smaller than that
            debugged = debugged or train_debug
            curb = self.step > self.memory.t * tter
            if (self.memory.t > self.replay_start and not curb) or train_debug:

                if self.step % 500 == 0:
                    print(f"{self.step} steps of training after {self.memory.t} steps of experience (idle for {wt} s)")
                    wt = 0.

                self.train_step()

                if self.step % 50000 == 0:
                    self.log_weights()

                self.step += 1
            else:
                sleep(.1)
                wt += .1

    def make_agent(self, test=False, memory_size=50000, name=None, logdir=None):
        return Agent(self.agent(test, memory_size), name, logdir)

    def agent(self, test=False, memory_size=50000):
        if test:
            def log_returns(rret: [], ret: [], qs, q_minus_ret, duration: []):
                layers.summarize_tensors([rret, ret, qs, q_minus_ret, duration])

            log_returns = Function(log_returns, async=True)
            memory = None

        else:
            memory = ReplayMemory(memory_size, batch_size=None)
            self.memory.children.append(memory)

        t = 0
        for ep in range(10000000000000):
            done = False
            annealing_time = 1000000
            qs = []
            unwrapped_rewards = []
            rewards = []

            ob = yield  # get initial observation
            annealing_factor = max(0, 1 - self.memory.t / annealing_time)
            tt = 0
            while not done:
                # select actions according to epsilon-greedy policy
                action, q = self.act(ob)

                if not test and (self.step == 0 or np.random.rand() < 1 * annealing_factor + .1):
                    action = np.random.randint(0, self.n_actions)

                qs.append(q[action])

                meta = {'action_values': q}
                if len(qs) > 1:
                    td = qs[-2] - (rewards[-1] - self.discount * qs[-1])
                    meta.update(td=td)

                ob2, r, done, info = yield action, meta  # return action and meta information and receive environment outputs

                if not test:
                    memory.enqueue(ob, action, r, done, info)

                ob = ob2

                rewards.append(r)
                unwrapped_rewards.append(info.get('unwrapped_reward', r))

                t += 1
                tt += 1

            if test:
                wrapped_return = sum(rewards)
                unwrapped_return = sum(unwrapped_rewards)
                discounted_returns = [sum(rewards[i:] * self.discount ** np.arange(len(rewards)-i)) for i, _ in enumerate(rewards)]
                q_minus_ret = np.subtract(qs, discounted_returns)
                log_returns(unwrapped_return, wrapped_return, qs, q_minus_ret, tt)


def deep_q_network():
    """ Architecture according to:
    http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
    """
    @tt.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),    # TODO: replace with original weight freeze
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

    return q_network


def delling_network():
    """ Architecture according to Duelling DQN:
    https://arxiv.org/abs/1511.06581
    """

    @tt.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),    # TODO: replace with original weight freeze
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


# Tests

def dqn_test(env='OneRoundDeterministicReward-v0'):
    def make_env(env=env):
        e = gym.make(env)
        e = ObservationShapeWrapper(e)
        return e

    env = make_env()
    env_test = make_env()

    @tt.model(tracker=tf.train.ExponentialMovingAverage(1-.01),
                         optimizer=tf.train.AdamOptimizer(.001))
    def q_network(x):
        x = layers.fully_connected(x, 32)
        x = layers.fully_connected(x, env.action_space.n, activation_fn=None,
                                                             weights_initializer=tf.random_normal_initializer(0, 1e-4))
        return x

    dqn = DQN(env.action_space.n, env.observation_space.shape, q_network)
    agent = dqn.make_agent()
    agent_test = dqn.make_agent(test=True)

    for ep in range(4000):
        r = agent.run_episode(env)
        if ep > 64:
            dqn.train_step()

        if ep % 100 == 0:
            rs = [agent_test.run_episode(env) for _ in range(100)]
            print(f'Return after episode {ep} is {sum(rs)/len(rs)}')


def test_dqn():
    with tf.Graph().as_default(), tf.Session().as_default():
        dqn_test()  # optimal return = 1

    with tf.Graph().as_default(), tf.Session().as_default():
        dqn_test('OneRoundNondeterministicReward-v0')  # optimal return = 1

    with tf.Graph().as_default(), tf.Session().as_default():
        dqn_test('TwoRoundDeterministicReward-v0')  # optimal return = 3


# Test Utils
class ObservationShapeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        from gym.spaces import Box
        super().__init__(env)
        self.observation_space = Box(1, 1, [1])

    def _observation(self, observation):
        return [observation]

if __name__ == '__main__':
    # chi.chi.tf_debug = True
    test_dqn()
