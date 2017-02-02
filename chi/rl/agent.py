#!/usr/bin/env python
import gym
import numpy as np
import tensorflow as tf
from .logger import logger

DEBUG = True


# class BaseAgentType(type):
#   def __call__(cls, *args,**kw):
#     obj = cls.__new__(cls,*args,**kw)
#     if "mock" in kw:
#       del kw["mock"]
#     obj.__init__(*args,**kw)
#     if not (obj,"_init__was_called"):
#       raise Exception("In {}.__init__(...),
#         please call super(self.__class__,self).__init__(...).".format(obj.__class__.__name__))
#     return obj
#
class Agent(object):
  """
  Reinforcement learning agent

  """
  # __metaclass__ = BaseAgentType
  VERSION = None
  GYM_ALGO_ID = None

  import flow
  flags = flow.get_flags().get_child()
  flags.env = 'Pendulum-v0', 'gym environment'
  flags.train = 10000, 'training time between tests. use 0 for test run only'
  flags.test = 10000, 'testing time between training'

  def __init__(self, **kwargs):
    """
    Args:
      **kwargs:
        env: a gym.Env
        session: a tf.Session
    """

    # make the instance attribute a child of the class attribute
    assert self.__class__.flags is self.flags
    self.flags = self.flags.get_child()
    self.flags.merge_kwargs(kwargs)
    self.flags.finalize()  # parse cmd line

    # create logging dir
    od = kwargs.get("outdir")
    if od:
      import os
      os.makedirs(od)

    self.env = kwargs.get("env")
    self.session = kwargs.get('session')

    self.t_train = 0  # global training time (increased after perceive)
    self.t_test = 0

    self._init_was_called = True
    self._init(**kwargs)

    # check if _buildGraph is implemented by subclass
    if self.__class__._build is not Agent._build:
      self.buildGraph()

  def _init(self, **kwargs):
    pass

  def buildGraph(self):
    self.session = self.session or tf.get_default_session()
    if not self.session:
      self.session = tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=4,
        log_device_placement=False,
        allow_soft_placement=True))
      self._owns_session = True

    with self.session.as_default():
      self._build()

    # initialize tf variables
    self.saver = tf.train.Saver(max_to_keep=1)
    ckpt = tf.train.latest_checkpoint(self.flags.outdir + "/tf")

    if ckpt:
      self.saver.restore(self.session, ckpt)
      self.session.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)))
    else:
      # print('\n'.join([v.name for v in tf.local_variables()+tf.all_variables()]))
      # print("init")
      self.session.run(tf.initialize_variables(
        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) +
        tf.get_collection(tf.GraphKeys.VARIABLES)))

    uninit = self.session.run(tf.report_uninitialized_variables())
    if uninit:
      raise Exception("Some variables are still uninitialized: {}".format(uninit))

    self.session.graph.finalize()

  def _build(self):
    """override to build tensorflow graph
    """
    pass

  def reset(self, **kwargs):
    """called at the start of every episode
    """
    self._reset_(**kwargs)
    self._reset(**kwargs)

  def _reset_(self, **kwargs):
    """base class and intermediate class implementation
    """
    self.mode = kwargs.get('mode', 'test')

  def _reset(self, **kwargs):
    """child class implementation
    """
    pass

  def act(self, state):
    self.state = state
    self.action = self._act(state)
    return self.action

  def _act(self, state):
    """
      return an action
    """
    raise NotImplementedError()

  def perceive(self, reward, terminal, next_state):

    self._perceive(self.state, self.action, reward, terminal, next_state)
    if self.mode == 'test':
      self.t_test += 1
    if self.mode == 'train':
      self.t_train += 1

  def _perceive(self, state, action, reward, terminal, next_state):
    """
      process environment feedback
      usually training is done here
    """
    pass

  def play(self, T=100000, mode='test-train', T_test=None, T_train=None):

    T_test = T_test or self.flags.test or self.env.spec.trials
    T_train = T_train or self.flags.train or T_test

    test_returns = []
    train_returns = []

    while self.t_train < T:

      if 'test' in mode:
        TT = self.t_test
        R = []
        while self.t_test - TT < T_test:
          R.append(self.play_episode(mode='test'))

        # if above environment return threshold continue testing up the necessary number of trials
        if self.env.spec.reward_threshold is not None and np.mean(R) > self.env.spec.reward_threshold:
          R += [self.play_episode(mode='test') for _ in range(self.env.spec.trials - len(R))]

        avr = np.mean(R)
        logger.info('Average test return\t{} after {} time steps of training'.format(avr, self.t_train))
        # save return
        test_returns.append((self.t_train, avr))
        np.save(self.flags.outdir + "/returns.npy", test_returns)

      if 'train' in mode:
        TT = self.t_train
        R = []
        while self.t_train - TT < T_train:
          R.append(self.play_episode(mode='train'))
        avr = np.mean(R)
        train_returns.append((self.t_train, avr))
        np.save(self.flags.outdir + "/train_returns.npy", train_returns)
        logger.info('Average training return\t{} after {} time steps of training'.format(avr, self.t_train))

    self.env.monitor.close()  # TODO: should monitor be modified here?

    print('play finished')

    # TODO: upload results
    # if FLAGS.upload:
    #   gym.upload(FLAGS.outdir+"/monitor",algorithm_id = self.GYM_ALGO_ID)

  def play_episode(self, mode='test'):
    from time import time
    self.env.monitor.configure(mode=mode)  # TODO: should monitor be modified here?
    state = self.env.reset()
    self.reset(mode=mode)
    i = 0
    R = 0  # return
    term = False
    _t = time()
    while not term:
      # self.env.render(mode='human')
      action = self.act(state)

      state, reward, term, info = self.env.step(action)

      self.perceive(reward, term, state)

      R += reward
      i += 1

    logger.debug('Episode finished: R={}, fps={}'.format(R, i / (time() - _t)))
    return R

  def __del__(self):
    if hasattr(self, '_owns_session'):
      self.session.close()


class RandomAgent(Agent):
  def _act(self, state):
    return self.env.action_space.sample()


if __name__ == '__main__':
  RandomAgent().play()

  # experiment.run(main)
