import gym
import flow
import os
import flow.logger as logger
import tensorflow as tf
def date():
  import datetime
  return datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')

with tf.Session().as_default():
  flow.logger.info("env")
  env = gym.make('Pendulum-v0')
  import pprint
  pprint.pprint(env.spec.__dict__, width=1)
  out = '{}/flow-results/{}-ddpg'.format(os.getenv("HOME"), date())
  env.monitor.start(out + '/gym')
  logger.info("Building agent...")
  agent = flow.agents.DdpgAgent(env=env, outdir=out + '/tf')
  logger.info("Playing episode....")
  agent.play()

  env.monitor.close()

def test_run_episode():
  flow.logger.info("env")
  env = gym.make('Pendulum-v0')
  import pprint
  pprint.pprint(env.spec.__dict__, width=1)
  out = '{}/flow-results/{}-ddpg'.format(os.getenv("HOME"), date())
  env.monitor.start(out+'/gym')
  logger.info("Building agent...")
  a = flow.agents.DdpgAgent(env=env, outdir=out+'/tf')
  logger.info("Playing episode....")
  a.play_episode(mode='test')
  print(a.actor.local_variables[0].eval(session=a.session))
  a.reset()
  print(a.actor.local_variables_dict)
  print(a.actor.local_variables[0].eval(session=a.session))
  # a.play()

  env.monitor.close()
