import chi

env = 'CartPole-v0'
# env = 'OneRoundDeterministicReward-v0'


@chi.experiment
def dqn_atari(env=env, logdir=""):
  # from PIL import Image
  import numpy as np
  import gym
  import tensorflow as tf
  from gym import wrappers
  from tensorflow.contrib import layers
  from tensorflow.contrib.framework import arg_scope
  from chi.util import in_collections

  chi.set_loglevel('debug')

  env = gym.make(env)
  if hasattr(env.observation_space, 'shape'):
    oshape = env.observation_space.shape
    obsp = env.observation_space
    print(f'Observations in [{obsp.low}, {obsp.high}]')
  else:
    oshape = [1]

  env = wrappers.Monitor(env, logdir+'/monitor')
  # wrappers.SkipWrapper(4)

  if hasattr(env, 'spec') and hasattr(env.spec, 'reward_threshold'):
    print(f'reward threshold: {env.spec.reward_threshold}')

  @chi.model(tracker=tf.train.ExponentialMovingAverage(1-.01))
  def deep_q_network(x: [oshape]):
    with arg_scope([layers.fully_connected],
                   outputs_collections=[tf.GraphKeys.ACTIVATIONS],
                   weights_regularizer=layers.l1_regularizer(.00001)):
      # x = layers.conv2d(x, 32, 8, 4)
      # x = layers.conv2d(x, 64, 4, 2)
      # x = layers.conv2d(x, 64, 3, 1)
      x = layers.fully_connected(x, 32)
      x = layers.fully_connected(x, 32)
      x = layers.fully_connected(x, env.action_space.n, activation_fn=None)
      x = tf.identity(x, name='Q')
      print(f'dqn activations: {chi.activations()}')
      return x

  deep_q_network.optimizer = tf.train.AdamOptimizer(.1)  # .00025
  m = chi.rl.ReplayMemory(500000, oshape, 1)

  @chi.function
  def act(x: [oshape]):
    qs = deep_q_network(x)
    a = tf.argmax(qs, axis=1)
    qm = tf.reduce_max(qs, axis=1)
    layers.summarize_tensor(a)
    return a, qm

  al = "\n".join([f"{v.name} - {in_collections(v)}" for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
  print('global vars \n' + al + '\n')

  @chi.function
  def train(o, a: (tf.int32, (None, 1)), r: (None,), t: (tf.bool, (None,)), o2):
    asq = tf.squeeze(a, axis=1)

    q = deep_q_network(o)
    # ac = tf.argmax(q, axis=1)

    with tf.name_scope('compute_targets'):
      q2 = deep_q_network.tracked(o2)
      q_target = tf.where(t, r, r + 0.99 * tf.reduce_max(q2, axis=1))
      q_target = tf.stop_gradient(q_target)

    oh = tf.one_hot(asq, env.action_space.n, 1.0, 0.0, axis=1)
    qs = tf.reduce_sum(q * oh, axis=1, name='q_max')
    td = tf.subtract(q_target, qs, name='td')
    # td = tf.clip_by_value(td, -10, 10)
    mse = tf.reduce_mean(tf.abs(td), axis=0, name='mae')
    # mse = tf.where(tf.abs(td) < 1.0, 0.5 * tf.square(td), tf.abs(td) - 0.5, name='mse_huber')
    # mse = tf.reduce_mean(tf.square(td), axis=0, name='mse')

    with tf.name_scope('minimize'):
      loss = deep_q_network.minimize(mse)

    with tf.name_scope('summarize'):
      print(f'train activations: {chi.activations()}')
      layers.summarize_tensors([td, mse, r, o, a,
                                tf.subtract(o2, o, name='state_dif'),
                                tf.reduce_mean(tf.cast(t, tf.float32), name='frac_terminal'),
                                tf.subtract(tf.reduce_max(q, 1, True), q, name='av_advantage')])
      layers.summarize_tensors(chi.activations())
      layers.summarize_tensors(chi.gradients())
    return loss

  @chi.function
  def log_weigths():
    v = deep_q_network.trainable_variables()
    print(f'log weights {v}')

    f = deep_q_network.tracker_variables
    print(f'log weights EMA {f}')

    difs = []
    for g in v:
      a = deep_q_network.tracker.average(g)
      difs.append(tf.subtract(g, a, name=f'ema/dif{g.name[:-2]}'))

    layers.summarize_tensors(v+f+difs)

  @chi.function
  def log_returns(R, qs):

    layers.summarize_tensors([R, qs, tf.subtract(R, qs, name='R-Q')])


  print('\n VARIABLES: \n')
  al = "\n".join([f"{v.name} - {in_collections(v)}" for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
  print(al)

  print('\n START TRAINING: \n')

  training_time = 1000000
  t = 0
  Rs = [0]
  for ep in range(1000000):
    ob = env.reset()
    done = False
    R = 0
    qs = []
    while not done:
      anneal = 1 - t / training_time
      if np.random.rand() < .05 * anneal:
        a = np.random.randint(0, env.action_space.n)
      else:
        # ob = [ob]
        a, qm = act(ob)
        qs.append(qm)

      ob2, r, done, _ = env.step(a)
      r = r
      m.enqueue(ob, a, r, done)

      ob = ob2
      R += r
      if t > 1000:
        train(*m.minibatch(32)[:-1])

      # if (t+1) % 100 == 0:
      #   print(m)
      #   exit()

      if t % 5000 == 0:
        print(f'Return average after {t} timesteps: {np.average(Rs)}')
        Rs = []

      if t % 20000 == 0:
        log_weigths()

      t += 1

    Rs.append(R)
    log_returns(R, qs)

    if t > training_time:
      break

      # dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=WINDOW_LENGTH, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, delta_range=(-1., 1.),
#                target_model_update=10000, train_interval=4)

#
# INPUT_SHAPE = (84, 84)
# WINDOW_LENGTH = 4
#
#
# class AtariProcessor(Processor):
#     def process_observation(self, observation):
#         assert observation.ndim == 3  # (height, width, channel)
#         img = Image.fromarray(observation)
#         img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
#         processed_observation = np.array(img)
#         assert processed_observation.shape == INPUT_SHAPE
#         return processed_observation.astype('uint8')  # saves storage in experience memory
#
#     def process_state_batch(self, batch):
#         # We could perform this processing step in `process_observation`. In this case, however,
#         # we would need to store a `float32` array instead, which is 4x more memory intensive than
#         # an `uint8` array. This matters if we store 1M observations.
#         processed_batch = batch.astype('float32') / 255.
#         return processed_batch
#
#     def process_reward(self, reward):
#         return np.clip(reward, -1., 1.)
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', choices=['train', 'test'], default='train')
# parser.add_argument('--env-name', type=str, default='Breakout-v0')
# parser.add_argument('--weights', type=str, default=None)
# args = parser.parse_args()
#
# # Get the environment and extract the number of actions.
# np.random.seed(123)
# env.seed(123)
# nb_actions = env.action_space.n

# We patch the environment to be closer to what Mnih et al. actually do: The environment
# repeats the action 4 times and a game is considered to be over during training as soon as a live
# is lost.
# def _step(a):
#     reward = 0.0
#     action = env._action_set[a]
#     lives_before = env.ale.lives()
#     for _ in range(4):
#         reward += env.ale.act(action)
#     ob = env._get_obs()
#     done = env.ale.game_over() or (args.mode == 'train' and lives_before != env.ale.lives())
#     return ob, reward, done, {}
# env._step = _step

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
#                               nb_steps=1000000)

# dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=WINDOW_LENGTH, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, delta_range=(-1., 1.),
#                target_model_update=10000, train_interval=4)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!