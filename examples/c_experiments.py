""" This script is just generating some data as a simple demo for chi.experiment and the chi.dashboard.
"""
import chi

chi.set_loglevel('debug')


@chi.experiment()
def my_experiment(logdir=None):  # automatically chooses a logdir
  #  import packages here
  import tensorflow as tf
  import time

  print(logdir)

  @chi.function  # inherits logdir from experiment by default
  def my_function(i):
    tf.summary.scalar('some_loss', i + tf.random_normal([], stddev=i**.5))

  for i in range(100000):
    my_function(i)


# you can watch all your running experiments with chiboard
# if the experiment page doesn't load try to refresh the site

# you can also call this script from the terminal now, try:
# python ./c_experiments.py -h
