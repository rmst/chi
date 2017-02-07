""" This script is just generating some data as a simple demo for chi.experiment and the chi.dashboard.
"""
import chi
import tensorflow as tf
import time


@chi.experiment
def my_experiment(logdir='~/chi-results/+'):

  @chi.function(logdir=logdir)
  def my_function(i):
    tf.summary.scalar('some_loss', i + tf.random_normal([], stddev=i**.5))

  for i in range(10000):
    my_function(i)
    time.sleep(.01)