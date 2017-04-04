#!/usr/bin/env python
""" Tutorial for chi.experiment
    ---------------------------
    This script is just generating some data as a simple demo for chi.experiment and the chi.board.
"""
import chi


@chi.experiment  # automatically creates a unique logdir if not specified
def my_experiment(logdir, a=0.5):
  import tensorflow as tf
  from time import sleep

  print(logdir)

  @chi.function  # inherits logdir from experiment by default
  def my_function(i):
    out = i + tf.random_normal([], stddev=a)

    # summaries will be automatically written to an event file in the logdir
    # (according to a certain logging policy)
    tf.summary.scalar('out', out)

    return out

  for i in range(1000000):
    out = my_function(i)

    if i % 1000 == 0:
      print('out =', out, 'at iteration', i)


# you can watch your running experiments with chiboard

# you can also call this script from the terminal now, try:
# python experiments.py -h
