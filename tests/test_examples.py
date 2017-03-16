import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_all():
  from examples import functions
  from examples import models


def test_experiment():
  from examples import experiments
  experiments.my_experiment()