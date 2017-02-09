import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_all():
  from examples import a_functions

  from examples import b_models

  from examples import c_experiments