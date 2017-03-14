import sys
import tensorflow as tf

from .main import chi, home
from .subgraph import SubGraph
from .model import Model, model
from .function import Function, function
from .app import app, App, SigtermException
from .experiment import experiment, Experiment
from .logger import set_loglevel
from . import rl
from . import ops
# __all__ = [chi, Model, Runnable]
# Override chi: only attributes of FuModule will be accessible via chi.
# sys.modules['chi'] = fu

if __name__ == '__main__':
  pass


# TODO: implement all subgraph functions
# TODO: allow root subgraph
def activations():
  assert SubGraph.stack
  return SubGraph.stack[-1].activations()


def gradients():
  assert SubGraph.stack
  return SubGraph.stack[-1].gradients()


def trainable_variables():
  assert SubGraph.stack
  return SubGraph.stack[-1].trainable_variables()


def update_ops():
  assert SubGraph.stack
  return SubGraph.stack[-1].update_ops()