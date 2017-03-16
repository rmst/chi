from chi import rl
from . import ops
from .app import app, App, SigtermException  # lazy loader for tensorflow
from .experiment import experiment, Experiment
from .function import Function, function
from .logger import set_loglevel
from .main import chi, home
from .model import Model, model
from .subgraph import SubGraph


# TODO: implement all subgraph functions
# TODO: allow root subgraph

def get_subgraph() -> SubGraph:
  assert SubGraph.stack
  return SubGraph.stack[-1]


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


# __all__ = [chi, Model, Runnable]
# Override chi: only attributes of FuModule will be accessible via chi.
# sys.modules['chi'] = fu

if __name__ == '__main__':
  pass

