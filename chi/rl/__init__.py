from .logger import logger, logging
from . import core
from . import chi

_default_run = core.Run()

def get_default_run():
  return _default_run

def get_flags():
  return get_default_run().flags


from . import agents