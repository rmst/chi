import tensorflow as tf

# TODO: make chi subclass of a SubModule (with "" scope)

class ChiModule:
  def __init__(self):
    self._sess = None
    self._reg = []

  def get_session(self) -> tf.Session:
    default = tf.get_default_session()

    if not (self._sess or default):
      self._sess = tf.Session()
      return self._sess
    elif self._sess and default:
      raise RuntimeError("Session clash. Create your TF session before the first graph evaluation!")
    else:
      return default or self._sess

chi = ChiModule()