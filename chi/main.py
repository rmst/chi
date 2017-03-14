
import os


class ChiModule:  # TODO: remove this class
  def __init__(self):
    self._sess = None

  def get_session(self):
    import tensorflow as tf
    default = tf.get_default_session()

    if not (self._sess or default):
      self._sess = tf.Session()
      return self._sess
    elif self._sess and default:
      raise RuntimeError("Session clash. Create your TF session before the first graph evaluation!")
    else:
      return default or self._sess

  def __del__(self):
    if self._sess:
      self._sess.close()

chi = ChiModule()

home = os.path.expanduser('~/.chi')


