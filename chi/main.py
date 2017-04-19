
import os


class ChiModule:  # TODO: remove this class
  def __init__(self):
    self._sess = None
    self.tf_debug = False

  def get_session(self):
    import tensorflow as tf

    default = tf.get_default_session()

    if not (self._sess or default):
      # config = tf.ConfigProto(
      #   intra_op_parallelism_threads=2,
      #   inter_op_parallelism_threads=4)
      # config.gpu_options.allow_growth = True
      config = None
      self._sess = tf.Session(config=config)
      s = self._sess
    elif self._sess and default:
      raise RuntimeError("Session clash. Create your TF session before the first graph evaluation!")
    else:
      s = default or self._sess

    if self.tf_debug:
      from tensorflow.python import debug as tf_debug
      s = tf_debug.LocalCLIDebugWrapperSession(s)

    return s

  def __del__(self):
    if self._sess:
      self._sess.close()

chi = ChiModule()

home = os.path.expanduser('~/.chi')


