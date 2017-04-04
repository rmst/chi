from .logger import logger, logging
from . import util
import tensorflow as tf
from .main import chi


class SubGraph:
  """
  Basically just a wrapper around a variable scope
  with some convenience functions
  """
  stack = []

  def __init__(self, f, scope=None, default_name=None, getter=None):
    """

    :param scope:
    :param default_name:
    :param getter: (relative_name) -> tf.Variable
    """

    self._reused_variables = []
    self._children = []
    self._getter = getter

    if SubGraph.stack:
      SubGraph.stack[-1]._children.append(self)

    with tf.variable_scope(scope, default_name, custom_getter=self._cg) as sc:
      if hasattr(self, '_scope'):
        assert self._scope is sc
      else:
        self._scope = sc
      self.name = sc.name
      SubGraph.stack.append(self)
      self.output = f()
      self.init_op = tf.variables_initializer(self.local_variables() + self.global_variables())
      self.init_local = tf.variables_initializer(self.local_variables())
      assert SubGraph.stack.pop() is self

      # self._init_check_op = tf.report_uninitialized_variables(self._init_vars())

  def _cg(self, getter, name, *args, **kwargs):
    from .util import in_collections
    relative_name = util.relpath(name + ':0', self._scope.name)
    v = self._getter(relative_name) if self._getter else None
    if v:
      logger.debug(f'reuse {v.name} as {name} - {in_collections(v)}')
      self._reused_variables.append(v)
    else:
      v = getter(name, *args, **kwargs)
      logger.debug(f'create {v.name} - {in_collections(v)}')

    return v

  def initialize(self):  # TODO: init from checkpoint
    # names = chi.get_session().run(self._init_check_op)
    # initvs = [v for v in self._init_vars() if v.name[:-2].encode() in names]
    chi.get_session().run(self.init_op)

  def initialize_local(self):
    chi.get_session().run(self.init_local)

  def _get_reused_variables(self):
    vs = self._reused_variables
    for c in self._children:
      vs += c._get_reused_variables()
    return vs

  def get_ops(self):
    all_ops = tf.get_default_graph().get_operations()
    scope_ops = [x for x in all_ops if x.name.startswith(self._scope.name)]
    return scope_ops

  def get_collection(self, name):
    return tf.get_collection(name, self._scope.name)

  def get_ops_by_type(self, type_name):
    return [op for op in self.get_ops() if op.type == type_name]

  def get_tensors_by_optype(self, type_name):
    return [op.outputs[0] for op in self.get_ops_by_type(type_name)]

  def global_variables(self):
    return self.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  def trainable_variables(self):
    return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  def model_variables(self):
    return self.get_collection(tf.GraphKeys.MODEL_VARIABLES)

  def local_variables(self):
    return self.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

  def activations(self):
    return self.get_collection(tf.GraphKeys.ACTIVATIONS)

  def gradients(self):
    return self.get_collection('chi_gradients')

  def losses(self):
    return self.get_collection(tf.GraphKeys.LOSSES)

  def regularization_losses(self):
    return self.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  def summaries(self):
    return self.get_collection(tf.GraphKeys.SUMMARIES)

  def update_ops(self):
    return self.get_collection(tf.GraphKeys.UPDATE_OPS)


