
import tensorflow as tf

threads = 4

# start tf session
sess = tf.Session(config=tf.ConfigProto(
  inter_op_parallelism_threads=threads,
  log_device_placement=False,
  allow_soft_placement=True))


