import itertools

import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

import numpy as np

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.hvd_accum_alltoall import update_alltoall_grad_func

if hasattr(tf, 'ConfigProto'):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

class BaseTensorFlowTests(tf.test.TestCase):
  def evaluate(self, tensors):
    if context.executing_eagerly():
      return self._eval_helper(tensors)
    sess = ops.get_default_session()
    if sess is None:
      with self.test_session(config=config) as sess:
        return sess.run(tensors)
    else:
      return sess.run(tensors)
class TensorFlowTests(BaseTensorFlowTests):
  def test_horovod_alltoall_grad_cpu(self):
    """Test the correctness of the alltoall gradient on CPU."""
    hvd.init()
    update_alltoall_grad_func()
    rank = hvd.rank()
    size = hvd.size()
    if rank == 0:
      print(f"eager: {context.executing_eagerly()}")
    # As of TensorFlow v1.9, gradients are not supported on
    # integer tensors
    dtypes = [tf.float32]#, tf.float64]
    dims = [1]#, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
      with tf.device("/cpu:0"):
        vals = []
        for i in range(size):
          vals += [i] * (rank+1)
        print(f"rank: {rank} vals: {vals} size: {len(vals)}")
        tensor = tf.convert_to_tensor(vals, dtype=dtype)
        for _ in range(dim - 1):
          tensor = tf.expand_dims(tensor, axis=1)
          tensor = tf.concat([tensor, tensor], axis=1)

        if context.executing_eagerly():
          tensor = tf.Variable(tensor)
          splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
          with tf.GradientTape() as tape:
            collected, received_splits = hvd.alltoall(tensor, splits)
        else:
          splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
          collected, received_splits = hvd.alltoall(tensor, splits)
        print(f"rank: {rank} collected: {collected}")
        grad_ys = tf.ones(tf.shape(collected))
        if context.executing_eagerly():
          grad_out = tape.gradient(collected, tensor, grad_ys)
        else:
          grad = tf.gradients(collected, tensor, grad_ys)[0]
          grad_out = self.evaluate(grad)
      print(f"rank: {rank} grad_out: {grad_out}")
      expected = np.ones(tensor.get_shape().as_list())
      err = np.linalg.norm(expected - grad_out)
      self.assertLess(err, 0.00000001,
                      "gradient %s differs from expected %s, "
                      "error: %s" % (grad_out, expected, str(err)))
