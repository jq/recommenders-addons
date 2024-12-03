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

def get_keys_and_splits(rank, batch):
  if rank == 0 and batch == 0:
    vals, splits = [0, 10], [1, 1]
  elif rank == 1 and batch == 0:
    vals, splits = [0, 2, 11, 12], [2, 2]
  elif rank == 0 and batch == 1:
    vals, splits = [0, 10], [1, 1]
  elif rank == 1 and batch == 1:
    vals, splits = [1, 2, 13, 12], [2, 2]
  else:
    raise ValueError(f"rank: {rank} batch: {batch}")
  tensor = tf.convert_to_tensor(vals)
  tensor = tf.Variable(tensor)
  splits = tf.convert_to_tensor(splits, dtype=tf.int32)
  return tensor, splits

# batch 1 - key, no-dup case.
# key [0, 10, 20]  splits [1, 1, 1],
# key [0, 2, 11, 12, 20, 22] splits [2, 2, 2]
# key [1, 10, 20]  splits [1, 1, 1],
# key - all2all
# 0: [0, 0, 1] [1, 1, 1]
# 1: [10, 11, 12, 10] [1, 2, 1]
# 2: [20, 20, 22, 20] [1, 2, 1]
# key -> value same as key
# value all2all forward step
# [0, 10, 20]  splits [1, 1, 1],
# key [0, 2, 11, 12, 20, 22] splits [2, 2, 2]
# key [1, 10, 20]  splits [1, 1, 1],
# backward step, no network, insert with grad_wrt_output
# 0: {0->g0, 10->g10, 20->g20} [1, 1, 1], key is batch1's key, value is grad_wrt_output
# 1: {0, 2, 11, 12, 20, 22} [2, 2, 2]
# 2: {1, 10, 20} [1, 1, 1]
# batch 2 - key
# key [0, 10, 20]  splits [1, 1, 1],
# [0, 2, 13, 12, 21, 22] splits [2, 2, 2]
# [1, 10, 21]  splits [1, 1, 1],
# key - all2all
# 0: [0, 0, 1] [1, 1, 1]
# 1: [10, 13, 12, 10] [1, 2, 1]
# 2: [20, 21, 22, 21] [1, 2, 1]
# key -> value same as key
# value all2all forward step
# [0, 10, 20]  splits [1, 1, 1],
# [0, 2, 13, 12, 21, 22] splits [2, 2, 2]
# [1, 10, 21]  splits [1, 1, 1],
# backward step
# 0: {0->g0, 10->g10, 20->g20} [1, 1, 1] accum with key [0, 10, 20], grad_wrt_output
# and then export,  and then all2all with new split
# accum(self, keys, values_or_deltas, exists) how to use exists to gen new split?
# maybe best is not to use exists, but use partition of keys to gen new split
# export_with_scores(self, split_size,) spit_size is the world size.
# all2all key [0, 10, 20]  value [g0, g10, g20]  splits [1, 1, 1]
# receive
# 0: key[0,0,1] value[g0, g0, g1]

class TensorFlowTests(BaseTensorFlowTests):
  def horovod_alltoall_lookup_grad_cpu(self):
    hvd.init()
    update_alltoall_grad_func()
    rank = hvd.rank()
    size = hvd.size()
    for batch in range(2):
      with tf.device("/cpu:0"):
        vals, splits = get_keys_and_splits(rank, batch)
        print(f"rank: {rank} vals: {vals} splits: {splits}")

        # tensor rank 0 [0,1] splits [1,1] rank1 [0, 0, 1, 1]  splits [2, 2]
        def look_up_table(key):
          return key
        key, key_splits = hvd.alltoall(vals, splits)
        # key rank 0 [0, 0, 0] key_splits [1 2], rank1 [1, 1, 1] key_splits [1 2]
        with tf.GradientTape() as tape:
          # the output of alltoall is not TF.Variable, so we need to assign it to a variable
          # otherwize, the tape.gradient will return None
          local_value = tf.Variable(look_up_table(key))
          value, value_splits = hvd.alltoall(local_value, key_splits)
          # value rank 0 [0,1] value_splits [1,1] rank1 [0, 0, 1, 1]  value_splits [2, 2]
        print(f"rank: {rank} value: {value} value_splits: {value_splits} splits: {splits}")
        grad_ys = tf.ones(tf.shape(value))
        grad_out = tape.gradient(value, local_value, grad_ys)
        if batch == 0:
          expected = np.ones(key.get_shape().as_list())

  def test_horovod_alltoall_lookup_grad_cpu(self):
    """Test the correctness of the alltoall gradient on CPU."""
    # if context.executing_eagerly():
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

        tensor = tf.Variable(tensor)
        splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
        # tensor rank 0 [0,1] splits [1,1] rank1 [0, 0, 1, 1]  splits [2, 2]
        def look_up_table(key):
          return key
        key, key_splits = hvd.alltoall(tensor, splits)
        # key rank 0 [0, 0, 0] key_splits [1 2], rank1 [1, 1, 1] key_splits [1 2]
        with tf.GradientTape() as tape:
          # the output of alltoall is not TF.Variable, so we need to assign it to a variable
          # otherwize, the tape.gradient will return None
          local_value = tf.Variable(look_up_table(key))
          value, value_splits = hvd.alltoall(local_value, key_splits)
          # value rank 0 [0,1] value_splits [1,1] rank1 [0, 0, 1, 1]  value_splits [2, 2]
        print(f"rank: {rank} local_value {local_value} value: {value} value_splits: {value_splits} splits: {splits}")
        grad_ys = tf.ones(tf.shape(value))
        grad_out = tape.gradient(value, local_value, grad_ys)
        # grad_out rank 0 [1, 1, 1] rank 1 [1, 1, 1]

      print(f"rank: {rank} grad_ys: {grad_ys}, grad_out: {grad_out}")
      expected = np.ones(key.get_shape().as_list())
      err = np.linalg.norm(expected - grad_out)
      self.assertLess(err, 0.00000001,
                      "gradient %s differs from expected %s, "
                      "error: %s" % (grad_out, expected, str(err)))

  def horovod_alltoall_grad_cpu(self):
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
        print(f"rank: {rank} collected: {collected} received_splits: {received_splits} splits: {splits}")
        grad_ys = tf.ones(tf.shape(collected))
        if context.executing_eagerly():
          grad_out = tape.gradient(collected, tensor, grad_ys)
        else:
          grad = tf.gradients(collected, tensor, grad_ys)[0]
          grad_out = self.evaluate(grad)
      print(f"rank: {rank} grad_ys: {grad_ys}, grad_out: {grad_out}")
      expected = np.ones(tensor.get_shape().as_list())
      err = np.linalg.norm(expected - grad_out)
      self.assertLess(err, 0.00000001,
                      "gradient %s differs from expected %s, "
                      "error: %s" % (grad_out, expected, str(err)))
