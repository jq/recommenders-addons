import tensorflow as tf

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import default_partition_fn
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.grad_accumulator import GradAccumulator, partition, \
  GradStore


def get_keys_and_splits(rank, batch):
  if rank == 0 and batch == 0:
    vals, splits = [0, 1, 2], [1, 1, 1]
  elif rank == 1 and batch == 0:
    vals, splits = [0, 2, 11, 12], [2, 2]
  elif rank == 0 and batch == 1:
    vals, splits = [0, 1, 2], [1, 1, 1]
  elif rank == 1 and batch == 1:
    vals, splits = [1, 2, 13, 12], [2, 2]
  else:
    raise ValueError(f"rank: {rank} batch: {batch}")
  tensor = tf.convert_to_tensor(vals, dtype=tf.int64)
  tensor = tf.Variable(tensor)
  splits = tf.convert_to_tensor(splits, dtype=tf.int32)
  return tensor, splits

def test_partitions():
  keys = tf.convert_to_tensor([1, 0, 2], dtype=tf.int64)
  values = tf.convert_to_tensor([0.0, 1.0, 2.0], dtype=tf.float32)
  keys_tensor, values_tensor, partitions_sizes = partition(keys, values, 3, default_partition_fn)
  assert keys_tensor == tf.convert_to_tensor([0, 1, 2], dtype=tf.int64)
  assert values_tensor == tf.convert_to_tensor([1.0, 0.0, 2.0], dtype=tf.float32)
  assert partitions_sizes == tf.convert_to_tensor([1, 1, 1], dtype=tf.int32)
  print(f"keys_tensor {keys_tensor} values_tensor {values_tensor} partitions_sizes {partitions_sizes}")

def test_mini_batches():
  mb = GradAccumulator(2, 3)
  r0 = "rank0"
  r1 = "rank1"
  table = mb.get_accum_table(100, 16)
  grad_store = GradStore(table, None)
  k0, s0 = get_keys_and_splits(0, 0)
  grad_store.keys = k0
  mb.forward(r0, grad_store)
  grad_wrt_output = [1.0, 1.0, 1.0]
  # reshape to 3, 1
  reshaped_grad = tf.reshape(grad_wrt_output, (3, 1))

  keys, values, _ = mb.backward(r0, reshaped_grad)

  assert keys is None
  assert values is None
  k1, s1 = get_keys_and_splits(0, 1)
  grad_store.keys = k1
  mb.inc_batch()
  mb.forward(r0, grad_store)
  keys, values, split = mb.backward(r0, reshaped_grad)
  print(f"keys {keys} values {values} split {split}")
  tf.debugging.assert_equal(keys, tf.convert_to_tensor([0, 1, 2], dtype=tf.int64))
  tf.debugging.assert_equal(values, [[2.0], [2.0], [2.0]])
  tf.debugging.assert_equal(split, [1, 1, 1])



