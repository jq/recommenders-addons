from dataclasses import dataclass
import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow_recommenders_addons.dynamic_embedding import HkvHashTable
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import default_partition_fn


@dataclass
class GradStore:
  table: HkvHashTable #de.Variable #HkvHashTable
  keys: tf.Tensor

def partition(keys, values, mpi_size, partition_fn):
  partition_index = partition_fn(keys, mpi_size)
  from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import make_partition
  ids_partitions, ids_indices = make_partition(keys, partition_index, mpi_size)
  partitions_sizes = tf.stack([tf.size(p) for p in ids_partitions], axis=0)
  keys_tensor = tf.concat(ids_partitions, axis=0)
  value_partitions = tf.dynamic_partition(values, partition_index, mpi_size)
  values_tensor = tf.concat(value_partitions, axis=0)
  return keys_tensor, values_tensor, partitions_sizes


class GradAccumulator:
  def __init__(self, size, mpi_size):
    self.batch = 1
    self.size = size
    self.mpi_size = mpi_size
    self.keys = {}

  def get_accum_table(self, table_size, dim):
    # default value is vector of dim * 0
    default_value = [0.0]
    table = HkvHashTable(key_dtype=tf.int64, value_dtype=tf.float32,
      init_capacity=table_size, max_capacity=table_size, max_hbm_for_values=table_size,
      reserved_key_start_bit = 2, default_value=default_value)
    return table

  def forward(self, op, grad_store: GradStore):
    assert self.keys.get(op) is None
    self.keys[op] = grad_store

  def backward(self, op, grad_wrt_output):
    last = self.accum(op, grad_wrt_output)
    if self.batch % self.size == 0:
      table = last.table
      # print(f"last.keys {last.keys}")
      # v = table.lookup(last.keys)
      # print(f"value {v}")
      # print(f"table._value_dtype {table._value_dtype}, table._key_dtype {table._key_dtype}")
      keys, values= table.export()#export_with_scores(self.split_size)
      #table.clear()
      return partition(keys, values, self.mpi_size, default_partition_fn)
    return None, None, None

  def accum(self, op, grad_wrt_output)-> GradStore:
    last = self.keys[op]
    assert last is not None
    v, exists = last.table.lookup(last.keys, return_exists=True)
    print(f"last.keys {last.keys} grad_wrt_output {grad_wrt_output} exists {exists}, v {v}")
    last.table.accum(last.keys, grad_wrt_output, exists)
    self.keys.pop(op)
    return last

  def inc_batch(self):
    self.batch += 1

grad_accumulator = GradAccumulator(2, 3)