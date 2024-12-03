from horovod.tensorflow.mpi_ops import _alltoall_grad

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.grad_accumulator import grad_accumulator


def alltoall_grad(op, grad_wrt_output, grad_wrt_received_splits):
  """Gradient for alltoall op.

  Args:
    op: Original operation.
    grad_wrt_output: `Tensor` gradient with respect to the output of the op.
    grad_wrt_received_splits: dead argument (integer output)

  Returns:
    The gradient with respect to the input of the op.
  """
  ignore_name_scope = op.get_attr('ignore_name_scope')
  process_set_id = op.get_attr('process_set_id')
  recvsplits = op.outputs[1]

  import horovod.tensorflow as hvd
  from horovod.common import process_sets

  grad_wrt_tensor, _ = hvd.alltoall(grad_wrt_output, splits=recvsplits, ignore_name_scope=ignore_name_scope,
                                process_set=process_sets._temp_process_set_object(process_set_id))
  grad_wrt_splits = None # not differentiable (integer variable)

  print(f"grad_wrt_output {grad_wrt_output} recvsplits {recvsplits}")
  print(f"grad_wrt_tensor {grad_wrt_tensor}")
  return [grad_wrt_tensor, grad_wrt_splits]


def accum_alltoall_grad(op, grad_wrt_output, grad_wrt_received_splits):
  """Gradient for alltoall op.

  Args:
    op: Original operation.
    grad_wrt_output: `Tensor` gradient with respect to the output of the op.
    grad_wrt_received_splits: dead argument (integer output)

  Returns:
    The gradient with respect to the input of the op.
  """
  ignore_name_scope = op.get_attr('ignore_name_scope')
  process_set_id = op.get_attr('process_set_id')
  #recvsplits = op.outputs[1]
  print(f"op: {op.__dict__} ignore_name_scope {ignore_name_scope} recvsplits {recvsplits} grad_wrt_output: {grad_wrt_output}")

  import horovod.tensorflow as hvd
  from horovod.common import process_sets
  # op.outputs[0] is the value tensor map to ( op.inputs[0] -> keys, and forward path save the  op.inputs[0] -> keys)
  key = op.outputs[0]
  keys, values, split = grad_accumulator.backward(key, grad_wrt_output)
  if keys is not None:
    keys_wrt, _ = hvd.alltoall(keys, splits=split, ignore_name_scope=ignore_name_scope,
                                process_set=process_sets._temp_process_set_object(process_set_id))
    values_wrt, _ = hvd.alltoall(values, splits=split, ignore_name_scope=ignore_name_scope,
                                  process_set=process_sets._temp_process_set_object(process_set_id))
    # combine keys and values into a single tensor
    # keys [1, 2, 2] values [[1.0], [1.0], [1.0]] -> keys [1, 2] value [[1,0], [2]]
    # how to update?

  else:
    # TODO is it an issue if grad_wrt_tensor is None in case of accum
    return [None, None]


def update_alltoall_grad_func():
  _alltoall_grad.__code__ = accum_alltoall_grad.__code__
