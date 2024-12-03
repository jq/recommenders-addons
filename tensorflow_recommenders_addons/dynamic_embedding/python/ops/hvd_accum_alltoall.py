from horovod.tensorflow.mpi_ops import _alltoall_grad

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
  recvsplits = op.outputs[1]

  import horovod.tensorflow as hvd
  from horovod.common import process_sets

  grad_wrt_tensor, _ = hvd.alltoall(grad_wrt_output, splits=recvsplits, ignore_name_scope=ignore_name_scope,
                                process_set=process_sets._temp_process_set_object(process_set_id))
  grad_wrt_splits = None # not differentiable (integer variable)

  print(f"grad_wrt_output {grad_wrt_output} recvsplits {recvsplits}")
  print(f"grad_wrt_tensor {grad_wrt_tensor}")
  return [grad_wrt_tensor, grad_wrt_splits]

def update_alltoall_grad_func():
  _alltoall_grad.__code__ = accum_alltoall_grad.__code__
