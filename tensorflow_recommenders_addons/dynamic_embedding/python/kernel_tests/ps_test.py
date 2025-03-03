
import os
import sys
from tensorflow.python.distribute import multi_process_lib
import multiprocessing

import contextlib
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import save as tf_save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import server_lib

class ParameterServerStrategyV2Test(test.TestCase):
  @classmethod
  def setUpClass(cls):
    super(ParameterServerStrategyV2Test, cls).setUpClass()
    cls.cluster = multi_worker_test_base.create_multi_process_cluster(
    num_workers=2, num_ps=3, rpc_layer="grpc")
    cls.cluster_resolver = cls.cluster.cluster_resolver

  @classmethod
  def tearDownClass(cls):
    super(ParameterServerStrategyV2Test, cls).tearDownClass()
    cls.cluster.stop()

  def testVariablePlacement(self):

    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      self.cluster_resolver)
    v1 = variables.Variable(initial_value=0.0)
    with strategy.scope():
      v2 = variables.Variable(initial_value=1.0)
      v3 = variables.Variable(initial_value=2.0)
      v4 = variables.Variable(initial_value=3.0)
      v5 = variables.Variable(initial_value=4.0)
    # v1 was created outside scope so should be on client.
    gpu_devices = context.num_gpus()
    if gpu_devices:
      # For tests with GPUs
      self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:GPU:0")
    else:
      self.assertEqual(v1.device, "/job:chief/replica:0/task:0/device:CPU:0")
    # v2 through v5 are created in scope and in a round-robin manner.
    self.assertEqual(v2.device, "/job:ps/replica:0/task:0/device:CPU:0")
    self.assertEqual(v3.device, "/job:ps/replica:0/task:1/device:CPU:0")
    self.assertEqual(v4.device, "/job:ps/replica:0/task:2/device:CPU:0")
    self.assertEqual(v5.device, "/job:ps/replica:0/task:0/device:CPU:0")

  def testSparselyReadForEmbeddingLookup(self):
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      self.cluster_resolver)

    class FakeModel(module.Module):

      def __init__(self):
        self._var0 = variables.Variable([1.0, 2.0, 3.0, 4.0])
        self._var1 = variables.Variable([5.0, 6.0, 7.0, 8.0])

      @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[2], dtype=dtypes.int32, name="inputs")
      ])
      def func(self, x):
        return embedding_ops.embedding_lookup([self._var0, self._var1], x)

    with strategy.scope():
      model = FakeModel()

    # Assert that ResourceGather op exists instead of Gather in training
    # function.
    found_resource_gather = False
    found_gather = False

    for n in model.func.get_concrete_function().graph.as_graph_def().node:
      if n.op == "ResourceGather":
        found_resource_gather = True
      elif n.op == "Gather":
        found_gather = True
    self.assertTrue(found_resource_gather)
    self.assertFalse(found_gather)

    # Assert that ResourceGather op exists instead of Gather in saved_model.
    found_resource_gather = False
    found_gather = False

    tmp_dir = self.get_temp_dir()
    tf_save.save(model, tmp_dir, signatures=model.func)

    with gfile.Open("%s/saved_model.pb" % tmp_dir, "rb") as f:
      saved_model_proto = saved_model_pb2.SavedModel().FromString(f.read())

    for function in saved_model_proto.meta_graphs[0].graph_def.library.function:
      for n in function.node_def:
        if n.op == "ResourceGather":
          found_resource_gather = True
          resource_gather_device = n.device
        elif n.op == "Gather":
          found_gather = True
    self.assertTrue(found_resource_gather)
    self.assertFalse(found_gather)

    # We also assert that the colocate_with in embedding_ops will not result in
    # a hard-coded device string.
    self.assertEmpty(resource_gather_device)

def custom_set_spawn_exe_path():
  print(f"custom_set_spawn_exe_path {sys.argv[0]} {os.environ['TEST_TARGET']}")
  if sys.argv[0].endswith('.py'):
    def guess_path(package_root):
      # If all we have is a python module path, we'll need to make a guess for
      # the actual executable path.
      if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
        package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
        binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
        print(f"package_root_base {package_root_base} binary {binary}")
        possible_path = os.path.join(package_root_base, package_root,
                                     binary)
        print('Guessed test binary path: %s', possible_path)
        if os.access(possible_path, os.X_OK):
          return possible_path
        return None
    path = guess_path('tf_recommenders_addons')
    if path is None:
      print(
        'Cannot determine binary path. sys.argv[0]=%s os.environ=%s',
        sys.argv[0], os.environ)
      raise RuntimeError('Cannot determine binary path')
    sys.argv[0] = path
  # Note that this sets the executable for *all* contexts.
  multiprocessing.get_context().set_executable(sys.argv[0])


if __name__ == "__main__":
  multi_process_lib._set_spawn_exe_path = custom_set_spawn_exe_path
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
