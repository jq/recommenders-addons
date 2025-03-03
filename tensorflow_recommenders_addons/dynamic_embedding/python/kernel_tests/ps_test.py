import contextlib
import functools
import os
import sys

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

from tensorflow.python.distribute import multi_process_lib
import multiprocessing

def custom_set_spawn_exe_path():
  path = sys.executable + " " + os.path.join(os.path.dirname(__file__), "ps_test.py")
  path = "pytest " + os.path.join(os.path.dirname(__file__), "ps_test.py")
  print(f"custom_set_spawn_exe_path {path}")
  multiprocessing.get_context().set_executable(path)
  # multiprocessing.get_context().set_executable("/Users/jqian2/src/recommenders-addons/tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/ps_test.py")
multi_process_lib._set_spawn_exe_path = custom_set_spawn_exe_path

# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
#
# os.environ["PYTHON_EXECUTABLE"] = sys.executable


if __name__ == "__main__":
  print("xxxx")
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
