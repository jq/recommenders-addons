import unittest

from tensorflow.python.distribute import multi_worker_test_base

from tensorflow_recommenders_addons.dynamic_embedding.python.train.utils import worker_devices


class TestWorkerDevices(unittest.TestCase):

  def test_valid_cases(self):
    self.assertEqual(
      worker_devices(["/device:CPU:0", "/device:CPU:1"], 4, "worker"), [
        "/job:worker/task:0/device:CPU:0",
        "/job:worker/task:1/device:CPU:1",
        "/job:worker/task:2/device:CPU:0",
        "/job:worker/task:3/device:CPU:1",
      ])
    self.assertEqual(worker_devices(["/device:GPU:0"], 2, "worker"), [
      "/job:worker/task:0/device:GPU:0",
      "/job:worker/task:1/device:GPU:0",
    ])

  def test_invalid_cases(self):
    with self.assertRaises(ValueError):
      worker_devices(["/device:CPU:0", "/device:CPU:1"], 3, "worker")