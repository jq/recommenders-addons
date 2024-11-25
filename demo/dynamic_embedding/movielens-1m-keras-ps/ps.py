import os
import multiprocessing
import portpicker
import json

# TFRA does some patching on TensorFlow so it MUST be imported after importing TF
import tensorflow as tf
import tensorflow_recommenders_addons.dynamic_embedding as de

BATCH_SIZE = 1
NUM_WORKERS = 2
NUM_PS = 2
LOG_EVERY_N = 2


def create_in_process_cluster():
  """Creates and starts local servers and sets tf_config in the environment."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(NUM_WORKERS)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(NUM_PS)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if NUM_PS > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < NUM_WORKERS + 1:
    worker_config.inter_op_parallelism_threads = NUM_WORKERS + 1
    worker_config.intra_op_parallelism_threads = NUM_WORKERS + 1

  for i in range(NUM_WORKERS):
    tf.distribute.Server(
      cluster_spec,
      job_name="worker",
      task_index=i,
      config=worker_config,
      protocol="grpc",
    )

  ps_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < NUM_PS + 1:
    ps_config.inter_op_parallelism_threads = NUM_PS + 1
    ps_config.intra_op_parallelism_threads = NUM_PS + 1

  for i in range(NUM_PS):
    tf.distribute.Server(
      cluster_spec, job_name="ps", task_index=i, protocol="grpc", config=ps_config
    )

  chief_port = portpicker.pick_unused_port()
  cluster_dict["chief"] = [f"localhost:{chief_port}"]
  tf_config = {"cluster": cluster_dict, "task": {"type": "chief", "index": 0}}

  os.environ["TF_CONFIG"] = json.dumps(tf_config)
  return tf_config


class TestModel(tf.keras.Model):
  def __init__(self):
    super(TestModel, self).__init__()

    self.gate = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(
          3,
          use_bias=False,
          activation="softmax",
          name=f"gate",
        ),
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
      ]
    )
    self.gate_mult = tf.keras.layers.Lambda(
      lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=False)
    )

    self.emb = de.keras.layers.embedding.Embedding(
      name="my_embedding_layer",
      embedding_size=4,
      devices=[
        "/job:ps/replica:0/task:{}/device:CPU:0".format(idx)
        for idx in range(NUM_PS)
      ],
      distribute_strategy=tf.distribute.get_strategy(),
      with_unique=False,
      init_capacity=1,
    )
    self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

  def call(self, x):
    embedding = self.emb(x)
    gate = self.gate(x)
    gate_mul = self.gate_mult([gate, embedding])
    output = self.dense(gate_mul)

    return output

  def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
    data, targets = inputs
    outputs = self(data)
    loss = tf.keras.losses.BinaryCrossentropy(
      from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )(
      tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=1, dtype=tf.int64),
      outputs,
    )

    return loss


def create_coordinator():
  resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  min_shard_bytes = 256 << 10
  max_shards = NUM_PS
  variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
    min_shard_bytes=min_shard_bytes, max_shards=max_shards
  )
  strategy = tf.distribute.ParameterServerStrategy(
    resolver, variable_partitioner=variable_partitioner
  )

  coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)
  return coordinator


def launch_training():
  # This is run on chief which is the process that launches this
  coordinator = create_coordinator()

  with coordinator.strategy.scope():
    model = TestModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = de.DynamicEmbeddingOptimizer(optimizer)

  strategy = coordinator.strategy

  steps_per_invocation = 2

  @tf.function
  def worker_train_step():
    all_losses = []
    for i in range(steps_per_invocation):

      def per_replica_step(data, targets):
        with tf.GradientTape() as tape:

          per_example_loss = model.compute_loss(
            (data, targets), training=True
          )

          for var in model.trainable_variables:
            tf.debugging.check_numerics(
              var, message=f"Pre Update Variable check failed {var.name}"
            )

          loss = tf.nn.compute_average_loss(per_example_loss)

          gradients = tape.gradient(
            loss,
            model.trainable_variables,
          )
          for grad in gradients:
            tf.debugging.check_numerics(
              grad, message="Gradient check failed"
            )

        optimizer.apply_gradients(
          zip(
            gradients,
            model.trainable_variables,
          )
        )

        for var in model.trainable_variables:
          tf.debugging.check_numerics(
            var, message=f"Post Update Variable check failed {var.name}"
          )

        for var in optimizer.variables():
          if var.dtype in [tf.float16, tf.float32, tf.float64, tf.bfloat16]:
            tf.debugging.check_numerics(
              var, message="Optimizer variable check failed"
            )

        return loss

      data, target = (
        tf.random.uniform(
          (BATCH_SIZE, 1), minval=0, maxval=10000, dtype=tf.int64
        ),
        tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=1, dtype=tf.int64),
      )

      all_losses.append(strategy.run(per_replica_step, args=(data, target)))

    return strategy.reduce(tf.distribute.ReduceOp.MEAN, all_losses, axis=None)

  num_train_steps = 10000
  total_steps_to_schedule = max(num_train_steps // steps_per_invocation, 1)

  losses = []
  for i in range(1, total_steps_to_schedule + 1):
    losses.append(coordinator.schedule(worker_train_step))

    if i % LOG_EVERY_N == 0:
      coordinator.join()

      total_steps = steps_per_invocation * i
      avg_loss = tf.math.reduce_mean([loss.fetch() for loss in losses])
      print(
        f"avg loss {avg_loss} on step {i}, done a total of {steps_per_invocation} steps each step and its been, "
        f"{i} steps so, a  total of {total_steps} of batch size"
        f" {BATCH_SIZE}, "
      )
      losses = []

  coordinator.join()


if __name__ == "__main__":
  _ = create_in_process_cluster()
  launch_training()
