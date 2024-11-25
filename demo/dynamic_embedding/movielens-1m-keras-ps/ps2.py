
import os
import multiprocessing

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra
import numpy as np
import portpicker

from tensorflow.keras.layers import Dense


class NCFModel(tf.keras.Model):
  def __init__(self, use_de):
    super(NCFModel, self).__init__()
    self.embedding_size = 32
    self.use_de = use_de
    self.d0 = Dense(
      256,
      activation='relu',
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d1 = Dense(
      64,
      activation='relu',
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d2 = Dense(
      1,
      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    if use_de:
      self.user_embeddings = tfra.dynamic_embedding.get_variable(
        name="user_dynamic_embeddings",
        dim=self.embedding_size,
        initializer=tf.keras.initializers.RandomNormal(-1, 1),
        key_dtype=tf.int64)
      self.movie_embeddings = tfra.dynamic_embedding.get_variable(
        name="moive_dynamic_embeddings",
        dim=self.embedding_size,
        initializer=tf.keras.initializers.RandomNormal(-1, 1),
        key_dtype=tf.int64)
    else:
      self.user_embeddings = self.add_weight(
        name=f"user_embeddings",
        shape=(10000, self.embedding_size),
        dtype=tf.float32,
        initializer=tf.keras.initializers.RandomNormal(-1, 1),
        trainable=True,
      )
      self.movie_embeddings = self.add_weight(
        name=f"movie_embeddings",
        shape=(10000, self.embedding_size),
        dtype=tf.float32,
        initializer=tf.keras.initializers.RandomNormal(-1, 1),
        trainable=True,
      )


  def call(self, batch):
    movie_id = batch["movie_id"]
    user_id = batch["user_id"]

    trainable_wrappers = []
    if self.use_de:
      user_id_weights, user_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup_unique(
        params=self.user_embeddings,
        ids=user_id,
        name="user-id-weights",
        return_trainable=True

      )
      movie_id_weights, movie_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup_unique(
        params=self.movie_embeddings,
        ids=movie_id,
        name="movie-id-weights",
        return_trainable=True
      )
      trainable_wrappers = [user_id_trainable_wrapper, movie_id_trainable_wrapper]
    else:
      user_id_weights = tf.gather(self.user_embeddings, user_id)
      movie_id_weights = tf.gather(self.movie_embeddings, movie_id)

    embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)
    dnn = self.d0(embeddings)
    dnn = self.d1(dnn)
    dnn = self.d2(dnn)
    out = tf.reshape(dnn, shape=[-1])
    return out, trainable_wrappers


def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
      cluster_spec,
      job_name="worker",
      task_index=i,
      config=worker_config,
      protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
      cluster_spec,
      job_name="ps",
      task_index=i,
      protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
    cluster_spec, rpc_layer="grpc")
  return cluster_resolver



os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 2
NUM_PS = 1
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

strategy = tf.distribute.experimental.ParameterServerStrategy(
  cluster_resolver,
  variable_partitioner=None)

use_de = True  # code works fine if use_de=False
with strategy.scope():
  model = NCFModel(use_de)
  optimizer = tf.keras.optimizers.Adam()
  if use_de:
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

@tf.function
def step_fn(iterator):
  def replica_fn(batch):
    with tf.GradientTape() as tape:
      pred, trainable_wrappers = model(batch, training=True)
      rating = batch['user_rating']
      per_example_loss = (pred - rating)**2
      loss = tf.nn.compute_average_loss(per_example_loss)
    gradients = tape.gradient(loss, model.trainable_variables + trainable_wrappers)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables + trainable_wrappers))
    return loss

  batch_data = next(iterator)
  losses = strategy.run(replica_fn, args=(batch_data,))
  sum_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
  return sum_loss


def get_dataset_fn(input_context):
  global_batch_size = 256
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  ratings = tfds.load("movielens/100k-ratings", split="train")
  ratings = ratings.map(lambda x: {
    "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
    "user_id": tf.strings.to_number(x["user_id"], tf.int64),
    "user_rating": x["user_rating"]
  })
  wid = input_context.input_pipeline_id
  shuffled = ratings.shuffle(100_000, seed=wid, reshuffle_each_iteration=False)
  dataset_train = shuffled.take(100_000).batch(batch_size).repeat()
  return dataset_train


@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(get_dataset_fn)


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)
num_epoches = 20
steps_per_epoch = 100
for i in range(num_epoches):
  total_loss = []
  for _ in range(steps_per_epoch):
    remote = coordinator.schedule(step_fn, args=(per_worker_iterator,))
    total_loss.append(remote.fetch())
  coordinator.join()
  print("epoch", i, "loss", np.mean(total_loss))
