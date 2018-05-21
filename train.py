# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import scipy.linalg as la
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step

import data_util
import myutils
import options

class MyLogger(object):
  def __init__(self, logfile_name):
    self.logfile = open(logfile_name, 'w')

  def log(self, message):
    print(message)
    self.logfile.write(message + '\n')

  def __del__(self):
    self.logfile.close()


def build_optimizer(opts, global_step):
  # Learning parameters post-processing
  num_batches = 1.0 * opts.sample_sizes['train'] / opts.batch_size
  decay_steps = int(num_batches * opts.learning_rate_decay_epochs)
  if opts.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(opts.learning_rate, name='fixed_learning_rate')
  elif opts.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                               global_step,
                                               decay_steps,
                                               opts.learning_rate_decay_rate,
                                               staircase=True,
                                               name='learning_rate')
  elif opts.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.train.polynomial_decay(opts.learning_rate,
                                              global_step,
                                              decay_steps,
                                              opts.min_learning_rate,
                                              power=1.0,
                                              cycle=False,
                                              name='learning_rate')

  if opts.full_tensorboard:
    tf.summary.scalar('learning_rate', learning_rate)
  # TODO: add individual adam options to these
  if opts.optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif opts.optimizer_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif opts.optimizer_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate)
  elif opts.optimizer_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  return optimizer

def matmul(x,y):
  return tf.einsum('bik,kj->bij', x, y)

def batch_matmul(x,y):
  return tf.einsum('bik,bkj->bij', x, y)

class DenseGraphLayerWeights(object):
  def __init__(self, opts, layer_lens=None, nlayers=None, activ=tf.nn.relu):
    super(DenseGraphLayerWeights, self).__init__()
    self._activ = activ
    self._nlayers = 5
    if layer_lens is not None:
      self._nlayers = len(layer_lens)
      self._layer_lens = layer_lens
    else:
      if nlayers is not None:
        self._nlayers = nlayers
      self._layer_lens = \
          [ opts.descriptor_dim ] + \
          [ 2**min(5+k,9) for k in range(self._nlayers) ] + \
          [ opts.final_embedding_dim ]
    # Build layers
    with tf.variable_scope("gnn_weights"):
      self._layers = []
      for i in range(len(self._layer_lens)-1):
        layer = tf.get_variable("weight_{:02d}".format(i),
                                [ self._layer_lens[i], self._layer_lens[i+1] ],
                                initializer=tf.random_normal_initializer())
        self._layers.append(layer)

  def apply(self, sample):
    """Applying this graph network to sample"""
    print
    lap = sample['Laplacian']
    init_emb = sample['InitEmbeddings']
    print(lap)
    print(init_emb)
    output = init_emb
    for l in range(self._nlayers):
      lin = matmul(output, self._layers[l])
      lin_graph = batch_matmul(lap, lin)
      output = self._activ(lin_graph)
    output = matmul(output, self._layers[-1])
    output = tf.nn.l2_normalize(output, axis=2)
    return output

def get_sim(x):
  x_T = tf.transpose(x, perm=[0, 2, 1])
  return batch_matmul(x, x_T)

def train(opts):
  # Get data
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch('train')
  # test_sample = dataset.load_batch('test')

  # Get network
  network = DenseGraphLayerWeights(opts) # Just use default options

  # Get loss
  emb = sample['TrueEmbedding']
  output = network.apply(sample)
  emb_sim, output_sim = get_sim(emb), get_sim(output)
  tf.summary.image('Output Similarity', output_sim)
  tf.summary.image('Embedding Similarity', emb_sim)
  tf.losses.mean_squared_error(emb_sim, output_sim)
  loss = tf.losses.get_total_loss()

  # # Get evaluation loss
  # test_emb = test_sample['TrueEmbedding']
  # test_output = network.apply(test_sample)
  # test_mse = tf.metrics.mean_squared_error(get_sim(test_emb),get_sim(test_output))

  # Training objects
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = slim.learning.create_train_op(total_loss=loss,
                                           optimizer=optimizer,
                                           global_step=global_step,
																					 clip_gradient_norm=5)

  tf.logging.set_verbosity(tf.logging.INFO)
  # num_batches = int(1.0 * opts.sample_sizes['train'] / opts.batch_size)
  num_batches = 1.0 * opts.sample_sizes['train'] / opts.batch_size
  max_steps = int(num_batches * opts.num_epochs)

  # Train-test loop
  slim.learning.train(
          train_op=train_op,
          logdir=opts.save_dir,
          number_of_steps=max_steps,
          log_every_n_steps=opts.log_steps,
          save_summaries_secs=opts.save_summaries_secs,
          save_interval_secs=opts.save_interval_secs)



if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

