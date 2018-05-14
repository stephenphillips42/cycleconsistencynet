# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
from tqdm import tqdm

import tensorflow as tf

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

class DenseGraphLayerWeights(object):
  def __init__(self, opts, layer_lens=None, nlayers=None, activ=tf.nn.relu):
    super(DenseGraphLayerWeights, self).__init__()
    self._nlayers = 5
    if layer_lens is not None:
      self._nlayers = len(layer_lens)
      self._layer_lens = layer_lens
    else:
      if nlayers is not None:
        self._nlayers = nlayers
      self._layers_lens = \
          [ 2**max(5+k,9) for k in range(self._nlayers) ] \
          [ opts.final_embedding_dim ]
    # Build layers
    with tf.variable_scope("gnn_weights"):
      self._layers = []
      for i in range(len(self._layer_lens)-1):
        layer = tf.get_variable("weight_{:02d}".format(i),
                                [ self._layer_lens[i], self._layer_lens[i+1] ],
                                initializer=stuff)
        self._layers.append(layer)
      

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

class DenseGraphLayerWeights(object):
  def __init__(self, opts, layer_lens=None, nlayers=None, activ=tf.nn.relu):
    super(DenseGraphLayerWeights, self).__init__()
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


def train(opts):
  weights = DenseGraphLayerWeights(opts, nlayers=5)
  # print(weights._layers)
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch('train')
  print(sample)
  

if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

