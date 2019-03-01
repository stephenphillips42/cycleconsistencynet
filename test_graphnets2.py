import os
import sys
import collections
import itertools
import glob
import time
import pickle
import tqdm

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

import tensorflow as tf
import sonnet as snt
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules

import options
import data_util

import pprint
pp = pprint.PrettyPrinter(indent=2)
# import pdb; pdb.set_trace()

def mygraphprint(x):
  print(x.n_node)
  print(x.n_edge)
  print(x.nodes.shape)

sys.argv.extend(['--dataset', 'spsynth0',
                 '--datasets_dir', '/data',
                 '--rome16k_dir', '/mount/data/Rome16K',
                 '--batch_size', '2'])
opts = options.get_opts()
dtype = opts.dataset_params.dtype
LD_FILE_SIZE = 128
DATASET_SIZE = 2 * LD_FILE_SIZE
BATCH_SIZE = 16
size_keys = [ 'n_node', 'n_edge' ]
size_types = [ tf.int64, tf.int64 ]
graph_keys = [ 'globals', 'nodes', 'edges', 'receivers', 'senders' ]
graph_types = [ tf.float32, tf.float32, tf.float32, tf.int64, tf.int64 ]
all_keys = size_keys + graph_keys
all_types = size_types + graph_types
SEED = 1432

class MLPInteractiveNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super().__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.InteractionNetwork(
                        edge_model_fn=lambda: snt.nets.MLP([30,20]),
                        node_model_fn=lambda: snt.nets.MLP([50,25]))

  def _build(self, inputs):
    return self._network(inputs)

print("network")
# network = MLPInteractiveNetwork()
# print(network)

print("data")
dataset = data_util.datasets.get_dataset(opts)
sample = dataset.load_batch('train')

# print("output")
# output = network(sample['graph'])
# print(output)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  pass
  init=tf.global_variables_initializer()
  sess.run(init)
  # for i in range(opts.dataset_params.sizes['train'] // opts.batch_size):
  for i in range(10):
    # import pdb; pdb.set_trace()
    o = sess.run(sample)
    print("o")
    mygraphprint(o['graph'])
    print(o['adjmat'])
    print("I FINALLY GOT THE OUTPUT :D")







