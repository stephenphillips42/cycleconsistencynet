# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import glob
import datetime
import tqdm
import networkx as nx

import tensorflow as tf
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules

from data_util import mydataset
from data_util import tf_helpers

def np_dense_to_sparse(arr):
  idx = np.where(arr != 0.0)
  return idx, arr[idx]

GRAPH_KEYS = [
  'n_node', 'nodes', 'n_edge', 'edges', 'receivers', 'senders', 'globals'
]

class GraphDataset(object):
  MAX_IDX=700
  def __init__(self, opts, params):
    self.opts = opts
    self.dataset_params = params
    self.data_dir = params.data_dir
    self.dtype = params.dtype
    p = params.points[-1]
    v = params.views[-1]
    self.n_views = v 
    self.n_pts = p
    d = p*v
    e = params.descriptor_dim
    # GraphsTuple(nodes=nodes,
    #           edges=edges,
    #           globals=globals,
    #           receivers=receivers,
    #           senders=senders,
    #           n_node=n_node,
    #           n_edge=n_edge)
    self.features = {
      'n_node':
           tf_helpers.Int64Feature(
                         key='n_node',
                         dtype='int32',
                         description='Number of nodes we are using'),
      'nodes':
           tf_helpers.TensorFeature(
                         key='nodes',
                         shape=[d, e],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'n_edge':
           tf_helpers.Int64Feature(
                         key='n_edge',
                         dtype='int32',
                         description='Number of edges in this graph'),
      'globals':
           tf_helpers.VarLenFloatFeature(
                         key='globals',
                         shape=[None],
                         description='Edge features'),
      'edges':
           tf_helpers.VarLenFloatFeature(
                         key='edges',
                         shape=[None, 1],
                         description='Edge features'),
      'receivers':
           tf_helpers.VarLenIntListFeature(
                         key='receivers',
                         dtype='int32',
                         description='Recieving nodes for edges'),
      'senders':
           tf_helpers.VarLenIntListFeature(
                         key='senders',
                         dtype='int32',
                         description='Sending nodes for edges'),
    }

  def get_placeholders(self):
    # TODO: Make this work with utils_tf for graph_nets for GraphTuple
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def gen_sample(self):
    print("Error: Unable to generate sample - Not Implemented")
    sys.exit(1)

  def process_features(self, loaded_features):
    features = {}
    for k, feat in self.features.items():
      features.update(feat.get_feature_write(loaded_features[k]))
    return features

  ######### Generic methods ##########
  # Hopefully after this point you won't have to subclass any of these
  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    other_keys = list(set(self.features.keys()) - set(GRAPH_KEYS))
    item_keys = GRAPH_KEYS + other_keys
    data_source_name = mode + '-*.tfrecords'
    data_sources = glob.glob(os.path.join(self.data_dir, mode, data_source_name))
    if opts.shuffle_data and mode != 'test':
      np.random.shuffle(data_sources) # Added to help the shuffle
    # Build dataset provider
    keys_to_features = {}
    for k, v in self.features.items():
      keys_to_features.update(v.get_feature_read()) 
    items_to_descriptions = { k: v.description
                              for k, v in self.features.items() }
    def parser_op(record):
      example = tf.parse_single_example(record, keys_to_features)
      return [ self.features[k].tensors_to_item(example)
               for k in item_keys ]
    dataset = tf.data.TFRecordDataset(data_sources)
    dataset = dataset.map(parser_op)
    dataset = dataset.repeat(None)
    if opts.shuffle_data and mode != 'test':
      dataset = dataset.shuffle(buffer_size=5*opts.batch_size)
    # dataset = dataset.prefetch(buffer_size=opts.batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_graphs = []
    batch_other = []
    for b in range(opts.batch_size):
      sample_ = iterator.get_next()
      # Extracting other keys outside of the graph
      sample_other_ = { k : sample_[i + len(GRAPH_KEYS)]
                        for i, k in enumerate(other_keys) }
      batch_other.append(sample_other_)
      # Constructing graph using relevant graph keys
      sample_graph = { k : sample_[i] for i, k in enumerate(GRAPH_KEYS) }
      batch_graphs.append(sample_graph)
    # Constructing output sample using known order of the keys
    sample = {}
    for k in other_keys:
      sample[k] = self.features[k].stack([
          batch_other[b][k] for b in range(opts.batch_size)
      ])
    sample['graph'] = utils_tf.data_dicts_to_graphs_tuple(batch_graphs)
    return sample

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    params = self.dataset_params
    fname = '{}-{:03d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(params.sizes[mode])):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.gen_sample()
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def create_np_dataset(self, out_dir, num_entries):
    """Create npz files to store dataset"""
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample()
      np.savez(outfile(index), **features)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('Numpy Dataset created {}'.format(str(datetime.datetime.now())))



