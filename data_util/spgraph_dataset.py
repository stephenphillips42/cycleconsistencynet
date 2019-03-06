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

class SpSynthGraphDataset(mydataset.MyDataset):
  MAX_IDX=700
  def __init__(self, opts, params):
    super().__init__(opts, params)
    p = self.n_pts
    v = self.n_views
    d = p*v
    e = params.descriptor_dim
    """
    GraphsTuple(nodes=nodes,
              edges=edges,
              globals=globals,
              receivers=receivers,
              senders=senders,
              n_node=n_node,
              n_edge=n_edge)
    """
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
      'adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='adj_mat',
                         shape=[d, d],
                         description='Sparse adjacency matrix of graph'),
      'true_adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='true_adj_mat',
                         shape=[d, d],
                         description='Sparse ground truth adjacency of graph'),
      'true_match':
           tf_helpers.VarLenIntListFeature(
                         key='true_match',
                         dtype='int64',
                         description='Ground truth matches of graph'),
    }
    self.graph_keys = [
      'n_node', 'nodes', 'n_edge', 'edges', 'receivers', 'senders'
    ]
    # self.other_keys = [ ]
    self.other_keys = list(set(self.features.keys()) - set(self.graph_keys))
    self.item_keys = self.graph_keys + self.other_keys

  def get_placeholders(self):
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def gen_init_emb(self, matches):
    params = self.dataset_params
    desc_var = params.descriptor_var
    var = params.descriptor_noise_var
    desc_dim = params.descriptor_dim
    point_descs = desc_var*np.random.randn(self.n_pts, desc_dim)
    desc_noise = var*np.random.randn(len(matches), desc_dim)
    return point_descs[matches] + desc_noise

  def gen_adj_mat_noise(self, true_emb):
    adj_mat = np.dot(true_emb,true_emb.T) - np.eye(true_emb.shape[0])
    # if np.random.randn() > 0:
    #   d = self.n_pts*self.n_views
    #   adj_mat[]
    return adj_mat

  def gen_sample(self):
    # Pose graph and related objects
    params = self.dataset_params
    # Embedding objects
    matches_ = np.concatenate([ np.random.permutation(self.n_pts)
                                for i in range(self.n_views) ])
    TrueEmbedding = np.eye(self.n_pts)[matches_]
    InitEmbeddings = self.gen_init_emb(matches_)
    # Graph objects
    GTAdjMat = np.dot(TrueEmbedding, TrueEmbedding.T)
    AdjMat = self.gen_adj_mat_noise(TrueEmbedding)
    # Build spart graph representation
    G_nx = nx.from_numpy_matrix(AdjMat, create_using=nx.DiGraph)
    node_attrs = { i : InitEmbeddings[i].astype(np.float32)
                   for i in range(len(G_nx)) }
    edges_attrs = { (i, j) : np.array([ AdjMat[i,j] ]).astype(np.float32)
                    for (i,j) in G_nx.edges }
    nx.set_node_attributes(G_nx, node_attrs, 'features')
    nx.set_edge_attributes(G_nx, edges_attrs, 'features')
    G = utils_np.networkx_to_data_dict(G_nx)
    G['globals'] = np.array([0,0])
    G['adj_mat'] = np_dense_to_sparse(AdjMat)
    G['true_adj_mat'] = np_dense_to_sparse(GTAdjMat)
    G['true_match'] = matches_
    return G

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
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
               for k in self.item_keys ]
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
      sample_other_ = { k : sample_[i + len(self.graph_keys)]
                        for i, k in enumerate(self.other_keys) }
      batch_other.append(sample_other_)
      # Constructing graph using relevant graph keys
      sample_graph = { k : sample_[i] for i, k in enumerate(self.graph_keys) }
      sample_graph['globals'] = tf.zeros([2])
      batch_graphs.append(sample_graph)
    # Constructing output sample using known order of the keys
    sample = {}
    for k in self.other_keys:
      sample[k] = self.features[k].stack([
          batch_other[b][k] for b in range(opts.batch_size)
      ])
    sample['graph'] = utils_tf.data_dicts_to_graphs_tuple(batch_graphs)
    return sample


