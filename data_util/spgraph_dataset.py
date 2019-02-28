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
from data_util import synth_graphs
from data_util import tf_helpers

def denseNDArrayToSparseTensor(arr):
  idx  = np.where(arr != 0.0)
  return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)

class SpSynthGraphDataset(mydataset.MyDataset):
  MAX_IDX=700
  def __init__(self, opts, params):
    super().__init__(opts, params)
    d = self.n_pts*self.n_views
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
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'n_edge':
           tf_helpers.Int64Feature(
                         key='n_edge',
                         dtype='int32',
                         description='Number of edges in this graph'),
      'edges': # TODO: How to do this?
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
    #   'adjmat':
    #        tf_helpers.SparseTensorFeature(
    #                      key='adjmat',
    #                      shape=[d, d],
    #                      description='Sparse adjacency matrix of graph'),
    self.graph_keys = [
      'n_node', 'nodes', 'n_edge', 'edges', 'receivers', 'senders'
    ]
    # self.other_keys = [ 'adjmat' ]
    self.other_keys = [ ]
    self.item_keys = self.graph_keys + self.other_keys

  def get_placeholders(self):
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def gen_init_emb_noise(self, init_emb):
    return init_emb

  def gen_adjmat_noise(self, true_emb):
    adj_mat = np.dot(true_emb,true_emb.T) - np.eye(true_emb.shape[0])
    return adj_mat

  def gen_sample(self):
    # Pose graph and related objects
    params = self.dataset_params
    pose_graph = synth_graphs.PoseGraph(params,
                                        n_pts=self.n_pts,
                                        n_views=self.n_views)
    # Embedding objects
    perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
               for i in range(pose_graph.n_views) ]
    TrueEmbedding = np.concatenate(perms_, 0)
    InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                      for i in range(pose_graph.n_views) ], 0)
    # Graph objects
    AdjMat = self.gen_adjmat_noise(TrueEmbedding)
    InitEmbeddings = self.gen_init_emb_noise(InitEmbeddings)
    # Build spart graph representation
    G_nx = nx.from_numpy_matrix(AdjMat, create_using=nx.DiGraph)
    node_attrs = { i : InitEmbeddings[i].astype(np.float32)
                   for i in range(len(G_nx)) }
    edges_attrs = { (i, j) : np.array([ AdjMat[i,j] ]).astype(np.float32)
                    for (i,j) in G_nx.edges }
    nx.set_node_attributes(G_nx, node_attrs, 'features')
    nx.set_edge_attributes(G_nx, edges_attrs, 'features')
    G = utils_np.networkx_to_data_dict(G_nx)
    G['globals'] = np.array([0])
    idx = np.where(AdjMat != 0.0)
    value = AdjMat[idx]
    G['adjmat'] = (idx, value)
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
    if opts.batch_size > 1:
      item_sizes = [ self.features[k].shape for k in self.item_keys ]
      # dataset = dataset.padded_batch(opts.batch_size, padded_shapes=item_sizes)
      dataset = dataset.batch(opts.batch_size)
      dataset = dataset.prefetch(buffer_size=opts.batch_size)

    iterator = dataset.make_one_shot_iterator()
    sample_ = iterator.get_next()
    # Constructing graph using relevant graph keys
    sample_graph_ = { k : sample_[i] for i, k in enumerate(self.graph_keys) }
    print(sample_graph_)
    import pdb; pdb.set_trace()
    sample_graph = utils_tf.data_dicts_to_graphs_tuple([ sample_graph_ ])
    print("OK")
    graph = utils_tf.concat(sample_graph, 0)
    # Constructing output sample using known order of the keys
    sample = { k : sample_[i + len(self.graph_keys)]
               for i, k in enumerate(self.other_keys) }
    sample['graph'] = graph
    return graph


