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

from data_util import graph_dataset
from data_util import tf_helpers

class SynthGraphDataset(graph_dataset.GraphDataset):
  MAX_IDX=700
  def __init__(self, opts, params):
    super().__init__(opts, params)
    p = self.n_pts
    v = self.n_views
    d = p*v
    self.features.update({
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
    })

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
    G['adj_mat'] = graph_dataset.np_dense_to_sparse(AdjMat)
    G['true_adj_mat'] = graph_dataset.np_dense_to_sparse(GTAdjMat)
    G['true_match'] = matches_
    return G


class SynthNoiseGraphDataset(SynthGraphDataset):
  def __init__(self, opts, params):
    super().__init__(opts, params)

  def gen_adj_mat_noise(self, true_emb):
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    noise_lvl = self.dataset_params.noise_level
    subnoise_lvl = self.dataset_params.subnoise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    adjmat = np.zeros((p*n,p*n))
    # Create the noise for each image pair
    for i in range(n):
      true_emb_i = true_emb[p*i:p*i+p,:]
      for j in range(i+1, n):
        true_emb_j = true_emb[p*j:p*j+p,:]
        gt = (1-noise_lvl)*np.eye(p)
        # Noise adds in other soft matches randomly
        noise = noise_lvl*sum([ perm(p) for i in range(r) ])
        val_ij = np.dot(true_emb_i, np.dot(noise + gt, true_emb_j.T))
        adjmat[p*i:p*i+p, p*j:p*j+p] = val_ij
        adjmat[p*j:p*j+p, p*i:p*i+p] = val_ij.T
    adjmat = adjmat*(1 + subnoise_lvl*np.random.randn(*adjmat.shape))
    adj_mat = np.minimum(1, adjmat)
    return adj_mat

class SynthOutlierGraphDataset(SynthNoiseGraphDataset):
  def __init__(self, opts, params):
    super().__init__(opts, params)

  def create_outlier_indeces(self, o, n):
    ind_pairs = [ (x,y) for x in range(n) for y in range(x+1,n) ]
    probs = [ 1.0/len(ind_pairs) ] * len(ind_pairs)
    outlier_ind_pairs = np.random.multinomial(o, probs, size=1)[0]
    outlier_sel = np.zeros((n,n), dtype=np.int64)
    for i in range(len(outlier_ind_pairs)):
      # outlier_sel[ind_pairs[i]] = int(outlier_ind_pairs[i])
      outlier_sel[ind_pairs[i]] = (outlier_ind_pairs[i])
    return outlier_sel

  def gen_adj_mat_noise(self, true_emb):
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    o = self.dataset_params.num_outliers
    o2 = self.dataset_params.num_outliers_each
    noise_lvl = self.dataset_params.noise_level
    subnoise_lvl = self.dataset_params.subnoise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    adjmat = np.zeros((p*n,p*n))
    # Create the noise and outliers for each image pair
    outlier_sel =  self.create_outlier_indeces(o, n)
    for i in range(n):
      true_emb_i = true_emb[p*i:p*i+p,:]
      for j in range(i+1, n):
        true_emb_j = true_emb[p*j:p*j+p,:]
        gt = (1-noise_lvl)*np.eye(p)
        # If we select this for outliers...
        if outlier_sel[i,j] > 0:
          gt = np.eye(p)
          # Swap rows - create intentional outliers
          for _ in range(o2):
            s0, s1 = np.random.choice(range(p), size=2, replace=False)
            tmp = gt[s1,:].copy()
            gt[s1,:] = gt[s0,:]
            gt[s0,:] = tmp
          gt = (1-noise_lvl)*gt
        # Noise adds in other soft matches randomly
        noise = noise_lvl*sum([ perm(p) for i in range(r) ])
        val_ij = np.dot(true_emb_i, np.dot(noise + gt, true_emb_j.T))
        adjmat[p*i:p*i+p, p*j:p*j+p] = val_ij
        adjmat[p*j:p*j+p, p*i:p*i+p] = val_ij.T
    adjmat = adjmat*(1 + subnoise_lvl*np.random.randn(*adjmat.shape))
    adj_mat = np.minimum(1, adjmat)
    return adj_mat


