# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os
import glob
import tqdm

import torch
import torch.utils.data as tdata

import options
import nputils
import sim_graphs

def generate_graph(opts):
  pose_graph = sim_graphs.PoseGraph(opts)
  n_pts = pose_graph.n_pts
  n_poses = pose_graph.n_poses
  perms = [ np.eye(n_pts)[:,pose_graph.get_perm(i)]
            for i in range(n_poses) ]
  descs = np.concatenate([ pose_graph.get_proj(i).d
                           for i in range(n_poses) ], 0)
  perms_ = np.concatenate(perms, 0)
  graph = pose_graph.get_feature_matching_mat()
  aug_adj = np.eye(n_poses) + pose_graph.adj_mat
  mask = np.kron(aug_adj, np.ones((n_pts, n_pts)))
  graph_true = mask*np.dot(perms_,perms_.T)
  return { 
    'n_pts': np.array([n_pts]),
    'n_poses': np.array([n_poses]),
    'graph': graph.astype(opts.np_type),
    'embeddings': descs.astype(opts.np_type),
    'graph_true': graph_true.astype(opts.np_type),
    'perms': perms_.astype(opts.np_type) 
  }

def build_alt_lap(opts, data):
  Adj = data['graph']
  Adj_alt = Adj + np.eye(Adj.shape[0]).astype(opts.np_type)
  D_h_inv = np.diag(1./np.sqrt(np.sum(Adj_alt,1)))
  alt_lap_np = np.dot(D_h_inv, np.dot(Adj_alt, D_h_inv))
  return alt_lap_np

def build_weight_mask(opts, data):
  Adj = data['graph']
  n_pts = data['n_pts'][0]
  n_poses = data['n_poses'][0]
  D_sqrt_inv = np.diag(1./np.sqrt(np.sum(Adj,1)))
  nm_ = np.eye(n_pts) - np.ones((n_pts,n_pts))
  neg_mask_np = (np.kron(np.eye(n_poses),nm_)).astype(opts.np_type)
  return neg_mask_np + Adj

def build_weight_offset(opts, data):
  n_pts = data['n_pts'][0]
  n_poses = data['n_poses'][0]
  nm_ = np.ones((n_pts,n_pts)) - np.eye(n_pts) 
  neg_mask_np = (np.kron(np.eye(n_poses),nm_)).astype(opts.np_type)
  return neg_mask_np

# TODO: 
# def generate_constant_graph(opts):

class CycleConsitencyGraphDataset(tdata.Dataset):
  """Dataset for Cycle Consistency graphs"""
  def __init__(self, root_dir):
    self._root_dir = root_dir

  def __len__(self):
    return len(glob.glob(os.path.join(self._root_dir,'*.npz')))

  def __getitem__(self, idx):
    fname = os.path.join(self._root_dir,'{:09d}.npz'.format(idx))
    ld = np.load(fname)
    sample = dict(zip(ld.keys(), [torch.from_numpy(ld[k]) for k in ld.keys()]))
    # TODO: Add augmentation here??
    return sample

  def generate_data(self, sz):
    for i in tqdm.tqdm(range(sz)):
      data = generate_graph(opts)
      data.update({
        'alt_lap': build_alt_lap(opts,data),
        'weight_mask': build_weight_mask(opts,data),
        'weight_offset': build_weight_offset(opts,data),
      })
      # Save out
      name = '{:09d}.npz'.format(i)
      np.savez(os.path.join(self._root_dir,name), **data)

if __name__ == '__main__':
  opts = options.get_opts()
  print("Generating Pose Graphs")
  if not os.path.exists(opts.data_dir):
    os.makedirs(opts.data_dir)

  sizes = {
    'train' : opts.num_gen_train,
    'test' : opts.num_gen_test
  }
  for t, sz in sizes.iteritems():
    dname = os.path.join(opts.data_dir,t)
    if not os.path.exists(dname):
      os.makedirs(dname)
    dataset = CycleConsitencyGraphDataset(dname)
    dataset.generate_data(sz)




