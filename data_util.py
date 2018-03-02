# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os
import sys
import glob
import tqdm
from enum import Enum

import torch
import torch.utils.data as tdata

import options
import nputils
import sim_graphs

class GraphDataset(tdata.Dataset):
  """Dataset for Cycle Consistency graphs"""
  def __init__(self, root_dir):
    self._root_dir = root_dir
    self.keys = [
      'AdjMat',
      'Degrees'
      'TrueEmbedding',
    ]

  def __len__(self):
    return len(glob.glob(os.path.join(self._root_dir,'*.npz')))

  def __getitem__(self, idx):
    fname = os.path.join(self._root_dir,'{:09d}.npz'.format(idx))
    ld = np.load(fname)
    sample = dict(zip(self.keys, [torch.from_numpy(ld[k]) for k in self.keys]))
    # TODO: Add augmentation here??
    return sample

  def generate_graph(self, n_points, n_pts, opts):
    pass

  def generate_data(self, sz):
    for i in tqdm.tqdm(range(sz)):
      data = generate_graph(opts)
      name = '{:09d}.npz'.format(i)
      np.savez(os.path.join(self._root_dir,name), **data)

class GraphSimDataset(tdata.Dataset):
  """Dataset for Cycle Consistency graphs"""
  def __init__(self, opts, gen_type, dataset_len, n_pts=None, n_poses=None):
    self.opts = opts
    self._dataset_len = dataset_len
    self.n_pts = n_pts
    self.n_poses = n_poses
    self.keys = [
      'AdjMat',
      'Degrees'
      'TrueEmbedding',
    ]

  def get_sample(self):
    pose_graph = sim_graphs.PoseGraph(self.opts, self.n_pts, self.n_poses)
    if not self.opts.soft_edges:
      if self.opts.descriptor_noise_var == 0:
        perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
                   for i in range(pose_graph.n_poses) ]
        TrueEmbedding = np.concatenate(perms_, 0)
        AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
        print(self.opts.sparse)
        if self.opts.sparse:
          sz = (pose_graph.n_pts, pose_graph.n_pts)
          mask = np.kron(pose_graph.adj_mat,np.ones(sz))
          AdjMat = AdjMat * mask
        else:
          AdjMat = AdjMat - np.eye(len(AdjMat))
        Degrees = np.sum(AdjMat,0)
    else:
      if self.opts.sparse and self.opts.descriptor_noise_var > 0:
        AdjMat = pose_graph.get_feature_matching_mat()
        Degrees = np.sum(AdjMat,0)
        TrueEmbedding = 0

    return {
      'AdjMat': AdjMat,
      'Degrees': Degrees,
      'TrueEmbedding': TrueEmbedding
    }
    
  def __len__(self):
    return self._dataset_len

  def __getitem__(self, idx):
    return self.get_sample()



if __name__ == '__main__':
  opts = options.get_opts()
  dataset = GraphSimDataset(opts, 40, 20)
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  sample = dataset.get_sample()
  plt.imshow(sample['AdjMat'])
  plt.show()

  sys.exit()
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




