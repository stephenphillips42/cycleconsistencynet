# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os
import tqdm

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
    'n_pts': n_pts,
    'n_poses': n_poses,
    'graph': graph.astype(opts.np_type),
    'embeddings': descs.astype(opts.np_type),
    'graph_true': graph_true.astype(opts.np_type),
    'perms': perms_.astype(opts.np_type) 
  }

# TODO: 
# def generate_constant_graph(opts):

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
    if not os.path.exists(os.path.join(opts.data_dir,t)):
      os.makedirs(os.path.join(opts.data_dir, t))
    for i in tqdm.tqdm(range(sz)):
      data = generate_graph(opts)
      name = "{:09d}.npz".format(i)
      np.savez(os.path.join(opts.data_dir,t,name), **data)




