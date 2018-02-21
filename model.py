# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
from tqdm import tqdm

import torch
from torch.autograd import Variable

import nputils
import options

# TODO: Better documentation
class DenseGraphLayer(torch.nn.Module):
  """Implements graph layer from GCN, using dense matrices."""
  def __init__(self,  alternate_laplacian, input_size, output_size,
               activation=torch.nn.ReLU, name="dense_graph_layer"):
    """Build graph layer and specify input and output size.
    
    Args:
      alternate_laplacian: Laplcian variant used for propogating nodes in
        the graph
      output_size: Number of outputs from the layer
      activation: Pointwise non-linearity of the layer (default: tf.nn.relu)
    """
    super(DenseGraphLayer, self).__init__(name=name)
    self._out = output_size
    self._lapl = alternate_laplacian
    self._activ = activation

  def _build(self, inputs):
    """Compute next layer of embeddings."""
    print(inputs)
    self._linear = torch.nn.Linear(self._out)
    return self._activ(torch.mm(self._lapl, self._linear(inputs)))

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def build_alt_lap(opts, data):
  Adj = data['graph']
  Adj_alt = Adj + np.eye(Adj.shape[0]).astype(opts.np_type)
  D_h_inv = np.diag(1./np.sqrt(np.sum(Adj_alt,1)))
  alt_lap_np = np.dot(D_h_inv, np.dot(Adj_alt, D_h_inv))
  return Variable(torch.from_numpy(alt_lap_np))

def build_weight_mask(opts, data):
  Adj = data['graph']
  n_pts = data['n_pts']
  n_poses = data['n_poses']
  D_sqrt_inv = np.diag(1./np.sqrt(np.sum(Adj,1)))
  nm_ = np.eye(n_pts) - np.ones((n_pts,n_pts))
  neg_mask_np = (np.kron(np.eye(n_poses),nm_)).astype(opts.np_type)
  return Variable(torch.from_numpy(neg_mask_np + Adj))

if __name__ == "__main__":
  import data_util
  opts = options.get_opts()
  nsteps = 10
  nout = 15
  # Test
  # Build network
  linear0 = torch.nn.Linear(opts.descriptor_dim, nout)
  activ0 = torch.nn.ReLU()
  linear1 = torch.nn.Linear(nout, nout)
  activ1 = torch.nn.ReLU()
  # Get data
  i = 0
  data = data_util.generate_graph(opts)
  alt_lap = build_alt_lap(opts, data)
  x = Variable(torch.from_numpy(data['embeddings']))
  # Run network
  out0 = activ0(torch.mm(alt_lap, linear0(x)))
  print(out0)
  out1 = activ1(torch.mm(alt_lap, linear1(out0)))
  print(out1)
  print("Computing dists")
  dists = pairwise_distances(out1)
  weight_mask = build_weight_mask(opts, data)
  weighted_dists = dists*weight_mask
  print("Done")
  print(dists.size())
  total_weight = torch.sum(weighted_dists)
  print(total_weight)
  total_weight.backward()

  print("{}: {}".format(i, out1.size()))
  print("{}: {}".format(i, dists.size()))
  print("{}: {}".format(i, dists.size()))







