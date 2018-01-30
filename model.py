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


if __name__ == "__main__":
  import gen_data
  opts = options.get_opts()
  nsteps = 10
  nout = 15
  # Test
  # Build network
  linear0 = torch.nn.Linear(opts.descriptor_dim, nout)
  activ0 = torch.nn.ReLU
  linear1 = torch.nn.Linear(nout, nout)
  activ1 = torch.nn.ReLU
  # Get data
  i = 0
  data = gen_data.generate_graph(opts)
  A = data['graph'] + np.eye(data['graph'].shape[0])
  D_h_inv = np.diag(1./np.sqrt(np.sum(A,1)))
  alt_lap = torch.autograd.Variable(np.dot(D_h_inv, np.dot(A, D_h_inv)))
  x = torch.autograd.Variable(data['embeddings'])
  # Run network
  out0 = activ0(torch.mm(alt_lap, linear0(x)))
  out1 = activ1(torch.mm(alt_lap, linear1(out0)))

  print("{}: {}".format(i, 0))



