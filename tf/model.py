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

import tensorflow as tf
import sonnet as snt

import nputils
import options

# TODO: Better documentation
class DenseGraphLayer(snt.AbstractModule):
  """Implements graph layer from GCN, using dense matrices."""
  def __init__(self,  alternate_laplacian, output_size, activation=tf.nn.relu,
               name="dense_graph_layer"):
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
    self._linear = snt.Linear(self._out)
    return self._activ(tf.matmul(self._lapl, self._linear(inputs)))


if __name__ == "__main__":
  import gen_data
  opts = options.get_opts()
  nsteps = 10
  nout = 15
  alternate_laplacian = tf.placeholder(tf.float32, [None, None])
  x = tf.placeholder(tf.float32, [None, opts.descriptor_dim])
  dgl = DenseGraphLayer(alternate_laplacian, nout)
  dgl2 = DenseGraphLayer(alternate_laplacian, nout)
  out0 = dgl(x)
  print(out0)
  out = dgl2(out0)
  print(out)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(nsteps):
      data = gen_data.generate_graph(opts)
      A = data['graph'] + np.eye(data['graph'].shape[0])
      D_h_inv = np.diag(1./np.sqrt(np.sum(A,1)))
      alt_lap = np.dot(D_h_inv, np.dot(A, D_h_inv))
      out_ = sess.run(out, {alternate_laplacian:alt_lap, x:data['embeddings']})
      print("{}: {}".format(i, out_.shape))
    

  


