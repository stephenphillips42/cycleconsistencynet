# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys

try:
  import tensorflow as tf
  import tfutils
except:
  print("ERROR: Tensorflow failed to load")
  tf = None
  tfutils = None

import myutils
import options

class DenseGraphLayerWeights(object):
  def __init__(self, opts, arch):
    super(DenseGraphLayerWeights, self).__init__()
    self.tf_init = False
    self.np_init = False
    self.activ = arch.activ
    self._activ = None # tfutils.get_tf_activ(arch.activ)
    self._np_activ = None # myutils.get_np_activ(arch.activ)
    self._nlayers = arch.nlayers
    self._layer_lens = \
        [opts.descriptor_dim] + arch.layer_lens + [opts.final_embedding_dim]
    self._np_layers = []
    self._layers = []

  def build_tf_layers(self):
    """Build layers"""
    self._activ = tfutils.get_tf_activ(arch.activ)
    with tf.variable_scope("gnn_weights"):
      for i in range(len(self._layer_lens)-1):
        layer = tf.get_variable("weight_{:02d}".format(i),
                                [ self._layer_lens[i], self._layer_lens[i+1] ],
                                initializer=tf.random_normal_initializer())
        self._layers.append(layer)
    self.tf_init = True

  def apply(self, sample):
    """Applying this graph network to sample"""
    if not self.tf_init:
      self.build_tf_layers()
    lap = sample['Laplacian']
    init_emb = sample['InitEmbeddings']
    output = init_emb
    for l in range(self._nlayers):
      lin = tfutils.matmul(output, self._layers[l])
      lin_graph = tfutils.batch_matmul(lap, lin)
      output = self._activ(lin_graph)
    output = tfutils.matmul(output, self._layers[-1])
    output = tf.nn.l2_normalize(output, axis=2)
    return output

  def save_np(self, saver, save_dir):
    checkpoint_file = tf.train.latest_checkpoint(save_dir)
    with tf.Session() as np_sess:
      saver.restore(np_sess, checkpoint_file)
      for i in range(len(self._layers)):
        self._np_layers.append(np_sess.run(self._layers[i]))
    outdir = myutils.next_dir(os.path.join(save_dir, 'np_weights'))
    np.savez(os.path.join(outdir, 'numpy_weights.npz'), *self._np_layers)

  def load_np(self, save_dir):
    numpy_weights = np.load(os.path.join(save_dir, 'numpy_weights.npz'))
    self._np_layers = [ numpy_weights['arr_{}'.format(i)]
                        for i in range(len(numpy_weights.files)) ]
    self._np_activ = myutils.get_np_activ(self.activ)
    self.np_init = True

  def apply_np(self, sample):
    """Applying this graph network to sample, using numpy input.
    Only takes in one input at a time."""
    lap = sample['Laplacian']
    init_emb = sample['InitEmbeddings']
    output = init_emb
    for l in range(self._nlayers):
      lin = np.dot(output, self._np_layers[l])
      lin_graph = np.dot(lap, lin)
      output = self._np_activ(lin_graph)
    output = np.dot(output, self._np_layers[-1])
    output = myutils.dim_normalize(output)
    return output

def get_network(opts, arch):
  network = DenseGraphLayerWeights(opts, arch)
  return network

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







