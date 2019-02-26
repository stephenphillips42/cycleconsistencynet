import os
import sys
import collections
import itertools
import glob
import time
import pickle
import tqdm

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

import tensorflow as tf
import sonnet as snt
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules

import options
from data_util import synth_graphs

import pprint
pp = pprint.PrettyPrinter(indent=2)
# import pdb; pdb.set_trace()

opts = options.get_opts()
dtype = opts.dataset_params.dtype
LD_FILE_SIZE = 128
DATASET_SIZE = 2 * LD_FILE_SIZE
BATCH_SIZE = 16
size_keys = [ 'n_node', 'n_edge' ]
size_types = [ tf.int64, tf.int64 ]
graph_keys = [ 'globals', 'nodes', 'edges', 'receivers', 'senders' ]
graph_types = [ tf.float32, tf.float32, tf.float32, tf.int64, tf.int64 ]
all_keys = size_keys + graph_keys
all_types = size_types + graph_types
SEED = 1432

def generate_graph(opts):
  params = opts.dataset_params
  p = params.points[-1]
  v = params.views[-1]
  r = 3 #params.num_repeats
  noise = 0.1 # dataset_params.noise_level
  def perm(p):
    return np.eye(p)[np.random.permutation(p),:]
  
  pose_graph = synth_graphs.PoseGraph(params, n_pts=p, n_views=v)
  sz = (pose_graph.n_pts, pose_graph.n_pts)
  sz2 = (pose_graph.n_views, pose_graph.n_views)
  mask = np.kron(np.ones(sz2)-np.eye(sz2[0]),np.ones(sz))
  
  perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
             for i in range(pose_graph.n_views) ]
  # Embedding objects
  TrueEmbedding = np.concatenate(perms_, 0)
  TEmb = TrueEmbedding
  InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                    for i in range(pose_graph.n_views) ], 0)
  
  # Graph objects
  AdjMat = np.zeros((p*v,p*v))
  for i in range(v):
    TEmb_i = TEmb[p*i:p*i+p,:]
    for j in range(i+1, v):
      TEmb_j = TEmb[p*j:p*j+p,:]
      Noise = (1-noise)*np.eye(p) + noise*sum([ perm(p) for i in range(r) ])
      Val_ij = np.dot(TEmb_i, np.dot(Noise, TEmb_j.T))
      AdjMat[p*i:p*i+p, p*j:p*j+p] = Val_ij
      AdjMat[p*j:p*j+p, p*i:p*i+p] = Val_ij.T
  AdjMat = np.minimum(1, AdjMat)
  Degrees = np.diag(np.sum(AdjMat,0))
  
  # Laplacian objects
  Ahat = AdjMat + np.eye(*AdjMat.shape)
  Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
  Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
  
  G = nx.from_numpy_matrix(AdjMat, create_using=nx.DiGraph)
  node_attrs = { i : InitEmbeddings[i].astype(np.float32)
                 for i in range(len(G)) }
  edges_attrs = { (i, j) : np.array([ AdjMat[i,j] ]).astype(np.float32)
                  for (i,j) in G.edges }
  nx.set_node_attributes(G, node_attrs, 'features')
  nx.set_edge_attributes(G, edges_attrs, 'features')
  # return G
  G_ = utils_np.networkx_to_data_dict(G)
  G_['globals'] = np.array([0])
  return G_

def generate_dataset(opts):
  fname = os.path.join(opts.save_dir, 'g{:05d}.pickle')
  for ld_file in tqdm.tqdm(range(DATASET_SIZE // LD_FILE_SIZE)):
    graph_list = []
    for i in range(LD_FILE_SIZE):
      G = generate_graph(opts)
      graph_list.append(G)
    with open(fname.format(ld_file), 'wb') as f:
      pickle.dump(graph_list, f)
    pass

def _dict_to_tuple(d):
  return ( d[k] for k in all_keys )

def _tuple_to_dict(t):
  return dict(zip(all_keys, t))

def _concat(x):
  if len(x) < 1:
    z = None
  elif type(x[0]) == None:
    z = np.array([0])
  elif type(x[0]) == int:
    z = np.array(x)
  elif type(x[0]) == np.ndarray:
    z = np.concatenate(x,0)
  else:
    z = None
  return z

# Fix randomness
np.random.seed(SEED)
tf.set_random_seed(SEED)

# # Dataset saving
generate_dataset(opts)
fnames = glob.glob(os.path.join(opts.save_dir, '*.pickle'))
print(fnames)

# Dataset loading
dataset_range = range(len(fnames)*(LD_FILE_SIZE // BATCH_SIZE))
def get_data(index, batch_size=BATCH_SIZE):
  filename = fnames[index // (LD_FILE_SIZE // batch_size)]
  with open(filename, 'rb') as f:
    pickle_decoded = pickle.load(f)
  graphs = np.random.choice(pickle_decoded, size=batch_size)
  # graphs_ = [ pickle_decoded[i] for i in range(batch_size) ]
  return graphs

print("Starting things...")
eg_data = get_data(0)
# input_graphs = utils_tf.placeholders_from_networkxs([eg_data])
print("input_graphs")
input_graphs = utils_tf.placeholders_from_data_dicts(eg_data)
print(input_graphs)


class MLPInteractiveNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super().__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.InteractionNetwork(
                        edge_model_fn=lambda: snt.nets.MLP([30,20]),
                        node_model_fn=lambda: snt.nets.MLP([50,25]))

  def _build(self, inputs):
    return self._network(inputs)

print("network")
network = MLPInteractiveNetwork()
print(network)


print("output")
output = network(input_graphs)
print(output)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  pass
  init=tf.global_variables_initializer()
  sess.run(init)
  for i in dataset_range:
    g = get_data(i)
    g_input = utils_np.data_dicts_to_graphs_tuple(g)
    print("g_input")
    print(g_input)
    feed_dict = utils_tf.get_feed_dict(input_graphs, g_input)
    print("feed_dict")
    pp.pprint([ (k,v) for k, v in feed_dict.items() ])
    pp.pprint([ (key.name, key.shape, type(value)) for key, value in feed_dict.items() ])
    # import pdb; pdb.set_trace()
    o = sess.run(output, feed_dict=feed_dict)
    print("o")
    print(o)
    print("I FINALLY GOT THE OUTPUT :D")







