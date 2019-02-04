import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

import options
from data_util import synth_graphs

# Fix randomness
SEED = 1432
np.random.seed(SEED)
tf.set_random_seed(SEED)

opts = options.get_opts()
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

# # Graph objects
# AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
# AdjMat = AdjMat - np.eye(len(AdjMat))
# Degrees = np.diag(np.sum(AdjMat,0))
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

# # Mask objects
# neg_offset = np.kron(np.eye(sz2[0]),np.ones(sz)-np.eye(sz[0]))
# Mask = AdjMat - neg_offset
# MaskOffset = neg_offset
# dtype = 'float32'
# sample = {
#   'InitEmbeddings': InitEmbeddings.astype(dtype),
#   'AdjMat': AdjMat.astype(dtype),
#   'Degrees': Degrees.astype(dtype),
#   'Laplacian': Laplacian.astype(dtype),
#   'Mask': Mask.astype(dtype),
#   'MaskOffset': MaskOffset.astype(dtype),
#   'TrueEmbedding': TrueEmbedding.astype(dtype),
#   'NumViews': pose_graph.n_views,
#   'NumPoints': pose_graph.n_pts,
# }

G = nx.from_numpy_matrix(AdjMat)
print(G.nodes)
print(G.edges)
node_attrs = { i : InitEmbeddings[i] for i in range(len(G)) }
edges_attrs = { (i, j) : AdjMat[i,j] for (i,j) in G.edges }
nx.set_node_attributes(G, node_attrs, 'embeddings')
nx.set_edge_attributes(G, edges_attrs, 'weight')
print(G[0])




