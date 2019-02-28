import numpy as np
import scipy.linalg as la
import os
import sys
import glob
import datetime
import tqdm
import pickle

import tensorflow as tf

from data_util import mydataset
from data_util import real_dataset
from data_util import tf_helpers
from data_util.rome16k import parse

class SpGeomKNNRome16KDataset(Rome16KTupleDataset):
  def __init__(self, opts, params):
    super(GeomKNNRome16KDataset, self).__init__(opts, params, tuple_size=params.views[-1])
    d = self.n_pts*self.n_views
    e = params.descriptor_dim
    """
    GraphsTuple(nodes=nodes,
              edges=edges,
              globals=globals,
              receivers=receivers,
              senders=senders,
              n_node=n_node,
              n_edge=n_edge)
    """
    self.features.update({
      'nodes':
           tf_helpers.TensorFeature(
                         key='nodes',
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      # TODO: How to do this?
      'edges':
           tf_helpers.TensorFeature(
                         key='edges',
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Edge features'),
      'recievers':
           tf_helpers.TensorFeature(
                         key='recievers',
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Recieving nodes for edges'),
      'senders':
           tf_helpers.TensorFeature(
                         key='senders',
                         shape=[d, e + 2 + 1 + 1],
                         dtype=self.dtype,
                         description='Sending nodes for edges'),
      'Rotations':
           tf_helpers.TensorFeature(
                         key='Rotations',
                         shape=[self.tuple_size, 3, 3],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
      'Translations':
           tf_helpers.TensorFeature(
                         key='Translations',
                         shape=[self.tuple_size, 3],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
    })

  def build_mask(self):
    p = self.n_pts
    v = self.n_views
    return tf.convert_to_tensor(1-np.kron(np.eye(v), np.ones((p,p))))

  def gen_sample_from_tuple(self, scene, tupl):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]
    mask = np.kron(np.ones((v,v))-np.eye(v),np.ones((n,n)))
    cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
    point_set = set.intersection(*[ cam_pt(t) for t in tupl ])
    # Build features
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [] 
    for camid in tupl:
      fset = [ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ]
      fset = sorted(fset, key=lambda x: x.id)
      features.append([ fset[x] for x in feat_perm ])
    # Build descriptors
    xy_pos_ = [ np.array([ f.pos for f in feats ]) for feats in features ]
    scale_ = [ np.array([ f.scale for f in feats ]) for feats in features ]
    orien_ = [ np.array([ f.orien for f in feats ]) for feats in features ]
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    # Apply permutation to features
    rids = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in rids ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))
    xy_pos = np.dot(perm,np.concatenate(xy_pos_))
    # We have to manually normalize these values as they are much larger than the others
    logscale = np.dot(perm, np.log(np.concatenate(scale_)) - 1.5).reshape(-1,1)
    orien = np.dot(perm,np.concatenate(orien_)).reshape(-1,1) / np.pi
    # Build Graph
    desc_norms = np.sqrt(np.sum(descs**2, 1).reshape(-1, 1))
    ndescs = descs / desc_norms
    Dinit = np.dot(ndescs,ndescs.T)
    # Rescaling
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin)
    L = np.copy(D)
    for i in range(v):
      for j in range(v):
        Lsub = L[n*i:n*(i+1),n*j:n*(j+1)]
        for u in range(n):
          Lsub[u,Lsub[u].argsort()[:-k]] = 0
    LLT = np.maximum(L,L.T)

    # TODO: Make intra-image graph Laplacians
    # Build dataset options
    InitEmbeddings = np.concatenate([ndescs,xy_pos,logscale,orien], axis=1)
    AdjMat = LLT*mask
    Degrees = np.diag(np.sum(AdjMat,0))
    TrueEmbedding = np.concatenate(perm_mats,axis=0)
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    Rotations = np.stack([ scene.cams[i].rot.T for i in tupl ], axis=0)
    Translations = np.stack([ -np.dot(scene.cams[i].rot.T, scene.cams[i].trans)
                              for i in tupl ], axis=0)

    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'Rotations': Rotations,
      'Translations': Translations,
      'NumViews': v,
      'NumPoints': n,
    }


