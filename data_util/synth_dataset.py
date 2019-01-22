# -*- coding: utf-8 -*-
import numpy as np
import os

from data_util import mydataset
from data_util import tf_helpers
import sim_graphs

class GraphSimDataset(mydataset.MyDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    super(GraphSimDataset, self).__init__(opts, params)
    d = self.n_pts*self.n_views
    self.features['Mask'] = \
           tf_helpers.TensorFeature(
                 key='Mask',
                 shape=[d, d],
                 dtype=self.dtype,
                 description='Mask for valid values of matrix')
    self.features['MaskOffset'] = \
           tf_helpers.TensorFeature(
                 key='MaskOffset',
                 shape=[d, d],
                 dtype=self.dtype,
                 description='Mask offset for loss')

  def gen_sample(self):
    # Pose graph and related objects
    params = self.dataset_params
    pose_graph = sim_graphs.PoseGraph(self.dataset_params,
                                      n_pts=self.n_pts,
                                      n_views=self.n_views)
    sz = (pose_graph.n_pts, pose_graph.n_pts)
    sz2 = (pose_graph.n_views, pose_graph.n_views)
    if params.sparse:
      mask = np.kron(pose_graph.adj_mat,np.ones(sz))
    else:
      mask = np.kron(np.ones(sz2)-np.eye(sz2[0]),np.ones(sz))

    perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
               for i in range(pose_graph.n_views) ]
    # Embedding objects
    TrueEmbedding = np.concatenate(perms_, 0)
    InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                      for i in range(pose_graph.n_views) ], 0)

    # Graph objects
    if not params.soft_edges:
      if params.descriptor_noise_var == 0:
        AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
        if params.sparse:
          AdjMat = AdjMat * mask
        else:
          AdjMat = AdjMat - np.eye(len(AdjMat))
        Degrees = np.diag(np.sum(AdjMat,0))
    else:
      if params.sparse and params.descriptor_noise_var > 0:
        AdjMat = pose_graph.get_feature_matching_mat()
        Degrees = np.diag(np.sum(AdjMat,0))

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))

    # Mask objects
    neg_offset = np.kron(np.eye(sz2[0]),np.ones(sz)-np.eye(sz[0]))
    Mask = AdjMat - neg_offset
    MaskOffset = neg_offset
    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'Mask': Mask.astype(self.dtype),
      'MaskOffset': MaskOffset.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': pose_graph.n_views,
      'NumPoints': pose_graph.n_pts,
    }

class GraphSimNoisyDataset(GraphSimDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    GraphSimDataset.__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = super(GraphSimNoisyDataset, self).gen_sample()

    # Graph objects
    p = self.n_pts
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.eye(p) + noise*(np.eye(p, k=-1) + np.eye(p, k=-1))
    AdjMat = np.dot(np.dot(TEmb, Noise), TEmb.T)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimGaussDataset(GraphSimDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    super(GraphSimGaussDataset, self).__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = super(GraphSimGaussDataset, self).gen_sample()

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.abs(np.random.randn(p*n,p*n)*noise)
    AdjMat = np.dot(TEmb, TEmb.T) + Noise - np.eye(p*n)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimSymGaussDataset(GraphSimDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    super(GraphSimSymGaussDataset, self).__init__(self, opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = super(GraphSimSymGaussDataset, self).gen_sample()

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    noise = self.dataset_params.noise_level
    TEmb = sample['TrueEmbedding']
    Noise = np.abs(np.random.randn(p*n,p*n)*noise)
    Mask = np.kron(np.ones((n,n))-np.eye(3),np.ones((p,p)))
    AdjMat = np.dot(TEmb, TEmb.T) + ((Noise+Noise.T)/2.0)*Mask - np.eye(p*n)
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimPairwiseDataset(GraphSimDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    super(GraphSimPairwiseDataset, self).__init__(opts, params)

  def gen_sample(self):
    # Pose graph and related objects
    sample = super(GraphSimPairwiseDataset, self).gen_sample()

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    noise = self.dataset_params.noise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    TEmb = sample['TrueEmbedding']
    AdjMat = np.zeros((p*n,p*n))
    for i in range(n):
      TEmb_i = TEmb[p*i:p*i+p,:]
      for j in range(i+1, n):
        TEmb_j = TEmb[p*j:p*j+p,:]
        Noise = (1-noise)*np.eye(p) + noise*sum([ perm(p) for i in range(r) ])
        Val_ij = np.dot(TEmb_i, np.dot(Noise, TEmb_j.T))
        AdjMat[p*i:p*i+p, p*j:p*j+p] = Val_ij
        AdjMat[p*j:p*j+p, p*i:p*i+p] = Val_ij.T
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

class GraphSimOutlierDataset(GraphSimDataset):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    super(GraphSimOutlierDataset, self).__init__(self, opts, params)

  def create_outlier_indeces(self, o, n):
    ind_pairs = [ (x,y) for x in range(n) for y in range(x+1,n) ]
    probs = [ 1.0/len(ind_pairs) ] * len(ind_pairs)
    outlier_ind_pairs = np.random.multinomial(o, probs, size=1)[0]
    outlier_sel = np.zeros((n,n), dtype=np.int64)
    for i in range(len(outlier_ind_pairs)):
      outlier_sel[ind_pairs[i]] = int(outlier_ind_pairs[i])
      outlier_sel[ind_pairs[i]] = (outlier_ind_pairs[i])
    # for i in range(n):
    #   for j in range(i+1,n):
    #     outlier_sel[i,j] = outlier_ind_pairs[i*n + j]
    #     outlier_sel[j,i] = outlier_ind_pairs[i*n + j]
    return outlier_sel

  def gen_sample(self):
    # Pose graph and related objects
    sample = super(GraphSimOutlierDataset, self).gen_sample()

    # Graph objects
    p = self.n_pts
    n = self.n_views 
    r = self.dataset_params.num_repeats
    o = self.dataset_params.num_outliers
    noise = self.dataset_params.noise_level
    perm = lambda p: np.eye(p)[np.random.permutation(p),:]
    TEmb = sample['TrueEmbedding']
    AdjMat = np.zeros((p*n,p*n))
    outlier_sel =  self.create_outlier_indeces(o, n)
    # Generate matrix
    for i in range(n):
      TEmb_i = TEmb[p*i:p*i+p,:]
      for j in range(i+1, n):
        TEmb_j = TEmb[p*j:p*j+p,:]
        if outlier_sel[i,j] > 0:
          Noise = np.eye(p)
          # for _ in range(outlier_sel[i,j]):
          for _ in range(1):
            s0, s1 = np.random.choice(range(p), size=2, replace=False)
            tmp = Noise[s1,:].copy()
            Noise[s1,:] = Noise[s0,:]
            Noise[s0,:] = tmp
          Val_ij = np.dot(TEmb_i, np.dot(Noise, TEmb_j.T))
        else:
          Val_ij = np.dot(TEmb_i, TEmb_j.T)
        AdjMat[p*i:p*i+p, p*j:p*j+p] = Val_ij
        AdjMat[p*j:p*j+p, p*i:p*i+p] = Val_ij.T
    AdjMat = np.minimum(1, AdjMat)
    Degrees = np.diag(np.sum(AdjMat,0))
    sample['AdjMat'] = AdjMat.astype(self.dtype)
    sample['Degrees'] = Degrees.astype(self.dtype)

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))
    sample['Laplacian'] = Laplacian.astype(self.dtype)

    return sample

