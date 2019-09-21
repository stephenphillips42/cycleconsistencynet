import os
import sys
import numpy as np
import scipy.linalg as la
import imageio
import networkx as nx
import functools

import tensorflow as tf
import cv2
from graph_nets import utils_np

from data_util import graph_dataset
from data_util import tf_helpers


def size_to_octave_desc(s, octv_size=3.6, nlayers=3):
  octv = np.log2(s / octv_size)
  octave = int(np.floor(octv))
  layer  = int(np.floor(nlayers * octv) - nlayers*octave) + 1
  if octave < -1: # This is apparently the minimum
    octave, layer = -1, 1
  return (octave & 255) | ((layer & 255) << 8)

def get_size_angle(x0, y0, H, kpt):
  s = kpt.size
  theta = kpt.angle
  dx_, dy_ = np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))
  x_0, y_0, _ = tuple(h_apply(H, (x0+s*dx_, y0+s*dy_, 1)))
  x_1, y_1, _ = tuple(h_apply(H, (x0-s*dy_, y0+s*dx_, 1)))
  x_2, y_2, _ = tuple(h_apply(H, (x0+s*dy_, y0-s*dx_, 1)))
  x_3, y_3, _ = tuple(h_apply(H, (x0-s*dx_, y0-s*dy_, 1)))
  size_new = np.mean([ np.linalg.norm((x_0 - x_3, y_0 - y_3))/2,
                       np.linalg.norm(((x_1 - x_2, y_1 - y_2)))/2 ])
  angle_new = np.arctan2(-y_0 + y_3, x_0 - x_3)
  angle_new = np.rad2deg(angle_new + 2*np.pi*(angle_new < 0))
  return size_new, angle_new

def h_apply(H, x):
  y = H @ x
  return y / y[-1]

def dist_mat(x,y):
  x2 = np.sum(x**2, 0).reshape(-1,1)
  y2 = np.sum(y**2, 0).reshape(1,-1)
  Dist = np.sqrt(x2 + y2 - x.T @ y - y.T @ x)
  return Dist

def row_comp(x, thresh):
  Xstr = ''.join([ '0' if x_ < thresh else '1' for x_ in x ])
  return Xstr

def distint_locs(xy, thresh):
  Dist = dist_mat(xy, xy)
  d_list = [ Dist[i] for i in range(len(Dist)) ]
  D_ = sorted(d_list, key=lambda x: row_comp(x, thresh))
  Duniq = []
  ids_ = []
  for i in range(len(D_)-1):
    di = 1.0 * (D_[i] < thresh)
    di1 = 1.0 * (D_[i+1] < thresh)
    if not np.isclose(np.linalg.norm(di-di1), 0):
      Duniq.append(D_[i])
      ids_.append(i)
  Duniq.append(D_[-1])
  ids_.append(len(D_)-1)
  return ids_

def load_graffiti(graffiti_dir='./graff', N_imgs=6, N_data=80, thresh=15):
  # Load graffiti dataset images
  n = N_data
  N_points = 10*n
  data_files = [
    os.path.join(graffiti_dir, 'img{}.png'.format(i+1)) for i in range(N_imgs)
  ]
  imgs = [
    imageio.imread(fname)[:,:,:3] for fname in data_files
  ]
  # Build Homographies 
  H = [ np.eye(3) ]
  for i in range(1,N_imgs):
    h = []
    with open(os.path.join(graffiti_dir, 'H1to{}p'.format(i+1)), 'r') as f:
      for l in f:
        h.append([ float(x) for x in l.split() ])
    H.append(np.array(h))
  # Compute sift descriptors for first image
  sift = cv2.xfeatures2d.SIFT_create(N_points)
  keypoints0, descs0 = sift.detectAndCompute(imgs[0], None)
  # Select features with valid x,y coordinates not too close to others
  descs0 = [ d / np.linalg.norm(d) for d in descs0[:N_points] ]
  xy0 = np.stack([ [ k.pt[0], k.pt[1], 1 ] for k in keypoints0 ], axis=1)
  ids1 = distint_locs(xy0, thresh)
  xy0 = np.stack([ xy0[:, ii] for ii in ids1 ], axis=1)
  sel = xy0[0,:] < np.inf # Should all be true
  for j in range(1,N_imgs):
    xyj = h_apply(H[j], xy0)
    and_lists = [ 
        xyj[0,:] > 0, xyj[1,:] > 0,
        xyj[0,:] < imgs[j].shape[1], xyj[1,:] < imgs[j].shape[0],
    ]
    sel = list(functools.reduce(np.logical_and, and_lists, sel))
  # Sort features by their selectiveness
  descs0 = np.stack([ descs0[ids1[i]] for i in range(len(ids1)) if sel[i] ])
  keypoints0 = [ keypoints0[ids1[i]] for i in range(len(ids1)) if sel[i] ]
  Sim = np.dot(descs0,descs0.T)
  SimSortList = [ np.sort(s)[::-1] for s in Sim ]
  vec = (np.arange(len(descs0))[::-1])**(2)
  best_idxs = sorted(np.arange(len(descs0)), key=lambda x: SimSortList[x] @ vec)
  # Select the features and pull out their features
  xy_ = xy0[:, sel]
  xy = [ xy_[:, best_idxs[:n]] ]
  descs = [ descs0[best_idxs[:n], :] ]
  keypoints = [ [ keypoints0[i] for i in best_idxs[:n] ] ]
  # Compute the features for their location in the other images
  for i in range(1,N_imgs):
    xy.append(h_apply(H[i], xy[0]))
    kpts = []
    for j in range(xy[0].shape[1]):
      x0, y0 = xy[0][0,j], xy[0][1,j]
      size_new, angle_new = get_size_angle(x0, y0, H[i], keypoints0[j])
      octv = size_to_octave_desc(size_new)
      kpts.append(cv2.KeyPoint(x=xy[i][0,j], y=xy[i][1,j],
                               _size=size_new,
                               _angle=angle_new,
                               _response=keypoints0[j].response,
                               _octave=octv,
                               _class_id=keypoints0[j].class_id))
    keypoints.append(kpts)
  # Compute the full descriptors
  # These include the feature descriptor, x, y, log scale, and orientation
  FullDescs = []
  for kpts, img in zip(keypoints, imgs):
    h, w, c = imgs[i].shape
    _, descs = sift.compute(img, kpts)
    d_ = [ (d / np.linalg.norm(d)) for d in descs ]
    dkp_ = [ np.array(list(d_[i]) + [
                 (k.pt[0]-w/2)/w, (k.pt[1]-h/2)/h,
                 np.log(k.size), np.deg2rad(k.angle)
             ])
             for i, k in enumerate(kpts) ]
    FullDescs.append(dkp_)
  return np.stack(FullDescs)


class GraffitiDataset(GraphDataset):
  def __init__(self, opts, params):
    super().__init__(opts, params)
    # TODO: Hack - should make new parameter (imgs_dir or graffiti_dir)
    self.graff_dir = opts.rome16k_dir
    p = self.n_pts
    v = self.n_views
    d = p*v
    self.features.update({
      'adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='adj_mat',
                         shape=[d, d],
                         description='Sparse adjacency matrix of graph'),
      'true_adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='true_adj_mat',
                         shape=[d, d],
                         description='Sparse ground truth adjacency of graph'),
      'true_match':
           tf_helpers.VarLenIntListFeature(
                         key='true_match',
                         dtype='int64',
                         description='Ground truth matches of graph'),
    })

  def gen_sample(self):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]

    # Apply permutation to features
    perms = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in perms ]
    perm = la.block_diag(*perm_mats)
    # Finish the inital node embeddings
    init_emb = load_graffiti(graffiti_dir=self.graff_dir,
                             N_imgs=v, N_data=n,
                             thresh=15)
    ndescs = init_emb[:, :128] # SIFT descriptor size
    # Build Graph
    Dinit = np.dot(ndescs,ndescs.T)
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin) # Rescaling
    A_ = np.zeros_like(D)
    for i in range(v):
      for j in range(v):
        if i == j:
          continue
        # Perhaps not the most efficient... oh well
        Asub = np.copy(D[n*i:n*(i+1),n*j:n*(j+1)])
        for u in range(n):
          Asub[u,Asub[u].argsort()[:-k]] = 0
        A_[n*i:n*(i+1),n*j:n*(j+1)] = Asub
    adj_mat = np.maximum(A_, A_.T)
    # Build dataset ground truth
    true_emb = np.concatenate(perm_mats,axis=0)
    gt_adj_mat = np.dot(true_emb, true_emb.T)
    matches_ = np.concatenate(perms)
    rots = np.stack([ np.eye(3) for i in tupl ], axis=0)
    trans = np.stack([ np.zeros(3) for i in tupl ], axis=0)
    # Build spart graph representation
    G_nx = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
    node_attrs = { i : init_emb[i].astype(np.float32)
                   for i in range(len(G_nx)) }
    edges_attrs = { (i, j) : np.array([ adj_mat[i,j] ]).astype(np.float32)
                    for (i,j) in G_nx.edges }
    nx.set_node_attributes(G_nx, node_attrs, 'features')
    nx.set_edge_attributes(G_nx, edges_attrs, 'features')
    # Make to dictionary, with additional graph attributes
    G = utils_np.networkx_to_data_dict(G_nx)
    G['globals'] = np.array([0,0])
    G['adj_mat'] = graph_dataset.np_dense_to_sparse(adj_mat)
    G['true_adj_mat'] = graph_dataset.np_dense_to_sparse(gt_adj_mat)
    G['true_match'] = matches_
    G['rots'] = rots
    G['trans'] = trans
    return G



