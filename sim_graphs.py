# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import scipy.linalg as la
from tqdm import tqdm

from myutils import *
import options

# Parameters
desc_var = 1.0
desc_noise_var = 0.4 # Somewhere between 0.7 and 0.8 seems optimal

# Classes
Points = collections.namedtuple("Points", ["p","d"]) # position and descriptor
Pose = collections.namedtuple("Pose", ["R","T"])
PoseEdge = collections.namedtuple("PoseEdge", ["idx", "g_ij"])
class PoseGraph(object):
  def __init__(self, opts, n_views=None, n_pts=None):
    self.opts = opts
    if n_views is None:
      self.n_views = np.random.randint(opts.min_views, opts.max_views+1)
    else:
      self.n_views = n_views
    if n_pts is None:
      self.n_pts = np.random.randint(opts.min_points, opts.max_points+1)
    else:
      self.n_pts = n_pts
    # Generate poses
    sph = dim_normalize(np.random.randn(self.n_views,3))
    rot = [ sph_rot(-sph[i]) for i in range(self.n_views) ]
    trans = self.opts.scale*sph
    # Create variables
    pts = self.opts.points_scale*np.random.randn(self.n_pts,3)
    self.desc_dim = self.opts.descriptor_dim
    self.desc_var = self.opts.descriptor_var
    desc = self.desc_var*np.random.randn(self.n_pts, self.desc_dim)
    self.pts_w = Points(p=pts,d=desc)
    self.g_cw = [ Pose(R=rot[i],T=trans[i]) for i in range(self.n_views) ]
    # Create graph
    eye = np.eye(self.n_views)
    dist_mat = 2 - 2*np.dot(sph, sph.T) + 3*eye
    AdjList0 = [ dist_mat[i].argsort()[:self.opts.knn].tolist() 
                 for i in range(self.n_views) ]
    A = np.array([ sum([ eye[j] for j in AdjList0[i] ])
                   for i in range(self.n_views) ])
    self.adj_mat = np.minimum(1, A.T + A)
    get_adjs = lambda adj: np.argwhere(adj.reshape(-1) > 0).T.tolist()[0]
    self.adj_list = []
    for i in range(self.n_views):
      pose_edges = []
      for j in get_adjs(self.adj_mat[i]):
        Rij = np.dot(rot[i].T,rot[j]),
        Tij = normalize(np.dot(rot[i].T, trans[j] - trans[i])).reshape((3,1))
        pose_edges.append(PoseEdge(idx=j, g_ij=Pose(R=Rij, T=Tij)))
      self.adj_list.append(pose_edges)

  def get_random_state(self, pts):
    seed = (np.sum(np.abs(pts**5)))
    return np.random.RandomState(int(seed))

  def get_proj(self, i):
    pts_c = np.dot(self.pts_w.p - self.g_cw[i].T, self.g_cw[i].R.T)
    s = self.get_random_state(pts_c)
    perm = s.permutation(self.n_pts)
    proj_pos = planer_proj(pts_c)[perm,:2]
    var = self.opts.descriptor_noise_var
    desc_noise = var*s.randn(self.n_pts, self.desc_dim)
    descs = self.pts_w.d[perm,:] + desc_noise
    return Points(p=proj_pos, d=descs)

  def get_perm(self, i):
    pts_c = np.dot(self.pts_w.p - self.g_cw[i].T, self.g_cw[i].R.T)
    s = self.get_random_state(pts_c)
    return s.permutation(self.n_pts)

  def get_all_permutations(self):
    return [ self.get_perm(i) for i in range(self.n_views) ]

  def get_feature_matching_mat(self):
    n = self.n_pts
    m = self.n_views
    perms = [ self.get_perm(i) for i in range(m) ]
    sigma = 2
    total_graph = np.zeros((n*m, n*m))
    for i in range(m):
      for j in ([ e.idx for e in self.adj_list[i] ]):
        s_ij = np.zeros((n, n))
        descs_i = self.get_proj(i).d
        descs_j = self.get_proj(j).d
        for x in range(n):
          u = perms[i][x]
          for y in range(n):
            v = perms[j][y]
            s_ij[u,v] = np.exp(-np.linalg.norm(descs_i[u] - descs_j[v])/(sigma))
        total_graph[i*n:(i+1)*n, j*n:(j+1)*n] = s_ij
    return total_graph # + np.eye(n*m)




