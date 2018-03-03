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

from nputils import *
import options

# Parameters
desc_var = 1.0
desc_noise_var = 0.4 # Somewhere between 0.7 and 0.8 seems optimal

# Classes
Points = collections.namedtuple("Points", ["p","d"]) # position and descriptor
Pose = collections.namedtuple("Pose", ["R","T"])
PoseEdge = collections.namedtuple("PoseEdge", ["idx", "g_ij"])
class PoseGraph(object):
  def __init__(self, opts, n_poses=None, n_pts=None):
    self.opts = opts
    if n_poses is None:
      self.n_poses = np.random.randint(opts.min_views, opts.max_views+1)
    else:
      self.n_poses = n_poses
    if n_pts is None:
      self.n_pts = np.random.randint(opts.min_points, opts.max_points+1)
    else:
      self.n_pts = n_pts
    # Generate poses
    sph = dim_normalize(np.random.randn(self.n_poses,3))
    rot = [ sph_rot(-sph[i]) for i in range(self.n_poses) ]
    trans = self.opts.scale*sph
    # Create variables
    self.cmap = plt.cm.get_cmap('hsv',self.n_poses)
    pts = self.opts.points_scale*np.random.randn(self.n_pts,3)
    self.desc_dim = self.opts.descriptor_dim
    self.desc_var = self.opts.descriptor_var
    desc = self.desc_var*np.random.randn(self.n_pts, self.desc_dim)
    self.pts_w = Points(p=pts,d=desc)
    self.g_cw = [ Pose(R=rot[i],T=trans[i]) for i in range(self.n_poses) ]
    # Create graph
    eye = np.eye(self.n_poses)
    dist_mat = 2 - 2*np.dot(sph, sph.T) + 3*eye
    AdjList0 = [ dist_mat[i].argsort()[:self.opts.knn].tolist() 
                 for i in range(self.n_poses) ]
    A = np.array([ sum([ eye[j] for j in AdjList0[i] ])
                   for i in range(self.n_poses) ])
    self.adj_mat = np.minimum(1, A.T + A)
    get_adjs = lambda adj: np.argwhere(adj.reshape(-1) > 0).T.tolist()[0]
    self.adj_list = []
    for i in range(self.n_poses):
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
    return [ self.get_perm(i) for i in range(self.n_poses) ]

  def get_feature_matching_mat(self):
    n = self.n_pts
    m = self.n_poses
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

  def plot_poses(self, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(self.opts.viewer_size,
                                           self.opts.viewer_size))
    ax = fig.add_subplot(111, projection='3d')
    arr = np.array([0., 0.2])
    for i in range(self.n_poses):
      Ri, Ti = self.g_cw[i]
      for j in range(3):
        axis_line_j = (Ti[k] + Ri[j,k]*arr for k in range(3))
        ax.plot(*axis_line_j, c=self.cmap(i))
      for e in self.adj_list[i]:
        Tj = self.g_cw[e.idx].T
        edge_line_k = ([Ti[k], Tj[k]] for k in range(3))
        ax.plot(*edge_line_k, c='k')
    ax.scatter(self.pts_w.p[:,0], self.pts_w.p[:,1], self.pts_w.p[:,2])
    return fig, ax

  def plot_index(self, idx, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(3*self.opts.viewer_size+0.2,
                                           self.opts.viewer_size))
    ax = fig.add_subplot(131, projection='3d')
    Ri, Ti = self.g_cw[idx]
    arr = np.array([0., 0.8])
    clrs = [ 'r', 'g', 'b' ]
    for j in range(3):
      axes_line_j = ( Ti[k] + Ri[j,k]*arr for k in range(3) )
      ax.plot(*axes_line_j, c=clrs[j])
    for j in range(self.n_pts):
      pt_line_j = ( [Ti[k], self.pts_w.p[j,k]] for k in range(3) )
      ax.plot(*pt_line_j, c='k')
    ax.scatter(self.opts.scale*np.array([-1.] * 4 + [+1.] * 4),
               self.opts.scale*np.array(([-1.] * 2 + [+1.] * 2) * 2),
               self.opts.scale*np.array([-1., +1.] * 4))
    ax.scatter(0, 0, 0, c='c', s=100)
    ax.scatter(self.pts_w.p[:,0], self.pts_w.p[:,1], self.pts_w.p[:,2])

    ax = fig.add_subplot(132, projection='3d')
    pts_c = np.dot(self.pts_w.p - Ti, Ri.T)
    ax.scatter(pts_c[:,0], pts_c[:,1], pts_c[:,2])
    ax.plot(0.8*arr, 0.0*arr, 0.0*arr, c='r')
    ax.plot(0.0*arr, 0.8*arr, 0.0*arr, c='g')
    ax.plot(0.0*arr, 0.0*arr, 0.8*arr, c='b')
    ax.scatter(1.*np.array([-1.] * 4 + [+1.] * 4),
               1.*np.array(([-1.] * 2 + [+1.] * 2) * 2),
               1.*np.array([-1., +1.] * 4),c='k')

    ax = fig.add_subplot(133)
    proj_pts = self.get_proj(idx)
    ax.scatter(proj_pts.p[:,0], proj_pts.p[:,1])
    ax.scatter(0.3*np.array([-1., -1., +1., +1.]),
               0.3*np.array([-1., +1., -1., +1.]))

  def plot_planar_points(self, use_perm=False, perm=None, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(self.opts.viewer_size,
                                           self.opts.viewer_size))
    idxs = [ 0 ] + [ e.idx for e in self.adj_list[0] ]
    ax = [ fig.add_subplot(2,2,i+1) for i in range(4) ]
    connections = [ 0, 1, 3, 2 ]

    for i in range(4):
      proj_pts = self.get_proj(idxs[i])
      ax[i].scatter(proj_pts.p[:,0], proj_pts.p[:,1])
      ax[i].set_title('Position {}'.format(idxs[i]), color=self.cmap(idxs[i]))
    for i in range(4):
      if use_perm and perm is None:
        perm = np.random.permutation(range(self.n_pts))
      elif not use_perm:
        perm = np.array(range(self.n_pts))
      i1 = connections[i]
      i2 = connections[(i + 1) % 4]
      for p in range(self.n_pts):
        proj_pts1 = self.get_proj(idxs[i1])
        proj_pts2 = self.get_proj(idxs[i2])
        con = ConnectionPatch(xyA=proj_pts1.p[perm[p],:2],
                              xyB=proj_pts2.p[p,:2],
                              coordsA="data",
                              coordsB="data",
                              axesA=ax[i1],
                              axesB=ax[i2],
                              color=self.cmap(p))
        ax[i1].add_artist(con)
        con = ConnectionPatch(xyA=proj_pts2.p[p,:2],
                              xyB=proj_pts1.p[perm[p],:2],
                              coordsA="data",
                              coordsB="data",
                              axesA=ax[i2],
                              axesB=ax[i1],
                              color=self.cmap(p))
        ax[i2].add_artist(con)

if __name__ == '__main__':
  opts = options.get_opts()
  print("Synthetic Pose Graphs")
  plot_level = 0
  pose_graph = PoseGraph(opts)
  perms = [ np.eye(pose_graph.n_pts)[:, pose_graph.get_perm(i)]
            for i in range(pose_graph.n_poses) ]
  perms_ = np.concatenate(perms, 0)
  graph0 = pose_graph.get_feature_matching_mat()
  aug_adj = np.eye(pose_graph.n_poses) + pose_graph.adj_mat
  mask = np.kron(aug_adj, np.ones((pose_graph.n_pts, pose_graph.n_pts)))
  graph_true = mask*np.dot(perms_,perms_.T)
  fig = plt.figure()
  ax = fig.add_subplot(1,2,1)
  ax.imshow(graph0)
  ax = fig.add_subplot(1,2,2)
  ax.imshow(graph_true)
  plt.show()
  plt.imshow(graph0)
  plt.colorbar()
  plt.show()
  _, sv_true, _ = np.linalg.svd(graph_true)
  u, sv, v = np.linalg.svd(graph0)
  plt.plot(sv_true)
  plt.plot(sv)
  plt.show()
  myN = pose_graph.n_pts
  print(u[:,:myN].shape, np.diag(sv[:myN]).shape, v[:myN,:].shape)
  graph_recon = np.dot(np.dot(u[:,:myN], np.diag(sv[:myN])),v[:myN,:])
  plt.imshow(graph_recon)
  plt.colorbar()
  plt.show()


  if plot_level > 0:
    pose_graph.plot_poses()
    plt.show()
  if plot_level > 1:
    pose_graph.plot_planar_points()
    pose_graph.plot_planar_points(fig_num=2,use_perm=True)
    plt.show()
  if plot_level > 2:
    for i in range(pose_graph.n_poses):
      pose_graph.plot_index(i)
      plt.show()




