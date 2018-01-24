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

# 

# Parameters
k = 3 # Number of nearest neighbors
sz = 8
scale = 3
pts_scale = 0.1
N_poses = 5
N_pts = 9
n_iters = 1000
eye = np.eye(N_poses)

# Classes
Pose = collections.namedtuple("Pose", ["R","T"])
PoseEdge = collections.namedtuple("PoseEdge", ["idx", "g_ij"])
class PoseGraph(object):
  def __init__(self, N_poses, N_pts):
    self.n_poses = N_poses
    self.n_pts = N_pts
    # Generate poses
    sph = dim_normalize(np.random.randn(N_poses,3))
    rot = [ sph_rot(-sph[i]) for i in range(N_poses) ]
    trans = scale*sph
    # Create variables
    self.cmap = plt.cm.get_cmap('hsv',N_poses)
    self.pts_w = pts_scale*np.random.randn(N_pts,3)
    self.g_cw = [ Pose(R=rot[i],T=trans[i])
                  for i in range(N_poses) ]
    # Create graph
    dist_mat = 2 - 2*np.dot(sph, sph.T) + 3*eye
    AdjList0 = [ dist_mat[i].argsort()[:3].tolist() for i in range(N_poses) ]
    A = np.array([sum([ eye[j] for j in AdjList0[i] ]) for i in range(N_poses)])
    self.adj_mat = np.minimum(1, A.T + A)
    get_adjs = lambda adj: np.argwhere(adj.reshape(-1) > 0).T.tolist()[0]
    self.adj_list = []
    for i in range(N_poses):
      pose_edges = []
      for j in get_adjs(self.adj_mat[i]):
        Rij = np.dot(rot[i].T,rot[j]),
        Tij = normalize(np.dot(rot[i].T, trans[j] - trans[i])).reshape((3,1))
        pose_edges.append(PoseEdge(idx=j, g_ij=Pose(R=Rij, T=Tij)))
      self.adj_list.append(pose_edges)

  def get_proj(self, i):
    return planer_proj(np.dot(self.pts_w - self.g_cw[i].T,self.g_cw[i].R.T))

  def plot_index(self, idx, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(3*sz+0.2, sz))
    ax = fig.add_subplot(131, projection='3d')
    Ri = self.g_cw[idx].R
    Ti = self.g_cw[idx].T
    arr = np.array([0., 0.8])
    ax.plot(Ti[0] + Ri[0,0]*arr, Ti[1] + Ri[0,1]*arr, Ti[2] + Ri[0,2]*arr, c='r')
    ax.plot(Ti[0] + Ri[1,0]*arr, Ti[1] + Ri[1,1]*arr, Ti[2] + Ri[1,2]*arr, c='g')
    ax.plot(Ti[0] + Ri[2,0]*arr, Ti[1] + Ri[2,1]*arr, Ti[2] + Ri[2,2]*arr, c='b')
    for j in range(self.n_pts):
      ax.plot([Ti[0], self.pts_w[j,0]],
              [Ti[1], self.pts_w[j,1]],
              [Ti[2], self.pts_w[j,2]], c='k')
    ax.scatter(scale*np.array([-1., -1., -1., -1., +1., +1., +1., +1.]),
               scale*np.array([-1., -1., +1., +1., -1., -1., +1., +1.]),
               scale*np.array([-1., +1., -1., +1., -1., +1., -1., +1.]))
    ax.scatter(0, 0, 0, c='c', s=100)
    ax.scatter(self.pts_w[:,0], self.pts_w[:,1], self.pts_w[:,2])

    ax = fig.add_subplot(132, projection='3d')
    pts_c = np.dot(self.pts_w - Ti, Ri.T)
    ax.scatter(pts_c[:,0], pts_c[:,1], pts_c[:,2])
    ax.plot(0.8*arr, 0.0*arr, 0.0*arr, c='r')
    ax.plot(0.0*arr, 0.8*arr, 0.0*arr, c='g')
    ax.plot(0.0*arr, 0.0*arr, 0.8*arr, c='b')
    ax.scatter(1*np.array([-1., -1., -1., -1., +1., +1., +1., +1.]),
               1*np.array([-1., -1., +1., +1., -1., -1., +1., +1.]),
               1*np.array([-1., +1., -1., +1., -1., +1., -1., +1.]),c='k')

    ax = fig.add_subplot(133)
    ax.scatter(self.get_proj(idx)[:,0], self.get_proj(idx)[:,1])
    ax.scatter(0.3*np.array([-1., -1., +1., +1.]),
               0.3*np.array([-1., +1., -1., +1.]))


  def plot_poses(self, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(sz,sz))
    ax = fig.add_subplot(111, projection='3d')
    arr = np.array([0., 0.2])
    for i in range(self.n_poses):
      Ri = self.g_cw[i].R
      Ti = self.g_cw[i].T
      for k in range(3):
        ax.plot(*(Ti[j] + Ri[k,j]*arr for j in range(3)), c=self.cmap(i))
      for e in self.adj_list[i]:
        j = e.idx
        edge_lines = ([self.g_cw[i].T[k], self.g_cw[j].T[k]] for k in range(3))
        ax.plot(*edge_lines, c='k')
    ax.scatter(self.pts_w[:,0], self.pts_w[:,1], self.pts_w[:,2])
    return fig, ax


  def plot_planar_points(self, use_perm=False, perm=None, fig_num=1):
    fig = plt.figure(num=fig_num, figsize=(sz,sz))
    idxs = [ 0 ] + [ e.idx for e in self.adj_list[0] ]
    ax = [ fig.add_subplot(2,2,i+1) for i in range(4) ]
    connections = [ 0, 1, 3, 2 ]

    for i in range(4):
      ax[i].scatter(self.get_proj(idxs[i])[:,0], self.get_proj(idxs[i])[:,1])
      ax[i].set_title('Position {}'.format(idxs[i]), color=self.cmap(idxs[i]))
    for i in range(4):
      if use_perm and perm is None:
        perm = np.random.permutation(range(N_pts))
      elif not use_perm:
        perm = np.array(range(N_pts))
      i1 = connections[i]
      i2 = connections[(i + 1) % 4]
      for p in range(N_pts):
        con = ConnectionPatch(xyA=self.get_proj(idxs[i1])[perm[p],:2],
                              xyB=self.get_proj(idxs[i2])[p,:2],
                              coordsA="data",
                              coordsB="data",
                              axesA=ax[i1],
                              axesB=ax[i2],
                              color=self.cmap(p))
        ax[i1].add_artist(con)
        con = ConnectionPatch(xyA=self.get_proj(idxs[i2])[p,:2],
                              xyB=self.get_proj(idxs[i1])[perm[p],:2],
                              coordsA="data",
                              coordsB="data",
                              axesA=ax[i2],
                              axesB=ax[i1],
                              color=self.cmap(p))
        ax[i2].add_artist(con)



if __name__ == '__main__':
  print("Synthetic Pose Graphs")
  # if full_plot:
  if True:
    pose_graph = PoseGraph(N_poses, N_pts)
    pose_graph.plot_poses()
    plt.show()
    for i in range(pose_graph.n_poses):
      pose_graph.plot_index(i)
      plt.show()
    pose_graph.plot_planar_points()
    pose_graph.plot_planar_points(fig_num=2)
    plt.show()




