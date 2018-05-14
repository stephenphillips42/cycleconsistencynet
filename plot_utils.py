import os
import sys
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

import sim_graphs

def axes3d():
  fig = plt.figure()
  return fig, fig.add_subplot(111, projection='3d')

def plot_poses(sgraph, fig_num=1):
  cmap = plt.cm.get_cmap('hsv',self.n_poses)
  fig = plt.figure(num=fig_num, figsize=(sgraph.opts.viewer_size,
                                         sgraph.opts.viewer_size))
  ax = fig.add_subplot(111, projection='3d')
  arr = np.array([0., 0.2])
  for i in range(sgraph.n_poses):
    Ri, Ti = sgraph.g_cw[i]
    for j in range(3):
      axis_line_j = (Ti[k] + Ri[j,k]*arr for k in range(3))
      ax.plot(*axis_line_j, c=cmap(i))
    for e in sgraph.adj_list[i]:
      Tj = sgraph.g_cw[e.idx].T
      edge_line_k = ([Ti[k], Tj[k]] for k in range(3))
      ax.plot(*edge_line_k, c='k')
  ax.scatter(sgraph.pts_w.p[:,0], sgraph.pts_w.p[:,1], sgraph.pts_w.p[:,2])
  return fig, ax

def plot_index(sgraph, idx, fig_num=1):
  fig = plt.figure(num=fig_num, figsize=(3*sgraph.opts.viewer_size+0.2,
                                         sgraph.opts.viewer_size))
  ax = fig.add_subplot(131, projection='3d')
  Ri, Ti = sgraph.g_cw[idx]
  arr = np.array([0., 0.8])
  clrs = [ 'r', 'g', 'b' ]
  for j in range(3):
    axes_line_j = ( Ti[k] + Ri[j,k]*arr for k in range(3) )
    ax.plot(*axes_line_j, c=clrs[j])
  for j in range(sgraph.n_pts):
    pt_line_j = ( [Ti[k], sgraph.pts_w.p[j,k]] for k in range(3) )
    ax.plot(*pt_line_j, c='k')
  ax.scatter(sgraph.opts.scale*np.array([-1.] * 4 + [+1.] * 4),
             sgraph.opts.scale*np.array(([-1.] * 2 + [+1.] * 2) * 2),
             sgraph.opts.scale*np.array([-1., +1.] * 4))
  ax.scatter(0, 0, 0, c='c', s=100)
  ax.scatter(sgraph.pts_w.p[:,0], sgraph.pts_w.p[:,1], sgraph.pts_w.p[:,2])

  ax = fig.add_subplot(132, projection='3d')
  pts_c = np.dot(sgraph.pts_w.p - Ti, Ri.T)
  ax.scatter(pts_c[:,0], pts_c[:,1], pts_c[:,2])
  ax.plot(0.8*arr, 0.0*arr, 0.0*arr, c='r')
  ax.plot(0.0*arr, 0.8*arr, 0.0*arr, c='g')
  ax.plot(0.0*arr, 0.0*arr, 0.8*arr, c='b')
  ax.scatter(1.*np.array([-1.] * 4 + [+1.] * 4),
             1.*np.array(([-1.] * 2 + [+1.] * 2) * 2),
             1.*np.array([-1., +1.] * 4),c='k')

  ax = fig.add_subplot(133)
  proj_pts = sgraph.get_proj(idx)
  ax.scatter(proj_pts.p[:,0], proj_pts.p[:,1])
  ax.scatter(0.3*np.array([-1., -1., +1., +1.]),
             0.3*np.array([-1., +1., -1., +1.]))

def plot_planar_points(sgraph, use_perm=False, perm=None, fig_num=1):
  cmap = plt.cm.get_cmap('hsv',self.n_poses)
  fig = plt.figure(num=fig_num, figsize=(sgraph.opts.viewer_size,
                                         sgraph.opts.viewer_size))
  idxs = [ 0 ] + [ e.idx for e in sgraph.adj_list[0] ]
  ax = [ fig.add_subplot(2,2,i+1) for i in range(4) ]
  connections = [ 0, 1, 3, 2 ]

  for i in range(4):
    proj_pts = sgraph.get_proj(idxs[i])
    ax[i].scatter(proj_pts.p[:,0], proj_pts.p[:,1])
    ax[i].set_title('Position {}'.format(idxs[i]), color=cmap(idxs[i]))
  for i in range(4):
    if use_perm and perm is None:
      perm = np.random.permutation(range(sgraph.n_pts))
    elif not use_perm:
      perm = np.array(range(sgraph.n_pts))
    i1 = connections[i]
    i2 = connections[(i + 1) % 4]
    for p in range(sgraph.n_pts):
      proj_pts1 = sgraph.get_proj(idxs[i1])
      proj_pts2 = sgraph.get_proj(idxs[i2])
      con = ConnectionPatch(xyA=proj_pts1.p[perm[p],:2],
                            xyB=proj_pts2.p[p,:2],
                            coordsA="data",
                            coordsB="data",
                            axesA=ax[i1],
                            axesB=ax[i2],
                            color=cmap(p))
      ax[i1].add_artist(con)
      con = ConnectionPatch(xyA=proj_pts2.p[p,:2],
                            xyB=proj_pts1.p[perm[p],:2],
                            coordsA="data",
                            coordsB="data",
                            axesA=ax[i2],
                            axesB=ax[i1],
                            color=cmap(p))
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
  

