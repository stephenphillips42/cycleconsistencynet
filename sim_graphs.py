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
# from init_optims import *


PoseGraph = collections.namedtuple("PoseGraph", ["nodes","edges"])
PoseEdge = collections.namedtuple("PoseEdge", ["u", "v", "transform"])

k = 3 # Number of nearest neighbors
sz = 8
N_sph = 30
N_pts = 5 # 13
eps = [1e-4, 1e-6, 1e-2]
sph = dim_normalize(np.random.randn(N_sph,3))
rot = [ sph_rot(sph[i]) for i in range(N_sph) ]
trans = 3*sph
cmap = plt.cm.get_cmap('hsv',N_sph)
pts = 0.1*np.random.randn(N_pts,3)
n_iters = 1000
connections = [ 0, 1, 3, 2 ]

eye = np.eye(N_sph)
D = 2 - 2*np.dot(sph, sph.T) + 3*eye
AdjList0 = [ D[i].argpartition(3)[:3].tolist() for i in range(N_sph) ]
A = np.array([ sum([ eye[j] for j in AdjList0[i] ]) for i in range(N_sph) ])
AdjMat = np.minimum(1, A.T + A)
AdjList = [ np.argwhere(AdjMat[i].reshape(-1) > 0).T.tolist()[0] for i in range(N_sph) ]
degs = np.array([ len(AdjList[i]) for i in range(len(AdjList)) ])

proj_pts = [ 0 for i in range(N_sph) ]
for i in range(N_sph):
  idx = i
  T = trans[idx]
  R = sph_rot(sph[idx])
  proj_pts[i] = planer_proj(np.dot(pts,R.T) + T)

# Relative Orientation List
G = [ [ 0 for j in range(N_sph) ] for i in range(N_sph) ]
for i in range(N_sph):
  for aj in range(len(AdjList[i])):
    j = AdjList[i][aj]
    G[i][j] = (np.dot(rot[i].T,rot[j]),
               normalize(np.dot(rot[i].T, trans[j] - trans[i])).reshape((3,1)),
               la.norm(trans[j] - trans[i]))


def plot_points(proj_pts, use_perm=False, perm=None, fig_num=1):
  fig = plt.figure(num=fig_num, figsize=(sz,sz))
  idxs = [ 0 ] + AdjList[0]
  ax = [ fig.add_subplot(2,2,i+1) for i in range(4) ]

  for i in range(4):
    ax[i].scatter(proj_pts[idxs[i]][:,0], proj_pts[idxs[i]][:,1])
    ax[i].set_title('Position {}'.format(idxs[i]), color=cmap(idxs[i]))
  for i in range(4):
    if use_perm and perm is None:
      perm = np.random.permutation(range(N_pts))
    else:
      perm = np.array(range(N_pts))
    i1 = connections[i]
    i2 = connections[(i + 1) % 4]
    for p in range(N_pts):
      con = ConnectionPatch(xyA=proj_pts[idxs[i1]][perm[p],:2], xyB=proj_pts[idxs[i2]][p,:2], coordsA="data", coordsB="data",
                            axesA=ax[i1], axesB=ax[i2], color=cmap(p))
      ax[i1].add_artist(con)
      con = ConnectionPatch(xyA=proj_pts[idxs[i2]][p,:2], xyB=proj_pts[idxs[i1]][perm[p],:2], coordsA="data", coordsB="data",
                            axesA=ax[i2], axesB=ax[i1], color=cmap(p))
      ax[i2].add_artist(con)



if __name__ == '__main__':
  plot = True
  if plot:
    plt.imshow(AdjMat)
    plt.show()

    plot_points(proj_pts, fig_num=2)
    plt.show()

    plot_points(proj_pts, use_perm=True, fig_num=3)
    plt.show()

  perms = [ np.random.permutation(N_pts) for i in xrange(N_sph) ]
  Eye = np.eye(N_pts)
  perm_mats = [ np.stack([ Eye[perms[i][p]] for p in xrange(N_pts) ], axis=0) for i in range(N_sph) ]
  P = np.concatenate(perm_mats[:3], axis=0)
  print(P)
  A = np.dot(P,P.T)
  d, e = np.linalg.eigh(A)
  P2 = np.abs(np.round(e[:,-N_pts:]*np.sqrt(3)))
  print("A.shape")
  print(A.shape)
  print("d")
  print(d)
  print("e[:,-N_pts:]*np.sqrt(3)")
  print(e[:,-N_pts:]*np.sqrt(3))
  print("P")
  print(P)
  print("P2")
  print(P2)
  print("np.dot(P.T,P)")
  print(np.dot(P.T,P))
  Q = np.dot(P.T,P2)/3
  print("Q")
  print(Q)
  print("la.norm(np.dot(P,Q) - np.abs(np.round(e[:,-N_pts:]*np.sqrt(3))))")
  print(la.norm(np.dot(P,Q) - np.abs(np.round(e[:,-N_pts:]*np.sqrt(3)))))
  print("e[:,-N_pts:].shape")
  print(e[:,-N_pts:].shape)
  print("la.norm(3*np.dot(e[:,-N_pts:], e[:,-N_pts:].T) - A)")
  print(la.norm(3*np.dot(e[:,-N_pts:], e[:,-N_pts:].T) - A))




