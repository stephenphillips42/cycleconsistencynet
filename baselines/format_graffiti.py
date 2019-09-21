# #!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import collections
import itertools
import functools
import time

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
# import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.signal as sig
import imageio
import cv2
import pickle
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from operator import indexOf
START_TIME = time.time()
plot = 1
# threshs = [ 1e-1, 1, 2, 5 ]
threshs = [ 1e-1, 1, 2, 5, 10, 15 ]
TAU = threshs[-1]
data_files = [
	'graff/img{}.png'.format(i+1) for i in range(6)
]

imgs = [
  imageio.imread(fname)[:,:,:3] for fname in data_files
]

N_imgs = len(imgs)
N_points = 800
N_data = 80

def cmp_to_key(mycmp):
  'Convert a cmp= function into a key= function'
  class K:
    def __init__(self, obj, *args):
      self.obj = obj
    def __lt__(self, other):
      return mycmp(self.obj, other.obj) < 0
    def __gt__(self, other):
      return mycmp(self.obj, other.obj) > 0
    def __eq__(self, other):
      return mycmp(self.obj, other.obj) == 0
    def __le__(self, other):
      return mycmp(self.obj, other.obj) <= 0
    def __ge__(self, other):
      return mycmp(self.obj, other.obj) >= 0
    def __ne__(self, other):
      return mycmp(self.obj, other.obj) != 0
  return K

def unpackSIFTOctave(_octave):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    # _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)

# def mybinary_helper(b, n):
#   if s < 0:
#     return mybinary_helper(2**n - b)
#   return str(s) if s<=1 else mybinary_helper(s>>1) + str(s&1)
# 
# def mybinary(b, n=8):
#   if b > 2**(n-1)-1 or b < -2**(n-1):
#     print("ERROR: Cannot handle {} binary string with only {} bits".format(b,n)
#     sys.exit()
#   bits = mybinary_helper(b, n)
#   return ''.join([ '0' for i in range(8-len(b))  ]) + b

def size_to_octave_desc(s):
  octave = int(np.floor(np.log2(s / 3.6)))
  layer  = int(np.floor(3 * np.log2(s / 3.6)) - 3*octave) + 1
  if octave < -1: # This is apparently the minimum
    octave, layer = -1, 1
  return (octave & 255) | ((layer & 255) << 8)

def h_apply(H, x):
  y = H @ x
  return y / y[-1]

def dist_mat(x,y):
  x2 = np.sum(x**2, 0).reshape(-1,1)
  y2 = np.sum(y**2, 0).reshape(1,-1)
  Dist = np.sqrt(x2 + y2 - x.T @ y - y.T @ x)
  return Dist

def dist_mat_old(x,y):
  Dist = np.zeros((x.shape[1], y.shape[1]))
  for i in range(x.shape[1]):
    for j in range(y.shape[1]):
      Dist[i,j] = np.linalg.norm(x[:,i] - y[:,j])
  return Dist

def row_comp_old(x):
  X = x < TAU
  Xstr = ''.join([ '0' if x_ != 0 else '1' for x_ in X ])
  return Xstr

def row_comp(x):
  Xstr = ''.join([ '0' if x_ < TAU else '1' for x_ in x ])
  return Xstr

def distint_locs(xy):
  Dist = dist_mat(xy, xy)
  d_list = [ Dist[i] for i in range(len(Dist)) ]
  D_ = sorted(d_list, key=row_comp)
  Duniq = []
  ids_ = []
  for i in range(len(D_)-1):
    di = 1.0 * (D_[i] < TAU)
    di1 = 1.0 * (D_[i+1] < TAU)
    if not np.isclose(np.linalg.norm(di-di1), 0):
      Duniq.append(D_[i])
      ids_.append(i)
  Duniq.append(D_[-1])
  ids_.append(len(D_)-1)

  if plot > 1:
    D = np.stack(Duniq)
    fig, ax = plt.subplots(nrows=1, ncols=1 + len(threshs))
    ax[0].imshow(D)
    for i in range(len(threshs)):
      ax[i+1].imshow(D < threshs[i])
    plt.show()
    print(xy[:, ids_].shape)
    Dist = dist_mat(xy[:, ids_], xy[:, ids_])
    fig, ax = plt.subplots(nrows=1, ncols=1 + len(threshs))
    ax[0].imshow(Dist)
    for i in range(len(threshs)):
      ax[i+1].imshow(Dist < threshs[i])
    plt.show()
    plt.scatter(xy[0, ids_], xy[1, ids_], c='r', marker='x')
    plt.show()

  return ids_

##############################################################################
print("Time loading: ", time.time() - START_TIME)
##############################################################################

H = {(0,0): np.eye(3)}
for i in range(1,6):
  h = []
  with open('graff/H1to{}p'.format(i+1), 'r') as f:
    for l in f:
      h.append([ float(x) for x in l.split() ])
  h = np.array(h)
  H[0,i] = h
  H[i,0] = np.linalg.inv(h)
  H[i,i] = np.eye(3)
  _, s, _ = np.linalg.svd(h)

for i, j in itertools.combinations(range(6), 2):
  if (i, j) in H:
    continue
  H[i,j] = np.linalg.solve(H[0,i], H[0, j])
  H[j,i] = np.linalg.solve(H[0,j], H[0, i])

if plot > 2:
  for i in range(6):
    fig, ax = plt.subplots(nrows=2, ncols=len(imgs))
    for j in range(6):
      if i == j:
        ax[0, i].imshow(imgs[i])
        ax[1, i].imshow(imgs[i])
      imdst = cv2.warpPerspective(imgs[i], H[i,j], (imgs[j].shape[1], imgs[j].shape[0]))
      ax[0, j].imshow(imgs[j])
      ax[1, j].imshow(imdst)
    plt.show()
  sys.exit()

##############################################################################
print("Time for Homographies: ", time.time() - START_TIME)
##############################################################################

sift = cv2.xfeatures2d.SIFT_create(N_points)
bfmatcher = cv2.BFMatcher()
keypoints = []
descs = []
for i, img in enumerate(imgs[:2]):
  kp, d = sift.detectAndCompute(img, None)
  descs.append(d)
  keypoints.append(kp)

matches = bfmatcher.knnMatch(descs[0], descs[1], k=10)
x_ = []
y_ = []
for i, dd in enumerate(descs):
  kps = keypoints[i]
  h, w, c = imgs[i].shape
  d_ = [ d / np.linalg.norm(d) for d in dd[:N_points] ]
  dkp_ = [ list(d_[i]) + [
              (k.pt[0]-w/2)/w,
              (k.pt[1]-h/2)/h,
              np.log(k.size),
              np.deg2rad(k.angle)
           ]
           for i, k in enumerate(kps[:N_points]) ]
  x_.append([ k.pt[0] for k in kps[:N_points] ])
  y_.append([ k.pt[1] for k in kps[:N_points] ])
print([ len(d) for d in dl ])

if plot > 3:
  myplot = []
  for i in range(len(x_)):
    for j in range(len(x_[0])):
      x0, y0 = int(x_[0][j]), int(y_[0][j])
      x, y, _ = tuple(h_apply(H[0,i], (x0, y0, 1)))
      s = keypoints[0][j].size
      octave, layer, scale = unpackSIFTOctave(keypoints[0][j].octave)
      theta = keypoints[0][j].angle
      dx_, dy_ = np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))
      x_0, y_0, _ = tuple(h_apply(H[0,i], (x0+s*dx_, y0+s*dy_, 1)))
      x_1, y_1, _ = tuple(h_apply(H[0,i], (x0-s*dy_, y0+s*dx_, 1)))
      x_2, y_2, _ = tuple(h_apply(H[0,i], (x0+s*dy_, y0-s*dx_, 1)))
      x_3, y_3, _ = tuple(h_apply(H[0,i], (x0-s*dx_, y0-s*dy_, 1)))
      s_new = np.mean([ np.linalg.norm((x_0 - x_3, y_0 - y_3))/2,
                        np.linalg.norm(((x_1 - x_2, y_1 - y_2)))/2 ])
      angle_new = np.arctan2(-y_0 + y_3, x_0 - x_3)
      angle_new = np.rad2deg(angle_new + 2*np.pi*(angle_new < 0))
      myplot.append((s_new, s, octave, layer, scale, theta, angle_new))
  # Angles
  a_orig = np.array([ x[5] for x in myplot ])
  a_new = np.array([ x[6] for x in myplot ])
  plt.scatter(a_orig, a_new)
  plt.title('Angle differences')
  plt.xlabel('Original')
  plt.ylabel('New')
  plt.show()
  sys.exit()
  # Plot sizes and octaves
  z = np.array([ 0 for x in myplot ])
  sorig = np.array([ x[1] for x in myplot ])
  snew = np.array([ x[0] for x in myplot ])
  c_octaves = np.array([ x[2] for x in myplot ])
  c_layers = np.array([ x[3] for x in myplot ])
  c_scale = np.array([ np.log2(x[4]) for x in myplot ])
  # c_fscale = np.array([ x[4] for x in myplot ])
  # print(np.unique(list(zip(c_octaves, c_fscale)), axis=0))
  # Build inverse octave dict
  print("Octave and log2 scale sum:", np.linalg.norm(c_octaves + c_scale))
  octave_layer = sorted([
      tuple(x) for x in np.unique(list(zip(c_octaves,c_layers)),axis=0) ])
  ol_dict = { k: [] for k in octave_layer }
  o_dict = { k: [] for k in c_octaves }
  false_formula = False
  # Test Octave encoding formula
  for j in range(len(x_[0])):
    s = keypoints[0][j].size
    octave, layer, scale = unpackSIFTOctave(keypoints[0][j].octave)
    ol_dict[(octave, layer)].append(s)
    o_dict[octave].append(s)
    # Check if my octave encoding formula is right
    o, l, _ = unpackSIFTOctave(size_to_octave_desc(s))
    check = (o == octave) and (l == layer)
    if not check:
      false_formula = True
      print("False Octave Encoding Formula for ({}, {}): ({}, {})".format(octave, layer, o, l))
  if not false_formula:
    print("False Octave Encoding Formula")
  # Test Octave Formula
  ff_bad = lambda x: np.floor(np.log(x/1.8) / np.log(1.25))
  gg = lambda x: int(np.floor(np.log2(x / 3.6)))
  ff = lambda x: int(np.floor(3 * np.log2(x / 3.6)) - 3*gg(x)) + 1
  false_formula = False
  for k in sorted(o_dict.keys()):
    check = [ k == gg(x) for x in o_dict[k] ]
    # check = [ (k == int((ff(x) // 3) - 2)) for x in o_dict[k] ]
    if not np.all(check):
      false_formula = True
      inds = [ i for i, c in enumerate(check) if not c ]
      print("False Octave Formula for {}: ".format(k))
      print([ (x, gg(x))
              for ii, x in enumerate(o_dict[k])
              if ii in inds ])
      sys.exit()
  if not false_formula:
    print("True Octave Formula")
  # Test Octave/Layer Formula
  overlap = False
  false_formula = False
  for i, k in enumerate(octave_layer):
    print(k, (np.min(ol_dict[k]), np.max(ol_dict[k]), len(ol_dict[k])))
    check = [ (k[0] == gg(x) and k[1] == ff(x)) for x in ol_dict[k] ]
    if not np.all(check):
      false_formula = True
      inds = [ i for i, c in enumerate(check) if not c ]
      print("False Octave/Layer Formula for {}: ".format(k))
      print([ (x, gg(x), ff(x))
              for ii, x in enumerate(ol_dict[k])
              if ii in inds ])
    if i > 0:
      cur = ol_dict[octave_layer[i]]
      prev = ol_dict[octave_layer[i-1]]
      if np.min(cur) < np.max(prev):
        print("Overlap between {} and {}".format(cur, prev))
        overlap = True
  if not overlap:
    print("No overlap between octaves/layers :D")
  if not false_formula:
    print("True Octave/Layer Formula")
  print("Observation: Looks like this formula is not 1-to-1 "
        "but it mostly works so we're sticking with it")
  # Plot old vs new size with colored octaves
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].scatter(sorig, snew, c=c_octaves)
  ax[0].scatter(z, snew, c=c_octaves)
  im=ax[0].scatter(sorig, z + c_octaves, c=c_octaves)
  fig.colorbar(im, ax=ax[0], orientation='horizontal')
  ax[0].set_xlabel('Original size')
  ax[0].set_ylabel('New size')
  ax[0].set_title('Octaves')
  print(np.unique(c_octaves))
  # Plot old vs new size with colored layers
  ax[1].scatter(sorig, snew, c=c_layers)
  ax[1].scatter(z, snew, c=c_layers)
  im=ax[1].scatter(sorig, z + c_octaves, c=c_layers)
  fig.colorbar(im, ax=ax[1], orientation='horizontal')
  ax[1].set_xlabel('Original size')
  ax[1].set_ylabel('New size')
  ax[1].set_title('Layers')
  print(np.unique(c_layers))
  # Plot old vs new size with colored scales (same as octave)
  # ax[2].scatter(sorig, snew, c=c_scale)
  # ax[2].scatter(z, snew, c=c_scale)
  # im=ax[2].scatter(sorig, z + c_octaves, c=c_scale)
  # fig.colorbar(im, ax=ax[2], orientation='horizontal')
  # ax[2].set_xlabel('Original size')
  # ax[2].set_ylabel('New size')
  # ax[2].set_title('Log Scale')
  # print(np.unique(c_scale))
  plt.show()
  sys.exit()


##############################################################################
print("Time for SIFT Keypoints/Descriptors: ", time.time() - START_TIME)
##############################################################################

xy1 = np.stack([ x_[0], y_[0], np.ones_like(x_[0]) ])
ids1 = distint_locs(xy1)
xy1 = np.stack([ xy1[:, ii] for ii in ids1 ], 1)
sel = xy1[0,:] < np.inf
for j in range(1,6):
  xyj = h_apply(H[0,j], xy1)
  and_lists = [ 
      xyj[0,:] > 0,
      xyj[1,:] > 0,
      xyj[0,:] < imgs[j].shape[1],
      xyj[1,:] < imgs[j].shape[0],
  ]
  # sel += and_lists
  sel = list(functools.reduce(np.logical_and, and_lists, sel))

if plot > 1:
  print("The errors are kind of large actually...")
  fig, ax = plt.subplots(nrows=3, ncols=len(imgs))
  ax[0, 0].imshow(imgs[0])
  im = ax[1, 0].imshow(imgs[0]-imgs[0])
  fig.colorbar(im, ax=ax[1, 0], orientation='vertical')
  ax[2, 0].imshow(imgs[0])
  for j in range(1,6):
    imdst = cv2.warpPerspective(imgs[0], H[0,j], (imgs[j].shape[1], imgs[j].shape[0]))
    ax[0, j].imshow(imgs[j])
    err = np.mean(imgs[j] - imdst, -1)
    mask = 1.0 - 1.0 * (np.mean(imdst, -1) == 0.0)
    im = ax[1, j].imshow(np.minimum(255.0, np.maximum(0, np.abs(mask*err)))/255)
    fig.colorbar(im, ax=ax[1, j], orientation='vertical')
    # ax[2, j].imshow(imdst)
    ax[2, j].imshow(imgs[j])
    xyj = h_apply(H[0,j], xy1)
  not_sel = np.logical_not(sel)
  for j in range(6):
    xyj = h_apply(H[0,j], xy1)
    ax[2, j].scatter(xyj[0,sel], xyj[1,sel], c='g', marker='x')
    ax[2, j].scatter(xyj[0,not_sel], xyj[1,not_sel], c='r', marker='x')
  print(np.sum(sel))
  print(np.sum(not_sel))
  plt.show()

##############################################################################
print("Time for point selection: ", time.time() - START_TIME)
##############################################################################

descs0 = np.stack([ dl[0][ids1[i]] for i in range(len(ids1)) if sel[i] ])
fulldescs0 = np.stack([ dkl[0][ids1[i]] for i in range(len(ids1)) if sel[i] ])
keypoints0 = [ keypoints[0][ids1[i]] for i in range(len(ids1)) if sel[i] ]
Sim = np.dot(descs0,descs0.T)
if plot > 1:
  fig, ax = plt.subplots(nrows=1,ncols=2)
  im = ax[0].imshow(Sim)
  fig.colorbar(im, ax=ax[0], orientation='vertical')
  im = ax[1].imshow(np.sort(Sim)[:,::-1])
  fig.colorbar(im, ax=ax[1], orientation='vertical')
  plt.show()

n = len(Sim)
SimSortList = [ np.sort(s)[::-1] for s in Sim ]
vec = (np.arange(n)[::-1])**(2)
msim2_idxs = sorted(np.arange(n), key=lambda x: SimSortList[x] @ vec)

##############################################################################
print("Time for descriptor selection: ", time.time() - START_TIME)
##############################################################################

xy_ = xy1[:, sel]
xy = [ xy_[:, msim2_idxs[:N_data]] ]
descs = [ descs0[msim2_idxs[:N_data], :] ]
keypoints = [ [ keypoints0[i] for i in msim2_idxs[:N_data] ] ]
for i in range(1,6):
  xy.append(h_apply(H[0,i], xy[0]))
  kpts = []
  for j in range(xy[0].shape[1]):
    x0, y0 = xy[0][0,j], xy[0][1,j]
    x, y = xy[i][0,j], xy[i][1,j]
    s = keypoints0[j].size
    octave, layer, scale = unpackSIFTOctave(keypoints0[j].octave)
    theta = keypoints0[j].angle
    dx_, dy_ = np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))
    x_0, y_0, _ = tuple(h_apply(H[0,i], (x0+s*dx_, y0+s*dy_, 1)))
    x_1, y_1, _ = tuple(h_apply(H[0,i], (x0-s*dy_, y0+s*dx_, 1)))
    x_2, y_2, _ = tuple(h_apply(H[0,i], (x0+s*dy_, y0-s*dx_, 1)))
    x_3, y_3, _ = tuple(h_apply(H[0,i], (x0-s*dx_, y0-s*dy_, 1)))
    s_new = np.mean([ np.linalg.norm((x_0 - x_3, y_0 - y_3))/2,
                      np.linalg.norm(((x_1 - x_2, y_1 - y_2)))/2 ])
    octv = size_to_octave_desc(s_new)
    angle_new = np.arctan2(-y_0 + y_3, x_0 - x_3)
    angle_new = np.rad2deg(angle_new + 2*np.pi*(angle_new < 0))
    kpts.append(cv2.KeyPoint(x=x, y=y,
                             _size=s_new,
                             _angle=angle_new,
                             _response=keypoints0[j].response,
                             _octave=octv,
                             _class_id=keypoints0[j].class_id))
  keypoints.append(kpts)

descriptors = []
Descs = []
FullDesc = []
for kpts, img in zip(keypoints, imgs):
  h, w, c = imgs[i].shape
  _, descs = sift.compute(img, kpts)
  d_ = [ (d / np.linalg.norm(d)) for d in descs ]
  descriptors.append(d_)
  dkp_ = [ np.array(list(d_[i]) + [
               (k.pt[0]-w/2)/w, (k.pt[1]-h/2)/h,
               np.log(k.size), np.deg2rad(k.angle)
           ])
           for i, k in enumerate(kpts) ]
  Descs.extend(descriptors[-1])
  FullDescs.append(dkp_)

##############################################################################
print("Time for feature adaptation: ", time.time() - START_TIME)
##############################################################################
# k = 5
# n = N_data
# v = N_imgs
# 
# ndescs = np.stack(Descs)
# init_emb = np.stack(FullDescs)
# # Build Graph
# Dinit = np.dot(ndescs, ndescs.T)
# Dmin = Dinit.min()
# Dmax = Dinit.max()
# D = (Dinit - Dmin)/(Dmax-Dmin) # Rescaling
# A_ = np.zeros_like(D)
# for i in range(v):
#   for j in range(v):
#     if i == j:
#       continue
#     # Perhaps not the most efficient... oh well
#     Asub = np.copy(D[n*i:n*(i+1),n*j:n*(j+1)])
#     for u in range(n):
#       Asub[u,Asub[u].argsort()[:-k]] = 0
#     A_[n*i:n*(i+1),n*j:n*(j+1)] = Asub
# adj_mat = np.maximum(A_, A_.T)
# # Build dataset ground truth
# true_emb = np.concatenate(perm_mats,axis=0)
# gt_adj_mat = np.dot(true_emb, true_emb.T)
# matches_ = np.concatenate(perms)
# # Nominal rotations and translations to make sure it works
# rots = np.stack([ np.eye(3) for i in tupl ], axis=0)
# trans = np.stack([ np.zeros(3) for i in tupl ], axis=0)
# 
# # Build actual graph
# G_nx = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
# node_attrs = { i : init_emb[i].astype(np.float32)
#                for i in range(len(G_nx)) }
# edges_attrs = { (i, j) : np.array([ adj_mat[i,j] ]).astype(np.float32)
#                 for (i,j) in G_nx.edges }
# nx.set_node_attributes(G_nx, node_attrs, 'features')
# nx.set_edge_attributes(G_nx, edges_attrs, 'features')
# # Make to dictionary, with additional graph attributes
# G = utils_np.networkx_to_data_dict(G_nx)
# G['globals'] = np.array([0,0])
# G['adj_mat'] = graph_dataset.np_dense_to_sparse(adj_mat)
# G['true_adj_mat'] = graph_dataset.np_dense_to_sparse(gt_adj_mat)
# G['true_match'] = matches_
# G['rots'] = rots
# G['trans'] = trans
# 



if plot > 1:
  VizDescs = [ [] for i in range(len(keypoints[0])) ]
  for kpts, descs in zip(keypoints, descriptors):
    for i in range(len(kpts)):
      VizDescs[i].append(descs[i])
  VizDescs = np.stack(sum(VizDescs, []))
  # plt.imshow(Descs @ Descs.T)
  # plt.show()
  VizSim = VizDescs @ VizDescs.T
  filt = np.ones((N_imgs, N_imgs))/N_imgs**2
  VizSimConv = sig.convolve2d(VizSim, filt)[N_imgs-1::N_imgs, N_imgs-1::N_imgs]
  plt.imshow(VizSimConv)
  plt.show()
  print(VizSimConv.shape)
  print(np.sum(VizSimConv * np.eye(N_data)) / N_data)
  print(np.sum(VizSimConv * (1 - np.eye(N_data))) / np.sum(1 - np.eye(N_data)))
  for i in range(N_data):
    v = VizSimConv[i,:]
    vi = v[i]
    velse = sorted([ v[j] for j in range(N_data) if j != i ], key = lambda x: -x)
    plt.plot([vi] + velse)
  plt.show()

if plot > 1:
  for kpts, img in zip(keypoints, imgs):
    print("Keypoints")
    print('\n'.join([ 
      "x={x:7.3f} y={y:7.3f} s={s:7.3f} o={o}".format(x=k.pt[0], y=k.pt[1], s=k.size, o=unpackSIFTOctave(k.octave))
      for k in kpts
    ]))
    kpts_, descs = sift.compute(img, kpts)
    for k1, k2 in zip(kpts, kpts_):
      dfs = {
        "xd": k1.pt[0]-k2.pt[0], "yd": k1.pt[1]-k2.pt[1],
        "sd": k1.size-k2.size,
        "ad": k1.angle-k2.angle,
        "od": k1.octave-k2.octave,
      }
      if np.any([ v != 0 for v in dfs.values() ]):
        print(("Pts diff: ({xd:6.3f}, {yd:6.3f}), "
               "Size diff: {sd:6.3f}, "
               "Angle diff: {ad:6.3f}, "
               "Octave diff: {od}").format(**dfs))

##############################################################################
print("Time for saving out features: ", time.time() - START_TIME)
##############################################################################

if plot > 1:
  myplot = []
  for i in range(1,6):
    for j in range(xy[0].shape[1]):
      x0, y0 = int(xy[0][0,j]), int(xy[0][1,j])
      x, y = int(xy[i][0,j]), int(xy[i][1,j])
      s = keypoints0[j].size
      octave, layer, scale = unpackSIFTOctave(keypoints0[j].octave)
      theta = keypoints0[j].angle
      dx_, dy_ = np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))
      x_0, y_0, _ = tuple(h_apply(H[0,i], (x0+s*dx_, y0+s*dy_, 1)))
      x_1, y_1, _ = tuple(h_apply(H[0,i], (x0-s*dy_, y0+s*dx_, 1)))
      x_2, y_2, _ = tuple(h_apply(H[0,i], (x0+s*dy_, y0-s*dx_, 1)))
      x_3, y_3, _ = tuple(h_apply(H[0,i], (x0-s*dx_, y0-s*dy_, 1)))
      s_new = np.mean([ np.linalg.norm((x_0 - x_3, y_0 - y_3))/2,
                        np.linalg.norm(((x_1 - x_2, y_1 - y_2)))/2 ])
      myplot.append((s_new, s, octave, layer, scale))
      # print('sizes: ', s_new, s, octave, layer, scale)

  c_octaves = np.array(list(range(-1,4)))
  c_layers = np.array(list(range(1,4)))
  octave_layer = [ x for x in itertools.product(c_octaves,c_layers) ]
  ol_dict = { k: [] for k in octave_layer }
  false_formula = False
  for j in range(xy[0].shape[1]):
    s = keypoints0[j].size
    octave, layer, scale = unpackSIFTOctave(keypoints[0][j].octave)
    ol_dict[(octave, layer)].append(s)
    # Check if my octave encoding formula is right
    o, l, _ = unpackSIFTOctave(size_to_octave_desc(s))
    check = (o == octave) and (l == layer)
    if not check:
      false_formula = True
      print("False Octave Encoding Formula for ({}, {}): ({}, {})".format(octave, layer, o, l))
  if not false_formula:
    print("True Octave Encoding Formula")

  gg = lambda x: int(np.floor(np.log2(x / 3.6)))
  ff = lambda x: int(np.floor(3 * np.log2(x / 3.6)) - 3*gg(x)) + 1
  overlap = False
  false_formula = False
  for i, k in enumerate(octave_layer):
    check = [True] + [ (k[0] == gg(x) and k[1] == ff(x)) for x in ol_dict[k] ]
    if not np.all(check):
      false_formula = True
      inds = [ i for i, c in enumerate(check) if not c ]
      print("False Octave/Layer Formula for {}: ".format(k))
      print([ (x, gg(x), ff(x))
              for ii, x in enumerate(ol_dict[k])
              if ii in inds ])
  if not false_formula:
    print("True Octave/Layer Formula")

if plot > 1:
  # fig, ax = plt.subplots(nrows=1, ncols=N_imgs)
  fig, ax_ = plt.subplots(nrows=2, ncols=3)
  ax = ax_.flatten()
  sc_size = 3
  line_sz = 4
  color_list = [ (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (0, 255, 255), (255, 0, 255),
                 (127, 255, 0), (0, 127, 255), (127, 0, 255),
                 (255, 127, 0), (0, 255, 127), (255, 0, 127) ]
  dispimg = np.copy(imgs[0])
  for j in range(xy[0].shape[1]):
    x, y = int(xy[0][0,j]), int(xy[0][1,j])
    s = keypoints0[j].size
    theta = keypoints0[j].angle
    dx_, dy_ = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    dx, dy, ddx, ddy = int(s*dx_), -int(s*dy_), int(0.5*s*dx_), -int(0.5*s*dy_)
    # print("Direction", theta, dx, dy)
    # cv2.circle(dispimg, (x, y), int(np.ceil(s)), c, thickness=line_sz)
    # Plot things
    c = color_list[j%len(color_list)]
    cv2.line(dispimg, (x, y), (x + dx, y + dy), c, thickness=line_sz) 
    cv2.line(dispimg, (x, y), (x - dy, y + dx), c, thickness=line_sz) 
    cv2.line(dispimg, (x, y), (x + dy, y - dx), c, thickness=line_sz) 
    cv2.line(dispimg, (x, y), (x - dx, y - dy), c, thickness=line_sz) 
    cv2.line(dispimg, (x, y), (x + ddx, y + ddy), (0, 0, 0), thickness=line_sz//2) 

  ax[0].imshow(dispimg)
  # ax[0].scatter(xy[0][0,:], xy[0][1,:], s=2*sc_size, c='w', marker='o')
  # ax[0].scatter(xy[0][0,:], xy[0][1,:], s=sc_size, c='k', marker='*')
  for i in range(1,6):
    dispimg = np.copy(imgs[i])
    kps = []
    for j in range(xy[0].shape[1]):
      x0, y0 = int(xy[0][0,j]), int(xy[0][1,j])
      x, y = int(xy[i][0,j]), int(xy[i][1,j])
      s = keypoints0[j].size
      theta = keypoints0[j].angle
      dx_, dy_ = np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))
      x_0, y_0, _ = tuple(h_apply(H[0,i], (x0+s*dx_, y0+s*dy_, 1)))
      x_1, y_1, _ = tuple(h_apply(H[0,i], (x0-s*dy_, y0+s*dx_, 1)))
      x_2, y_2, _ = tuple(h_apply(H[0,i], (x0+s*dy_, y0-s*dx_, 1)))
      x_3, y_3, _ = tuple(h_apply(H[0,i], (x0-s*dx_, y0-s*dy_, 1)))
      mytheta = np.arccos((x_0-x_3) / np.sqrt((x_0-x_3)**2+(y_0-y_3)**2))
      dxt, dyt = np.cos(mytheta), -np.sin(mytheta)
      xt, yt = int(x + s*dxt/2), int(y + s*dyt/2)

      # Plot things
      c = color_list[j%len(color_list)]
      cv2.line(dispimg, (x, y), (int(x_0), int(y_0)), c, thickness=line_sz)
      cv2.line(dispimg, (x, y), (int(x_1), int(y_1)), c, thickness=line_sz)
      cv2.line(dispimg, (x, y), (int(x_2), int(y_2)), c, thickness=line_sz)
      cv2.line(dispimg, (x, y), (int(x_3), int(y_3)), c, thickness=line_sz)
      cv2.line(dispimg, (x, y), (xt, yt), (0, 0, 0), thickness=line_sz//2)

    ax[i].imshow(dispimg)
    # ax[i].scatter(xy[i][0,:], xy[i][1,:], s=2*sc_size, c='w', marker='o')
    # ax[i].scatter(xy[i][0,:], xy[i][1,:], s=sc_size, c='k', marker='*')

  plt.show()

if plot > 1:
  x_ = xy[0, msim2_idxs]
  y_ = xy[1, msim2_idxs]
  # Plot selected x, y positions of selected features
  plt.imshow(imgs[0])
  plt.scatter(x_[:N_data], y_[:N_data], c='w', s=81, marker='*', facecolors='none')
  plt.scatter(x_[N_data:2*N_data], y_[N_data:2*N_data], c='g', marker='+')
  plt.scatter(x_[2*N_data:], y_[2*N_data:], c='r', marker='x')
  plt.show()
  # Plot avg. similiarity score of Simarities of selected labels
  plt.plot(meanSim, label='Sim')
  plt.plot(meanSim[msim2_idxs], label='SimSorted')
  plt.legend()
  plt.show()
  # Plot sorted similiarity matrix
  fig, ax = plt.subplots(nrows=2,ncols=2)
  Perm = np.eye(n)[msim_idxs]
  PSim = Perm @ Sim @ Perm.T
  im = ax[0,0].imshow(PSim)
  fig.colorbar(im, ax=ax[0], orientation='vertical')
  im = ax[1,0].imshow(np.sort(PSim)[:,::-1])
  fig.colorbar(im, ax=ax[1], orientation='vertical')
  Perm = np.eye(n)[msim2_idxs]
  PSim = Perm @ Sim @ Perm.T
  im = ax[0,1].imshow(PSim)
  fig.colorbar(im, ax=ax[0], orientation='vertical')
  im = ax[1,1].imshow(np.sort(PSim)[:,::-1])
  fig.colorbar(im, ax=ax[1], orientation='vertical')
  plt.show()

##############################################################################
print("Time to Finish: ", time.time() - START_TIME)
##############################################################################

sys.exit()





