# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt 
import time

def calc_classifications(o, a):
  TP = np.sum(o*a)
  TN = np.sum((1-o)*(1-a))
  FP = np.sum(o*(1-a))
  FN = np.sum((1-o)*a)
  return TP, TN, FP, FN

fname = '/home/stephen/Documents/Research/cycleconsistencynet/test_output.npz'
ld = dict(np.load(fname))
temb = ld['trueemb']
outemb = ld['out']
N_cutoffs = 100
adjmat = np.dot(temb, temb.T)
# plt.imshow(adjmat)
# plt.colorbar()
# plt.show()

output = np.abs(np.dot(outemb, outemb.T))
# plt.imshow(output)
# plt.colorbar()
# plt.show()

def compute_thresh_errs(output, adjmat, N_cutoffs=2048):
  a = adjmat.astype(np.int32)
  M_T = np.sum(a)
  M_F = a.size - M_T

  vals = np.zeros((N_cutoffs, 4)).astype(np.int32)
  for idx, i in enumerate(range(N_cutoffs-1,-1,-1)):
    thresh = (1.0*i) / (N_cutoffs-1)
    o = (output > thresh).astype(np.int32)
    TP, TN, FP, FN = calc_classifications(o,a)
    vals[idx,:] = (TP, TN, FP, FN)

  return (vals[:,0], vals[:,1], vals[:,2], vals[:,3])

def roc_lines(TP, TN, FP, FN):
  roc = np.stack((FP / np.maximum(1e-8, FP + TN),
                  TP / np.maximum(1e-8, TP + FN)), axis=-1)
  return roc

def compute_roc(TP, TN, FP, FN):
  FPR = FP / np.maximum(1e-8, FP + TN)
  TPR = TP / np.maximum(1e-8, TP + FN)
  return np.abs(np.trapz(TPR, x=FPR))

def precision_recall_lines(TP, TN, FP, FN):
  p_r = np.stack((TP / np.maximum(1e-8, TP + FP),
                  TP / np.maximum(1e-8, TP + FN)), axis=-1)
  return p_r

def compute_precision_recall(TP, TN, FP, FN):
  precision = TP / np.maximum(1e-8, TP + FP)
  recall    = TP / np.maximum(1e-8, TP + FN)
  return np.abs(np.trapz(recall, x=precision))

TP, TN, FP, FN = compute_thresh_errs(output, adjmat, N_cutoffs=N_cutoffs)
roc_area = compute_roc(TP, TN, FP, FN)
pr_area = compute_precision_recall(TP, TN, FP, FN)
print("             ROC Area: {}".format(roc_area))
print("Precision Recall Area: {}".format(pr_area))


roc = roc_lines(TP, TN, FP, FN)
p_r = precision_recall_lines(TP, TN, FP, FN)
plt.plot(roc[:,0], roc[:,1])
plt.plot(p_r[:,0], p_r[:,1])
plt.scatter([0,0,1,1], [0,1,0,1])
plt.show()




