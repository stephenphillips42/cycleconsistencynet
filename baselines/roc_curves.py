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


def compute_thresh_errs(output, adjmat, N_cutoffs=2048):
  a = adjmat.astype(np.int32)
  M_T = np.sum(a)
  M_F = a.size - M_T

  TP_, TN_, FP_, FN_ = (np.zeros(N_cutoffs).astype(np.int32) for i in range(4))
  TP_[ 0], TN_[ 0], FP_[ 0], FN_[ 0] =   0, M_F,   0, M_T
  TP_[-1], TN_[-1], FP_[-1], FN_[-1] = M_T,   0, M_F,   0
  for idx in range(1,N_cutoffs-1):
    i = (N_cutoffs - 1) - idx
    thresh = (1.0*i) / (N_cutoffs-1)
    o = (output > thresh).astype(np.int32)
    TP, TN, FP, FN = calc_classifications(o,a)
    TP_[idx], TN_[idx], FP_[idx], FN_[idx] = TP, TN, FP, FN

  return (TP_, TN_, FP_, FN_)

def roc_lines(TP, TN, FP, FN):
  FPR = np.maximum(1e-8, FP) / np.maximum(1e-8, FP + TN)
  TPR = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FN)
  roc = np.stack((FPR, TPR), axis=-1)
  return roc

def compute_roc(TP, TN, FP, FN):
  FPR = np.maximum(1e-8, FP) / np.maximum(1e-8, FP + TN)
  TPR = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FN)
  # sidx = np.argsort(FPR)
  # FPR = FPR[sidx]
  # TPR = TPR[sidx]
  return np.abs(np.trapz(TPR, x=FPR))

def precision_recall_lines(TP, TN, FP, FN):
  precision = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FP)
  recall = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FN)
  p_r = np.stack((precision, recall), axis=-1)
  return p_r

def compute_precision_recall(TP, TN, FP, FN):
  precision = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FP)
  recall    = np.maximum(1e-8, TP) / np.maximum(1e-8, TP + FN)
  return np.abs(np.trapz(recall, x=precision))

def plot_values(TP, TN, FP, FN, show=True):
  roc = roc_lines(TP, TN, FP, FN)
  p_r = precision_recall_lines(TP, TN, FP, FN)
  plt.plot(roc[:,0], roc[:,1])
  plt.plot(p_r[:,0], p_r[:,1])
  if show:
    plt.scatter([0,0,1,1], [0,1,0,1])
    plt.show()

def main():
  # Constants
  N_ = 2048
  # Variables
  roc_areas_ = []
  p_r_areas_ = []
  roc_lines_ = []
  p_r_lines_ = []
  # Tensorflow output file
  fname = 'GCN12Layer.npz'
  with open(fname, 'rb') as f:
    ld = dict(np.load(fname))
  temb = ld['trueemb']
  outemb = ld['out']
  # MATLAB Output Files
  opt_names = []
  opt_adjmats = []
  opt_outputs = []
  AdjmatFiles = sorted(glob.glob('*Adjmats.npy'))
  OutputFiles = sorted(glob.glob('*Outputs.npy'))
  for adjmat_name, output_name in zip(AdjmatFiles, OutputFiles):
    opt_names.append(adjmat_name[:-len('-Adjmats.npy')])
    with open(adjmat_name, 'rb') as f:
      a = np.load(f)
      opt_adjmats.append(a)
    with open(output_name, 'rb') as f:
      o = np.load(f)
      opt_outputs.append(o)

  for i in range(20):
    # MATLAB Outputs
    fig_roc, ax_roc = plt.subplots()
    fig_p_r, ax_p_r = plt.subplots()
    ax_roc.scatter([0,0,1,1],[0,1,0,1])
    ax_p_r.scatter([0,0,1,1],[0,1,0,1])
    adjmat = opt_adjmats[0][i] # They are all the same
    for k in range(len(opt_names)):
      # Compute things
      output = opt_outputs[k][i]
      TP, TN, FP, FN = compute_thresh_errs(output, adjmat, N_cutoffs=N_)
      # Get areas
      roc_areas_.append(compute_roc(TP, TN, FP, FN))
      p_r_areas_.append(compute_precision_recall(TP, TN, FP, FN))
      print('{0:04d} {1:<20}: ROC: {2:.03e}, P-R: {3:.03e}'.format(k, opt_names[k],
                                                                   roc_areas_[-1],
                                                                   p_r_areas_[-1]))
      # PLot lines
      roc = roc_lines(TP, TN, FP, FN)
      p_r = precision_recall_lines(TP, TN, FP, FN)
      ax_roc.plot(roc[:,0], roc[:,1], label='{} ROC ({:.03e})'.format(opt_names[k], roc_areas_[-1]))
      ax_p_r.plot(p_r[:,0], p_r[:,1], label='{} P-R ({:.03e})'.format(opt_names[k], p_r_areas_[-1]))
      roc_lines_.append(roc)
      p_r_lines_.append(p_r)
    # Tensorflow Outputs  
    output = np.abs(np.dot(outemb[i], outemb[i].T))
    TP, TN, FP, FN = compute_thresh_errs(output, adjmat, N_cutoffs=N_)
    roc_areas_.append(compute_roc(TP, TN, FP, FN))
    p_r_areas_.append(compute_precision_recall(TP, TN, FP, FN))
    print('{0:04d} {1:<20}: ROC: {2:.03e}, P-R: {3:.03e}'.format(k, 'GCN',
                                                                 roc_areas_[-1],
                                                                 p_r_areas_[-1]))
    # Tensorflow Plot lines
    roc = roc_lines(TP, TN, FP, FN)
    p_r = precision_recall_lines(TP, TN, FP, FN)
    ax_roc.plot(roc[:,0], roc[:,1], label='{} ROC ({:.03e})'.format('GCN', roc_areas_[-1]))
    ax_p_r.plot(p_r[:,0], p_r[:,1], label='{} P-R ({:.03e})'.format('GCN', p_r_areas_[-1]))
    roc_lines_.append(roc)
    p_r_lines_.append(p_r)

    # Finish plots
    ax_roc.set_xlabel('False Positive Rate')
    ax_p_r.set_xlabel('Precision')
    ax_roc.set_ylabel('True Positive Rate')
    ax_p_r.set_ylabel('Recall')
    ax_roc.set_title('ROC Curves')
    ax_p_r.set_title('Precision Recall Curves')
    ax_roc.legend()
    ax_p_r.legend()
    fig_roc.savefig('ROC-Curves-{:04d}.png'.format(i))
    fig_p_r.savefig('P-R-Curves-{:04d}.png'.format(i))
    plt.close(fig_roc)
    plt.close(fig_p_r)




if __name__ == '__main__':
  main()

