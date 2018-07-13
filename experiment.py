import os
import sys
import glob
import numpy as np
import types
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tqdm
import yaml

import myutils
import options
import model

def axes3d(nrows=1, ncols=1):
  fig = plt.figure()
  axes = [ fig.add_subplot(nrows, ncols, i+1, projection='3d') for i in range(nrows*ncols) ]
  return fig, axes

def npload(fdir,idx):
  return dict(np.load("{}/np_test-{:04d}.npz".format(fdir,idx)))

def get_sorted(labels):
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  return slabels, sorted_idxs

def plot_hist(save_dir, sim_mats, names, true_sim):
  fig, ax = plt.subplots(nrows=1, ncols=2)
  diags = [ np.reshape(v[true_sim==1],-1) for v in sim_mats ]
  off_diags = [ np.reshape(v[true_sim==0],-1) for v in sim_mats ]
  ax[0].hist(diags, bins=20, density=True, label=names)
  ax[0].legend()
  ax[0].set_title('Diagonal Similarity Rate')
  ax[1].hist(off_diags, bins=20, density=True, label=names)
  ax[1].set_title('Off Diagonal Similarity Rate')
  ax[1].legend()
  plt.savefig(os.path.join(save_dir, 'hist.png'))

def plot_baseline(save_dir, emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  rsim = np.abs(np.dot(srand, srand.T))
  print('Sorted labels')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(slabels)
  im1 = ax1.imshow(srand)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.savefig(os.path.join(save_dir, 'labels_sort.png'))
  print('Sorted similarites')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(lsim)
  im1 = ax1.imshow(rsim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.savefig(os.path.join(save_dir, 'sim_sort.png'))

def plot_index(save_dir, emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(soutput)
  im1 = ax1.imshow(osim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.savefig(os.path.join(save_dir, 'output.png'))
  diag = np.reshape(osim[lsim==1],-1)
  off_diag = np.reshape(osim[lsim==0],-1)
  baseline_diag = np.reshape(rsim[lsim==1],-1)
  baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  ax0.hist([ diag, baseline_diag ], bins=20, density=True,
           label=[ 'diag', 'baseline_diag' ])
  ax0.legend()
  ax0.set_title('Diagonal Similarity Rate')
  ax1.hist([ off_diag, baseline_off_diag ], bins=20, density=True,
           label=[ 'off_diag', 'baseline_off_diag' ])
  ax1.set_title('Off Diagonal Similarity Rate')
  ax1.legend()
  plt.savefig(os.path.join(save_dir, 'sim_hist.png'))

def plot_index_unsorted(save_dir, emb_init, emb_gt, emb_out, adjmat):
  labels = emb_gt
  rand = myutils.dim_normalize(emb_init)
  lsim = np.abs(np.dot(labels, labels.T))
  osim = np.abs(np.dot(emb_out, emb_out.T))
  rsim = np.abs(np.dot(rand, rand.T))
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
  im0 = ax0.imshow(output)
  im1 = ax1.imshow(osim)
  im2 = ax2.imshow(adjmat + np.eye(adjmat.shape[0]))
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.savefig(os.path.join(save_dir, 'unsorted_output.png'))
  # diag = np.reshape(osim[lsim==1],-1)
  # off_diag = np.reshape(osim[lsim==0],-1)
  # baseline_diag = np.reshape(rsim[lsim==1],-1)
  # baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  # ax0.hist([ diag, baseline_diag ], bins=20, density=True,
  #          label=[ 'diag', 'baseline_diag' ])
  # ax0.legend()
  # ax0.set_title('Diagonal Similarity Rate')
  # ax1.hist([ off_diag, baseline_off_diag ], bins=20, density=True,
  #          label=[ 'off_diag', 'baseline_off_diag' ])
  # ax1.set_title('Off Diagonal Similarity Rate')
  # ax1.legend()
  # plt.savefig(os.path.join(save_dir, 'sim_hist_unsorted.png'))

def plot_random(save_dir, emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  plots = [ rsim, osim, osim**9 ]
  names = [ 'rsim', 'osim', 'osim**9' ]
  fig, ax = plt.subplots(nrows=1, ncols=len(plots))
  diags = [ np.reshape(v[lsim==1],-1) for v in plots ]
  off_diags = [ np.reshape(v[lsim==0],-1) for v in plots ]
  for i in range(len(plots)):
    ax[i].hist([ diags[i], off_diags[i] ], bins=20, density=True, label=['diag', 'off_diag'])
    ax[i].legend()
    ax[i].set_title(names[i])
    print(np.min(diags[i]))
    print(np.max(off_diags[i]))
    print('--')
  plt.savefig(os.path.join(save_dir, 'random.png'))

def get_stats(emb_init, emb_gt, emb_out):
  slabels, sorted_idxs = get_sorted(emb_gt)
  soutput = emb_out[sorted_idxs]
  srand = myutils.dim_normalize(emb_init[sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  diag = np.reshape(osim[lsim==1],-1)
  off_diag = np.reshape(osim[lsim==0],-1)
  baseline_diag = np.reshape(rsim[lsim==1],-1)
  baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  return (np.mean(diag), np.std(diag), \
          np.mean(off_diag), np.std(off_diag), \
          np.mean(baseline_diag), np.std(baseline_diag), \
          np.mean(baseline_off_diag), np.std(baseline_off_diag))


if __name__ == "__main__":
  # Build options
  opts = options.get_opts()
  # Run experiment
  ld = np.load(opts.debug_data_path)
  emb_init = ld['input']
  emb_gt = ld['gt']
  emb_out = ld['output']
  adjmat = ld['adjmat']
  n = len(emb_gt)
  if opts.debug_plot == 'none':
    stats = np.zeros((n,8))
    if opts.verbose:
      for i in tqdm.tqdm(range(n)):
        stats[i] = get_stats(emb_init[i], emb_gt[i], emb_out[i])
    else:
      for i in range(n):
        stats[i] = get_stats(emb_init[i], emb_gt[i], emb_out[i])
    meanstats = np.mean(stats,0)
    print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Off Diag: {:.2e} +/- {:.2e}".format(*list(meanstats)))
    sys.exit()
  if opts.debug_index > n:
    print("ERROR: debug_index out of bounds")
    sys.exit()
  i = opts.debug_index
  if opts.debug_plot == 'plot':
    plot_index(opts.debug_log_dir, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.debug_plot == 'unsorted':
    plot_index_unsorted(opts.debug_log_dir, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.debug_plot == 'baseline':
    plot_baseline(opts.debug_log_dir, emb_init[i], emb_gt[i], emb_out[i])
  elif opts.debug_plot == 'random':
    plot_random(opts.debug_log_dir, emb_init[i], emb_gt[i], emb_out[i])



