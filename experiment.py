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

def get_sorted(sample):
  labels = sample['TrueEmbedding']
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  return slabels, sorted_idxs

def plot_baseline(opts, network, index):
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  slabels, sorted_idxs = get_sorted(sample)
  srand = myutils.dim_normalize(sample['InitEmbeddings'][sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  rsim = np.abs(np.dot(srand, srand.T))
  print('Sorted labels')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(slabels)
  im1 = ax1.imshow(srand)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.show()
  print('Sorted similarites')
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(lsim)
  im1 = ax1.imshow(rsim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.show()

def plot_index(opts, network, index):
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  output = network.apply_np(sample)
  slabels, sorted_idxs = get_sorted(sample)
  soutput = output[sorted_idxs]
  srand = myutils.dim_normalize(sample['InitEmbeddings'][sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  im0 = ax0.imshow(soutput)
  im1 = ax1.imshow(osim)
  fig.colorbar(im0, ax=ax0)
  fig.colorbar(im1, ax=ax1)
  plt.show()
  diag = np.reshape(osim[lsim==1],-1)
  off_diag = np.reshape(osim[lsim==0],-1)
  baseline_diag = np.reshape(rsim[lsim==1],-1)
  baseline_off_diag = np.reshape(rsim[lsim==0],-1)
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  ax0.hist([ diag, baseline_diag ], bins=20, normed=1,
           label=[ 'diag', 'baseline_diag' ])
  ax0.legend()
  ax0.set_title('Diagonal Similarity Rate')
  ax1.hist([ off_diag, baseline_off_diag ], bins=20, normed=1,
           label=[ 'off_diag', 'baseline_off_diag' ])
  ax1.set_title('Off Diagonal Similarity Rate')
  ax1.legend()
  plt.show()

def get_stats(opts, network, index):
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  output = network.apply_np(sample)
  slabels, sorted_idxs = get_sorted(sample)
  soutput = output[sorted_idxs]
  srand = myutils.dim_normalize(sample['InitEmbeddings'][sorted_idxs])
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
  debug_opts = options.get_opts()
  with open(os.path.join(debug_opts.debug_log_dir, 'options.yaml'), 'r') as yml:
    opts = types.SimpleNamespace(**yaml.load(yml))
  opts.data_dir = os.path.join(debug_opts.debug_data_dir, opts.dataset)
  opts.debug_plot = debug_opts.debug_plot
  network = model.get_network(opts, opts.arch)
  network.load_np(os.path.join(debug_opts.debug_log_dir, 'np_weights_001'))
  if opts.debug_plot == 'none':
    n = opts.dataset_params.sizes['test']
    stats = np.zeros((n,8))
    if opts.verbose:
      for i in range(n):
        stats[i] = get_stats(opts, network, i)
    else:
      for i in tqdm.tqdm(range(n)):
        stats[i] = get_stats(opts, network, i)
    meanstats = np.mean(stats,0)
    print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Off Diag: {:.2e} +/- {:.2e}".format(*list(meanstats)))
  elif opts.debug_plot == 'plot':
    plot_index(opts, network, opts.debug_index)
  elif opts.debug_plot == 'baseline':
    plot_baseline(opts, network, opts.debug_index)



