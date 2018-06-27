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

def get_stats():

def experiment(opts, network, index):
  # Load sample
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  output = network.apply_np(sample)
  # Sort by ground truth for better visualization
  labels = sample['TrueEmbedding']
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  soutput = output[sorted_idxs]
  rand = myutils.dim_normalize(sample['InitEmbeddings'][sorted_idxs])
  # rand = myutils.dim_normalize(np.random.randn(*shape))
  # Plot setup
  shape = slabels.shape
  t = list(range(shape[0]))
  xx0, yy0 = np.meshgrid(t, t)
  t = list(range(shape[1]))
  xx1, yy1 = np.meshgrid(t, t)
  lcorr = np.abs(np.dot(slabels.T, slabels))
  ocorr = np.abs(np.dot(soutput.T, soutput))
  rcorr = np.abs(np.dot(rand.T, rand))
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(rand, rand.T))
  if opts.debug_plot == 'plot':
    # Plotting images
    print('Adjacency and Laplacian')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(sample['AdjMat'])
    im1 = ax1.imshow(sample['Laplacian'])
    im2 = ax2.imshow(slabels)
    fig.colorbar(im0, ax=ax0)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()
    print('Sorted labels')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(slabels)
    im1 = ax1.imshow(soutput)
    im2 = ax2.imshow(rand)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()
    print('Sorted correlations')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(lcorr)
    im1 = ax1.imshow(ocorr)
    im2 = ax2.imshow(rcorr)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()
    print('Sorted similarites')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(lsim)
    im1 = ax1.imshow(osim)
    im2 = ax2.imshow(rsim)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()

  # Printing out parts
  npts = len(np.unique(idxs))
  mean_sims = np.zeros((npts,npts))

  diag = []
  off_diag = []
  baseline_diag = []
  baseline_off_diag = []
  for i in range(npts):
    for j in range(npts):
      simsij = np.abs(np.dot(output[idxs == i], output[idxs == j].T)).reshape(-1).tolist()
      baselineij = np.abs(np.dot(rand[idxs == i], rand[idxs == j].T)).reshape(-1).tolist()
      mean_sims[i,j] = np.mean(np.dot(output[idxs == i], output[idxs == j].T))
      # if i == j:
      #   for k in range(len(simsij)):
      #     diag.append(np.arccos(np.minimum(1, simsij[k])))
      # else:
      #   for k in range(len(simsij)):
      #     off_diag.append(np.arccos(np.maximum(-1,np.minimum(1,simsij[k]))))
      if i == j:
          diag.extend(simsij)
          baseline_diag.extend(baselineij)
      else:
          off_diag.extend(simsij)
          baseline_off_diag.extend(baselineij)
  stats = (np.mean(diag), np.std(diag), \
           np.mean(off_diag), np.std(off_diag), \
           np.mean(baseline_diag), np.std(baseline_diag), \
           np.mean(baseline_off_diag), np.std(baseline_off_diag))
  if opts.verbose:
    print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Off Diag: {:.2e} +/- {:.2e}".format(*stats))

  if opts.debug_plot == 'plot':
    i, j, k = 0, 1, 2
    l = np.concatenate([
          output[idxs == i, :],
          output[idxs == j, :],
          output[idxs == k, :]
        ])
    print('Similarities')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(np.dot(l, l.T))
    im1 = ax1.imshow(l)
    im2 = ax2.imshow(mean_sims)
    fig.colorbar(im0, ax=ax0)
    plt.show()
    print('Diagonal and off diagonal histograms')
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.hist([ diag, baseline_diag ], bins=20, normed=1, label=[ 'diag', 'baseline_diag' ])
    ax0.legend()
    ax0.set_title('Diagonal Similarity Rate')
    ax1.hist([ off_diag, baseline_off_diag ], bins=20, normed=1, label=[ 'diag', 'baseline_diag' ])
    ax1.set_title('Off Diagonal Similarity Rate')
    ax1.legend()
    plt.show()

  return stats
  

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
        stats[i] = experiment(opts, network, i)
    else:
      for i in tqdm.tqdm(range(n)):
        stats[i] = experiment(opts, network, i)
    meanstats = np.mean(stats,0)
    print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Diag: {:.2e} +/- {:.2e}, " \
          "Baseline Off Diag: {:.2e} +/- {:.2e}".format(*list(meanstats)))
  elif opts.debug_plot == 'plot':
    experiment(opts, network, opts.debug_index)

