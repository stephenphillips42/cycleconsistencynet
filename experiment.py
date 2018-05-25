import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tqdm

import myutils
import options
import model

def axes3d(nrows=1, ncols=1):
  fig = plt.figure()
  axes = [ fig.add_subplot(nrows, ncols, i+1, projection='3d') for i in range(nrows*ncols) ]
  return fig, axes

def npload(fdir,idx):
  return dict(np.load("{}/np_test-{:04d}.npz".format(fdir,idx)))

def experiment(opts, network, index):
  # Load sample
  sample = npload(os.path.join(opts.debug_dir, 'np_test'), index)
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
  # if opts.debug_plot:
  if False:
    # Plotting images
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(slabels)
    im1 = ax1.imshow(soutput)
    im2 = ax2.imshow(rand)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(lcorr)
    im1 = ax1.imshow(ocorr)
    im2 = ax2.imshow(rcorr)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(lsim)
    im1 = ax1.imshow(osim)
    im2 = ax2.imshow(rsim)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.show()

    # Plotting 3D figures
    fig, (ax0, ax1, ax2) = axes3d(nrows=1, ncols=3)
    im0 = ax0.plot_surface(xx0, yy0, lsim, cmap=cm.coolwarm)
    im1 = ax1.plot_surface(xx0, yy0, osim, cmap=cm.coolwarm)
    im2 = ax2.plot_surface(xx0, yy0, rsim, cmap=cm.coolwarm)
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

  if opts.debug_plot:
    i, j, k = 0, 1, 2
    l = np.concatenate([
          output[idxs == i, :],
          output[idxs == j, :],
          output[idxs == k, :]
        ])
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.imshow(np.dot(l, l.T))
    im1 = ax1.imshow(l)
    im2 = ax2.imshow(mean_sims)
    fig.colorbar(im0, ax=ax0)
    plt.show()
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.hist([ diag, baseline_diag ], bins=20, normed=1)
    ax0.set_title('Diagonal Similarity Rate')
    ax1.hist([ off_diag, baseline_off_diag ], bins=20, normed=1)
    ax1.set_title('Off Diagonal Similarity Rate')
    plt.show()

  return stats
  

if __name__ == "__main__":
  opts = options.get_opts()
  network = model.get_network(opts, opts.arch)
  network.load_np(opts.save_dir)
  if not opts.debug_plot:
    n = opts.dataset_params.sizes['test']
    stats = np.zeros((n,8))
    if opts.verbose:
      for i in range(n):
        stats[i] = experiment(opts, network, i)
    else:
      for i in tqdm.tqdm(range(n)):
        stats[i] = experiment(opts, network, i)
    print(np.mean(stats,0))
  else:
    experiment(opts, network, opts.debug_index)

