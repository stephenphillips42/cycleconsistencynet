import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import myutils
import options
import model

def axes3d(nrows=1, ncols=1):
  fig = plt.figure()
  axes = [ fig.add_subplot(nrows, ncols, i+1, projection='3d') for i in range(nrows*ncols) ]
  return fig, axes

def npload(fdir,idx):
  return dict(np.load("{}/np_test-{:04d}.npz".format(fdir,idx)))

def main(opts, index):
  # Load sample
  sample = npload(os.path.join(opts.debug_dir, 'np_test'), index)
  network = model.get_network(opts, opts.arch)
  network.load_np(opts.save_dir)
  output = network.apply_np(sample)
  # Sort by ground truth for better visualization
  labels = sample['TrueEmbedding']
  idxs = np.argmax(labels, axis=1)
  sorted_idxs = np.argsort(idxs)
  slabels = labels[sorted_idxs]
  soutput = output[sorted_idxs]
  # Plot setup
  shape = slabels.shape
  rand = myutils.dim_normalize(np.random.randn(*shape))
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
  if opts.debug_plot:
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
  for i in range(npts):
    for j in range(npts):
      mean_sims[i,j] = np.mean(np.dot(output[idxs == i], output[idxs == j].T))
      if i == j:
        diag.append(np.abs(mean_sims[i,j]))
      else:
        off_diag.append(np.abs(mean_sims[i,j]))
  stats = (np.mean(diag), np.std(diag), np.mean(off_diag), np.std(off_diag))
  print("Diag: {:.2e} +/- {:.2e}, Off Diag: {:.2e} +/- {:.2e}".format(*stats))
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
  

if __name__ == "__main__":
  opts = options.get_opts()
  if not opts.debug_plot:
    for i in range(opts.num_gen_test):
      main(opts, i)
  else:
    main(opts, opts.debug_index)

