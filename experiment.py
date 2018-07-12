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

# TODO: Get this to another file
def get_loss(opts, sample, output):
  emb = sample['TrueEmbedding']
  output_sim = tfutils.get_sim(output)
  if opts.use_unsupervised_loss:
    v = opts.dataset_params.views[-1]
    p = opts.dataset_params.points[-1]
    b = opts.batch_size 
    emb_true = sample['AdjMat'] + tf.eye(v*p, b)
  else:
    emb_true = tfutils.get_sim(emb)
  tf.summary.image('Output Similarity', tf.expand_dims(output_sim, -1))
  tf.summary.image('Embedding Similarity', tf.expand_dims(emb_true, -1))
  if opts.loss_type == 'l2':
    tf.losses.mean_squared_error(emb_true, output_sim)
  elif opts.loss_type == 'bce':
    bce_elements = tf.nn.sigmoid_cross_entropy_with_logits(labels=emb_true, logits=output_sim)
    bce = tf.reduce_sum(bce_elements)
    tf.losses.add_loss(bce)
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('Loss', loss)
  return loss


def plot_hist(sim_mats, names, true_sim):
  fig, ax = plt.subplots(nrows=1, ncols=2)
  diags = [ np.reshape(v[true_sim==1],-1) for v in sim_mats ]
  off_diags = [ np.reshape(v[true_sim==0],-1) for v in sim_mats ]
  ax[0].hist(diags, bins=20, density=True, label=names)
  ax[0].legend()
  ax[0].set_title('Diagonal Similarity Rate')
  ax[1].hist(off_diags, bins=20, density=True, label=names)
  ax[1].set_title('Off Diagonal Similarity Rate')
  ax[1].legend()
  plt.show()

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

def plot_index_unsorted(opts, network, index):
  print(index)
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  output = network.apply_np(sample)
  labels = sample['TrueEmbedding']
  rand = myutils.dim_normalize(sample['InitEmbeddings'])
  lsim = np.abs(np.dot(labels, labels.T))
  osim = np.abs(np.dot(output, output.T))
  rsim = np.abs(np.dot(rand, rand.T))
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
  im0 = ax0.imshow(output)
  im1 = ax1.imshow(osim)
  im2 = ax2.imshow(sample['AdjMat'] + np.eye(sample['AdjMat'].shape[0]))
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

def plot_random(opts, network, index):
  sample = npload(os.path.join(opts.data_dir, 'np_test'), index)
  output = network.apply_np(sample)
  slabels, sorted_idxs = get_sorted(sample)
  soutput = output[sorted_idxs]
  srand = myutils.dim_normalize(sample['InitEmbeddings'][sorted_idxs])
  lsim = np.abs(np.dot(slabels, slabels.T))
  osim = np.abs(np.dot(soutput, soutput.T))
  rsim = np.abs(np.dot(srand, srand.T))
  plots = [ rsim, osim, osim**9 ]
  names = [ 'rsim', 'osim', 'osim**9' ]
  # fig, ax = plt.subplots(nrows=1, ncols=len(plots))
  # for i in range(len(ax)):
  #   im = ax[i].imshow(plots[i])
  #   fig.colorbar(im, ax=ax[i])
  # plt.show()
  # plot_hist(plots, [ 'rsim', 'osim', 'osim**9' ], lsim)
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
  opts.debug_index = debug_opts.debug_index
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
    plot_index(opts, opts.debug_index)
  elif opts.debug_plot == 'unsorted':
    plot_index_unsorted(opts, opts.debug_index)
  elif opts.debug_plot == 'baseline':
    plot_baseline(opts, opts.debug_index)
  elif opts.debug_plot == 'random':
    plot_random(opts, opts.debug_index)



