# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import time
import sklearn.metrics as metrics

import tensorflow as tf

import data_util.datasets
import model
import myutils
import tfutils
import options


loss_fns = {
  'l1': tfutils.l1_loss,
  'l2': tfutils.l2_loss,
  'l1l2': tfutils.l1_l2_loss,
}

# TODO: Almost function from sptrain
def get_test_output_sim(self, output_graph):
  output = output_graph.nodes
  output_sim = tf.matmul(output, output, transpose_b=True)
  if self.use_abs_value:
    output_sim = tf.abs(output_sim)
  return output_sim

def get_tf_test_losses(opts, sample, output_sim):
  # Compute similarity matrices
  b = opts.batch_size
  v = opts.dataset_params.views[-1]
  p = opts.dataset_params.points[-1]
  sim = tf.sparse_tensor_to_dense(sample['true_adj_mat'])
  sim_true = sim + tf.reshape(tf.eye(num_rows=v*p), [v*p, v*p])
  osim = output_sim
  # Compute losses
  losses_ = {}
  # Standard losses
  losses_['l1'] = loss_fns['l1'](osim, sim_true, add_loss=False)
  losses_['l2'] = loss_fns['l2'](osim, sim_true, add_loss=False)
  # Histogram based losses
  num_same = tf.reduce_sum(sim_true)
  num_diff = tf.reduce_sum(1-sim_true)
  ssame_m, ssame_var = tf.nn.weighted_moments(osim, None, sim_true)
  losses_['ssame_m'], losses_['ssame_var'] = ssame_m, ssame_var
  sdiff_m, sdiff_var = tf.nn.weighted_moments(osim, None, 1-sim_true)
  losses_['sdiff_m'], losses_['sdiff_var'] = sdiff_m, sdiff_var
  # Return
  return losses_

def get_np_losses(opts, output_sim, matches):
  # return { 'roc': 0, 'p_r': 0 }
  true_emb = np.eye(opts.dataset_params.points[-1])[matches]
  adjmat = np.dot(true_emb, true_emb.T).reshape(-1)
  output = output_sim.reshape(-1)
  roc =  metrics.roc_auc_score(adjmat, output)
  p_r =  metrics.average_precision_score(adjmat, output)
  return { 'roc': roc, 'p_r': p_r }


def build_test_session(opts):
  config = tf.ConfigProto(device_count = {'GPU': 0})
  # config.gpu_options.allow_growth = True
  return tf.Session(config=config)

# TODO: Make this a class to deal with all the variable passing
def test_values(opts):
  # Get data and network
  dataset = data_util.datasets.get_dataset(opts)
  network = model.get_network(opts, opts.arch)
  # Sample and network output
  sample = dataset.load_batch('test', repeat=1)
  print('sample')
  print(sample)
  output_graph = network(sample)
  print('output_graph')
  print(output_graph)
  output_sim = get_test_output_sim(opts, output_graph)
  losses = get_tf_test_losses(opts, sample, output_sim)
  tf_evals = [ losses, output_sim, sample['true_match'] ] 

  # Tensorflow and logging operations
  disp_string =  '{idx:06d}: ' \
                 'L1: {l1:.03e},  L2: {l2:.03e}, ' \
                 'Same sim: {ssame_m:.03e} +/- {ssame_var:.03e}, ' \
                 'Diff sim: {sdiff_m:.03e} +/- {sdiff_var:.03e}, ' \
                 'ROC: {roc:.03e}, ' \
                 'P-R: {p_r:.03e}, ' \
                 'Time: {time:.03e}'

  # Build session
  glob_str = os.path.join(opts.dataset_params.data_dir, 'np_test', '*npz')
  # vars_restore = [ v for v in tf.get_collection('weights') ] + \
  #                [ v for v in tf.get_collection('biases') ]
  vars_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  # print(vars_restore)
  saver = tf.train.Saver(vars_restore)
  with open(os.path.join(opts.save_dir, 'test_output.log'), 'a') as log_file:
    with build_test_session(opts) as sess:
      saver.restore(sess, tf.train.latest_checkpoint(opts.save_dir))
      for i in range(opts.dataset_params.sizes['test']):
        start_time = time.time()
        values_, output_sim_, matches_ = sess.run(tf_evals)
        end_time = time.time()
        values_['idx'] = i
        values_['time'] = end_time - start_time
        values_.update(get_np_losses(opts, output_sim_, matches_[0]))
        dstr = disp_string.format(**values_)
        print(dstr)
        log_file.write(dstr)
        log_file.write('\n')

if __name__ == "__main__":
  opts = options.get_opts()
  print("Getting options from run...")
  # TODO: Separate yaml_opts and opts from command line
  opts = options.parse_yaml_opts(opts)
  print("Done")
  # Some value are set
  opts.batch_size = 1
  # Run tests
  test_values(opts)


