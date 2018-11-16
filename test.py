# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np

import tensorflow as tf

import data_util.datasets
import model
import myutils
import tfutils
import options

loss_fns = {
  'bce': tfutils.bce_loss,
  'l1': tfutils.l1_loss,
  'l2': tfutils.l2_loss,
  'l1l2': tfutils.l1_l2_loss,
}

# TODO: Make this not a hack
# HACK!!
end_bias_global = None
def get_end_bias():
  global end_bias_global
  if end_bias_global is None:
    with tf.variable_scope("end_variables", reuse=tf.AUTO_REUSE):
      end_bias_global = tf.get_variable('end_bias', 
                                        initializer=tf.zeros((1,),
                                        dtype=tf.float32))
    if end_bias_global not in tf.get_collection('biases'):
      tf.add_to_collection('biases', end_bias_global)
  return end_bias_global
# END HACK!!

def get_test_losses(opts, sample, output, return_gt=False, name='loss'):
  emb = sample['TrueEmbedding']
  output_sim = tfutils.get_sim(output)
  sim_true = tfutils.get_sim(emb)
  if opts.use_end_bias:
    end_bias = get_end_bias()
    output_sim = output_sim + end_bias
  if opts.loss_type == 'bce':
    osim = tf.sigmoid(output_sim)
    osim_log = output_sim
  else:
    osim = output_sim
    osim_log = tf.log(tf.abs(output_sim) + 1e-9)
  gt_l1_loss = loss_fns['l1'](sim_true, osim, add_loss=False)
  gt_l2_loss = loss_fns['l2'](sim_true, osim, add_loss=False)
  gt_bce_loss = loss_fns['bce'](sim_true, osim, add_loss=False)
  num_same = tf.reduce_sum(sim_true)
  num_diff = tf.reduce_sum(1-sim_true)
  ssame_m, ssame_var = tf.nn.weighted_moments(osim, None, sim_true)
  sdiff_m, sdiff_var = tf.nn.weighted_moments(osim, None, 1-sim_true)

  return gt_l1_loss, gt_l2_loss, gt_bce_loss, ssame_m, ssame_var, sdiff_m, sdiff_var

def build_test_session(opts):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)


# TODO: Make this a class to deal with all the variable passing
def test_values(opts):
  # Get data and network
  dataset = data_util.datasets.get_dataset(opts)
  network = model.get_network(opts, opts.arch)
  # Sample
  sample = dataset.get_placeholders()
  print(sample)
  output = network(sample['Laplacian'], sample['InitEmbeddings'])
  losses = get_test_losses(opts, sample, output)

  # Tensorflow and logging operations
  disp_string =  '{:06d} Errors: ' \
                 'L1: {:.03e},  L2: {:.03e}, BCE: {:.03e} ' \
                 'Same sim: {:.03e} +/- {:.03e} ' \
                 'Diff sim: {:.03e} +/- {:.03e}' 


  # Build session
  glob_str = os.path.join(opts.dataset_params.data_dir, 'np_test', '*npz')
  npz_files = sorted(glob.glob(glob_str))
  vars_restore = [ v for v in tf.get_collection('weights') ] + \
                 [ v for v in tf.get_collection('biases') ]
  print(vars_restore)
  saver = tf.train.Saver(vars_restore)
  with open(os.path.join(opts.save_dir, 'test_output.log'), 'a') as log_file:
    with build_test_session(opts) as sess:
      saver.restore(sess, tf.train.latest_checkpoint(opts.save_dir))
      for i, npz_file in enumerate(npz_files):
        sample_ = { k : np.expand_dims(v,0) for k, v in np.load(npz_file).items() }
        vals = sess.run(losses, { sample[k] : sample_[k] for k in sample.keys() })
        dstr = disp_string.format(i, *vals)
        print(dstr)
        log_file.write(dstr)
        log_file.write('\n')

if __name__ == "__main__":
  opts = options.get_opts()
  print("Getting options from run...")
  opts = options.parse_yaml_opts(opts)
  print("Done")
  test_values(opts)

