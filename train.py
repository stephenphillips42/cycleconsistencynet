# -*- coding: utf-8 -*-
import os
import sys
import collections
import signal
import time
import numpy as np

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog                 
                 
from data_util import dataset
import model
import myutils
import tfutils
import options


def get_loss(opts, sample, output, return_true_val=False, name='loss'):
  emb = sample['TrueEmbedding']
  output_sim = tfutils.get_sim(output)
  sim_true = tfutils.get_sim(emb)
  if opts.use_unsupervised_loss:
    v = opts.dataset_params.views[-1]
    p = opts.dataset_params.points[-1]
    b = opts.batch_size 
    sim = sample['AdjMat'] + tf.eye(num_rows=v*p, batch_shape=[b])
  else:
    sim = sim_true
  tf.summary.image('Output Similarity', tf.expand_dims(output_sim, -1))
  tf.summary.image('Embedding Similarity', tf.expand_dims(emb_true, -1))
  if opts.loss_type == 'l2':
    loss = tf.losses.mean_squared_error(sim, output_sim)
  elif opts.loss_type == 'bce':
    bce_elements = tf.nn.sigmoid_cross_entropy_with_logits(labels=sim, logits=output_sim)
    loss = tf.reduce_sum(bce_elements)
  tf.summary.scalar(name, loss)
  gt_loss = None
  if return_true_val or opts.full_tensorboard:
    gt_loss = tf.metrics.mean_squared_error(sim_true, output_sim)
  if opts.full_tensorboard and opts.use_unsupervised_loss:
    tf.summary.scalr('GT L2 loss', gt_loss)
  if return_true_val:
    return loss, gt_loss
  else:
    return loss

def build_optimizer(opts, global_step):
  # Learning parameters post-processing
  num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
  decay_steps = int(num_batches * opts.learning_rate_decay_epochs)
  use_staircase = (not opts.learning_rate_continuous)
  if opts.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(opts.learning_rate, name='fixed_learning_rate')
  elif opts.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                               global_step,
                                               decay_steps,
                                               opts.learning_rate_decay_rate,
                                               staircase=use_staircase,
                                               name='learning_rate')
  elif opts.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.train.polynomial_decay(opts.learning_rate,
                                              global_step,
                                              decay_steps,
                                              opts.min_learning_rate,
                                              power=1.0,
                                              cycle=False,
                                              name='learning_rate')

  if opts.full_tensorboard:
    tf.summary.scalar('learning_rate', learning_rate)
  # TODO: add individual adam options to these
  if opts.optimizer_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif opts.optimizer_type == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif opts.optimizer_type == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate,opts.momentum)
  elif opts.optimizer_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif opts.optimizer_type == 'adamw':
    optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate)

  return optimizer

def get_train_op(opts, loss):
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = None
  if opts.weight_decay > 0 or opts.weight_l1_decay > 0:
    reg_loss = tf.losses.get_regularization_loss()
    reg_optimizer = tf.train.GradientDescentOptimizer(
                            learning_rate=opts.weight_decay)
    reg_step = reg_optimizer.minimize(reg_loss, global_step=global_step)
    with tf.control_dependencies([reg_step]):
      train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
  
def build_session(opts):
  saver_hook = tf.train.CheckpointSaverHook(opts.save_dir,
                                            save_secs=opts.save_interval_secs)
  merged = tf.summary.merge_all()
  summary_hook = tf.train.SummarySaverHook(output_dir=opts.save_dir, 
                                           summary_op=merged,
                                           save_secs=opts.save_summaries_secs)
  all_hooks = [saver_hook, summary_hook]
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.train.SingularMonitoredSession(
            checkpoint_dir=opts.save_dir,
            hooks=all_hooks,
            config=config)

def get_intervals(opts):
  if opts.num_epochs > 0:
    num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
    train_steps = int(num_batches * opts.num_epochs)
  else:
    train_steps = None
  if opts.train_time > 0:
    train_time = opts.train_time * 60
  else:
    train_time = None
  if opts.test_freq > 0:
    test_freq = opts.test_freq * 60
  else:
    test_freq = None
  if opts.test_freq_steps > 0:
    test_freq_steps = opts.test_freq_steps
  else:
    test_freq_steps = None
  return train_steps, train_time, test_freq_steps, test_freq

def get_test_dict(opts, mydataset, network)
  test_data = {}
  test_data['sample'] = mydataset.load_batch('test')
  test_data['output'] = network(test_data['sample']['Laplacian'],
                                test_data['sample']['InitEmbeddings'])
  if opts.use_unsupervised_loss:
    test_loss, test_gt_loss = get_loss(opts,
                                 test_data['sample'],
                                 test_data['output'],
                                 name='test_loss')
    test_data['loss'] = test_loss
    test_data['loss_gt'] = test_gt_loss
  else:
    test_data['loss'] = get_loss(opts,
                                 test_data['sample'],
                                 test_data['output'],
                                 name='test_loss')
  num_batches = 1.0 * opts.dataset_params.sizes['test'] / opts.batch_size
  test_data['nsteps'] = int(num_batches)
  return test_data

def run_test(opts, sess, test_data, verbose=True):
  npsave = {}
  teststr = " ------------------- "
  start_time = time.time()
  if opts.use_unsupervised_loss:
    teststr += " Test loss = {:.4e}, GT Loss: {:4e} ({:.01} sec)"
    summed_loss, \
    summed_loss_gt, \
    npsave['output'], \
    npsave['input'], \
    npsave['adjmat'], \
    npsave['gt'] = \
      sess.run([ 
        test_data['loss'],
        test_data['loss_gt'],
        test_data['output'],
        test_data['sample']['InitEmbeddings'],
        test_data['sample']['AdjMat'],
        test_data['sample']['TrueEmbedding'],
      ])
    for _ in range(test_data['nsteps']-1):
      sl, slgt = sess.run(test_data['loss'], test_data['loss_gt'])
      summed_loss += sl
      summed_loss_gt += slgt
    strargs = (summed_loss / test_data['nsteps'], summed_loss_gt / test_data['nsteps'])
  else:
    teststr += " Test loss = {:.4e} ({:.01} sec)"
    summed_loss, \
    npsave['output'], \
    npsave['input'], \
    npsave['adjmat'], \
    npsave['gt'] = \
      sess.run([ 
        test_data['loss'],
        test_data['output'],
        test_data['sample']['InitEmbeddings'],
        test_data['sample']['AdjMat'],
        test_data['sample']['TrueEmbedding'],
      ])
    for _ in range(test_data['nsteps']-1):
      summed_loss += sess.run(test_data['loss'])
    strargs = (summed_loss / test_data['nsteps'], )
  np.savez(myutils.next_file(opts.save_dir, 'test', '.npz'), **npsave)
  ctime = time.time()
  tf.logging.info(teststr.format(*strargs, ctime-start_time))

# TODO: Make this a class to deal with all the variable passing
def train(opts):
  # Get data and network
  mydataset = dataset.get_dataset(opts)
  network = model.get_network(opts, opts.arch)
  # Training
  if opts.load_data:
    sample = mydataset.load_batch('train')
  else:
    sample = mydataset.gen_batch('train')
  output = network(sample['Laplacian'], sample['InitEmbeddings'])
  loss = get_loss(opts, sample, output)
  train_op = get_train_op(opts, loss)
  # Testing
  test_data = get_test_dict(opts, mydataset, network)

  # Tensorflow and logging operations
  step = 0
  train_steps, train_time, test_freq_steps, test_freq = get_intervals(opts)
  trainstr = "global step {}: loss = {} ({:.04} sec/step, time {:.04})"
  tf.logging.set_verbosity(tf.logging.INFO)
  # Build session
  with build_session(opts) as sess:
    # Train loop
    for run in range(opts.num_runs):
      stime = time.time()
      ctime = stime
      ttime = stime
      while step != train_steps and ctime - stime <= train_time:
        start_time = time.time()
        x = sess.run([ output ])
        _, loss_ = sess.run([ train_op, loss ])
        ctime = time.time()
        if (step % opts.log_steps) == 0:
          tf.logging.info(trainstr.format(step,
                                          loss_,
                                          ctime - start_time,
                                          ctime - stime))
        if ((test_freq_steps and step % test_freq_steps == 0) or \
            (ctime - ttime > test_freq)):
          raw_sess = sess.raw_session()
          run_test(opts, raw_sess, test_data)
          ttime = time.time()
        step += 1

if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

