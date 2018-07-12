# -*- coding: utf-8 -*-
import os
import sys
import collections
import signal
import time

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog                 
                 
import data_util
import model
import myutils
import tfutils
import options


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
  loss = tf.losses.get_total_loss(add_regularization_losses=False,
                                  name='total_loss')
  tf.summary.scalar('Loss', loss)
  return loss

def build_optimizer(opts, global_step):
  # Learning parameters post-processing
  num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
  decay_steps = int(num_batches * opts.learning_rate_decay_epochs)
  if opts.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(opts.learning_rate, name='fixed_learning_rate')
  elif opts.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(opts.learning_rate,
                                               global_step,
                                               decay_steps,
                                               opts.learning_rate_decay_rate,
                                               staircase=True,
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

  return optimizer

def get_train_op(opts, loss):
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = None
  if opts.weight_decay > 0 or opts.weight_l1_decay > 0:
    reg_loss = tf.losses.get_regularization_loss()
    reg_optimizer = tf.train.GradientDescentOptimizer(
                            learning_rate=opts.weight_decay)
    with tf.control_dependencies([reg_optimizer.minimize(reg_loss)]):
      train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
  
def build_session(opts):
  saver_hook = tf.train.CheckpointSaverHook(opts.save_dir,
                                            save_secs=opts.save_interval_steps)
  merged = tf.summary.merge_all()
  summary_hook = tf.train.SummarySaverHook(output_dir=opts.save_dir, 
                                           summary_op=merged,
                                           save_secs=opts.save_summaries_secs)
  all_hooks = [saver_hook, summary_hook]
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.train.SingularMonitoredSession(hooks=all_hooks, config=config)

def get_max_steps_and_time(opts):
  if opts.num_epochs > 0:
    num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
    max_steps = int(num_batches * opts.num_epochs)
  else:
    max_steps = float('inf')
  if opts.run_time > 0:
    run_time = opts.run_time * 60
  else:
    run_time = float('inf')
  return max_steps, run_time

def train(opts):
  # Get data and network
  dataset = data_util.get_dataset(opts)
  if opts.load_data:
    sample = dataset.load_batch('train')
  else:
    sample = dataset.gen_batch('train')
  network = model.get_network(opts, opts.arch)
  output = network(sample['Laplacian'], sample['InitEmbeddings'])
  loss = get_loss(opts, sample, output)
  train_op = get_train_op(opts, loss)

  # Tensorflow and logging operations
  step = 0
  max_steps, run_time = get_max_steps_and_time(opts)
  printstr = "global step {}: loss = {} ({:.04} sec/step)"
  tf.logging.set_verbosity(tf.logging.INFO)
  # Build session
  with build_session(opts) as sess:
    # Train loop
    for run in range(opts.num_runs):
      stime = time.time()
      ctime = stime
      while step != max_steps and ctime - stime <= run_time:
        start_time = time.time()
        _, loss_ = sess.run([ train_op, loss ])
        ctime = time.time()
        if ((step + 1) % opts.log_steps) == 0:
          tf.logging.info(printstr.format(step, loss_, ctime-start_time))
        step += 1
        # network.save_np(saver, opts.save_dir)

if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

