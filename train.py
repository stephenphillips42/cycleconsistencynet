# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import signal
import scipy.linalg as la
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step

import data_util
import model
import myutils
import tfutils
import options


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

def handler(signum, frame):
  print("Training finished")
  raise myutils.TimeRunException("Finished running script")

def train(opts):
  # Get data and network
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch('train')
  network = model.get_network(opts, opts.arch)

  # Get loss
  emb = sample['TrueEmbedding']
  output = network.apply(sample)
  emb_sim, output_sim = tfutils.get_sim(emb), tfutils.get_sim(output)
  tf.summary.image('Output Similarity', tf.expand_dims(output_sim, -1))
  tf.summary.image('Embedding Similarity', tf.expand_dims(emb_sim, -1))
  tf.losses.mean_squared_error(emb_sim, output_sim)
  loss = tf.losses.get_total_loss()
  tf.summary.scalar('Loss', loss)

  # Training objects
  global_step = tf.train.get_or_create_global_step()
  optimizer = build_optimizer(opts, global_step)
  train_op = slim.learning.create_train_op(total_loss=loss,
                                           optimizer=optimizer,
                                           global_step=global_step,
																					 clip_gradient_norm=5)

  tf.logging.set_verbosity(tf.logging.INFO)
  if opts.num_epochs > 0:
    num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
    max_steps = int(num_batches * opts.num_epochs)
  else:
    max_steps = None

  # Train-test loop
  saver = tf.train.Saver()
  if opts.run_time > 0:
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60*opts.run_time)
  try:
    slim.learning.train(
            train_op=train_op,
            logdir=opts.save_dir,
            number_of_steps=max_steps,
            log_every_n_steps=opts.log_steps,
            saver=saver,
            save_summaries_secs=opts.save_summaries_secs,
            save_interval_secs=opts.save_interval_secs)

    network.save_np(saver, opts.save_dir)
  except myutils.TimeRunException as exp:
    print("Exiting training...")


if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

