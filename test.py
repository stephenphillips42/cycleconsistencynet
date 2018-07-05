# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import scipy.linalg as la
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim

import data_util
import model
import myutils
import tfutils
import options


def test(opts):
  # Get data and network
  dataset = data_util.get_dataset(opts)
  sample = dataset.load_batch('test')
  network = model.get_network(opts, opts.arch)

  # Get loss
  emb = sample['TrueEmbedding']
  output = network.apply(sample)
  emb_sim, output_sim = tfutils.get_sim(emb), tfutils.get_sim(output)
  tf.summary.image('Output Similarity', tf.expand_dims(output_sim, -1))
  tf.summary.image('Embedding Similarity', tf.expand_dims(emb_sim, -1))
  tf.losses.mean_squared_error(emb_sim, output_sim)
  loss = tf.losses.get_total_loss()
  loss = tf.Print(loss, [loss], message='Loss Value: ', first_n=-1)
  tf.summary.scalar('Loss', loss)
  names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
    "mse": tf.metrics.mean_squared_error(emb_sim, output_sim),
  })


  # Test loop
  tf.logging.set_verbosity(tf.logging.INFO)
  logdir = os.path.join(opts.save_dir, 'test_logs')
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  slim.evaluation.evaluate_once(
          master='',
          checkpoint_path=tf.train.latest_checkpoint(opts.save_dir),
          logdir=logdir)

if __name__ == "__main__":
  opts = options.get_opts()
  test(opts)

