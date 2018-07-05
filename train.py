# -*- coding: utf-8 -*-
import os
import sys
import collections
import signal

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
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
    bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=emb_true, logits=output_sim))
    tf.losses.add_loss(bce)
  loss = tf.losses.get_total_loss()
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
  train_op = slim.learning.create_train_op(total_loss=loss,
                                           optimizer=optimizer,
                                           global_step=global_step,
                                           clip_gradient_norm=5)
  return train_op
  
def get_max_steps(opts):
  if opts.num_epochs > 0:
    num_batches = 1.0 * opts.dataset_params.sizes['train'] / opts.batch_size
    max_steps = int(num_batches * opts.num_epochs)
  else:
    max_steps = None
  return max_steps

def handler(signum, frame):
  print("Training finished")
  raise myutils.TimeRunException("Finished running script")

def train_with_generation(opts):
  # Get data and network
  dataset = data_util.get_dataset(opts)
  if opts.load_data:
    sample = dataset.load_batch('train')
  else:
    sample = dataset.get_placeholders()
  network = model.get_network(opts, opts.arch)
  output = network.apply(sample)
  loss = get_loss(opts, sample, output)
  train_op = get_train_op(opts, loss)

  # Tensorflow and logging operations
  init_op = tf.global_variables_initializer()
  global_step = tf.train.get_or_create_global_step()
  merged = tf.summary.merge_all()
  step = 0
  max_steps = get_max_steps(opts)
  INFO = "INFO:tensorflow:global step {}: loss = {} (0.00 sec/step)"
  saver = tf.train.Saver()

  # Build session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(opts.save_dir,
                                           sess.graph,
                                           flush_secs=opts.save_summaries_secs)
    # Train loop
    for run in range(opts.num_runs):
      if opts.run_time > 0:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60*opts.run_time) # run time in seconds
      try:
        while step != max_steps:
          c = 0
          batch = dataset.get_np_batch(opts.batch_size)
          feed = { sample[k] : batch[k] for k in batch.keys() }
          summary, loss_, _ = sess.run([merged, loss, train_op], feed_dict=feed)
          summary_writer.add_summary(summary, step)
          if ((step + 1) % opts.log_steps) == 0:
            print(INFO.format(step, loss_))
          if (step % opts.save_interval_steps) == 0:
            saver.save(sess,
                       os.path.join(opts.save_dir, 'model.ckpt'),
                       global_step=global_step)
            slog = SessionLog(status=SessionLog.CHECKPOINT,
                              checkpoint_path=opts.save_dir)
            summary_writer.add_session_log(slog, step)
          step += 1
      except myutils.TimeRunException as exp:
        print("Exiting training...")
      finally:
        network.save_np(saver, opts.save_dir)


def train(opts):
  # Get data and network
  dataset = data_util.get_dataset(opts)
  if opts.load_data:
    sample = dataset.load_batch('train')
  else:
    sample = dataset.get_placeholders()
  network = model.get_network(opts, opts.arch)
  output = network.apply(sample)
  loss = get_loss(opts, sample, output)
  train_op = get_train_op(opts, loss)
  global_step = tf.train.get_or_create_global_step()

  # Train loop
  saver = tf.train.Saver()
  tf.logging.set_verbosity(tf.logging.INFO)
  for run in range(opts.num_runs):
    if opts.run_time > 0:
      signal.signal(signal.SIGALRM, handler)
      signal.alarm(60*opts.run_time) # run time in seconds
    try:
      slim.learning.train(
              train_op=train_op,
              logdir=opts.save_dir,
              number_of_steps=get_max_steps(opts),
              log_every_n_steps=opts.log_steps,
              saver=saver,
              save_summaries_secs=opts.save_summaries_secs,
              save_interval_secs=opts.save_interval_secs)

    except myutils.TimeRunException as exp:
      print("Exiting training...")
    finally:
      network.save_np(saver, opts.save_dir)

if __name__ == "__main__":
  opts = options.get_opts()
  if opts.load_data:
    train(opts)
  else:
    train_with_generation(opts)

