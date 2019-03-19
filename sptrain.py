# -*- coding: utf-8 -*-
import os
import sys
import collections
import signal
import time
import numpy as np

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog                 
                 
import data_util.datasets
import model
import myutils
import tfutils
import options

loss_fns = {
  'l1': tfutils.l1_loss_sp,
  'l2': tfutils.l2_loss_sp,
  'l1l2': tfutils.l1_l2_loss_sp,
}

class Trainer(object):
  """Class responsible for training network and handling all logging"""
  def __init__(self, opts):
    self.log_file = open(os.path.join(opts.save_dir, 'logfile.log'), 'a')
    self.opts = opts
    self.__dict__.update(opts.__dict__)
    b = self.batch_size
    v = self.dataset_params.views[-1]
    p = self.dataset_params.points[-1]
    self.tensor_sizes = (b,v,p)
    # For training/testing time management
    self._global_step = None
    self._train_steps = None
    if self.num_epochs > 0:
      num_batches = 1.0 * self.dataset_params.sizes['train'] / self.batch_size
      self._train_steps = int(num_batches * self.num_epochs)
    else:
      self._train_steps = np.inf
    self._train_time = None
    if self.train_time > 0:
      self._train_time = self.train_time * 60
    else:
      self._train_time = np.inf
    self._test_freq_steps = None
    if self.test_freq_steps > 0:
      self._test_freq_steps = self.test_freq_steps
    else:
      self._test_freq_steps = None
    self._test_freq = None
    if self.test_freq > 0:
      self._test_freq = self.test_freq * 60
    else:
      self._test_freq = np.inf
    # For storing and printing test values
    self.min_test_value = np.inf
    self.test_saver = None
    self._test_values = {}
    self._test_save = {}
    self._test_logs = {}
    self._test_disp = {}

  def __del__(self, *args):
    self.log_file.close()

  ########## Time/Resource management functions ##########
  def log(self, string):
    tf.logging.info(string)
    self.log_file.write(string)
    self.log_file.write('\n')

  def keep_training(self, step, cur_time):
    """Check if we should keep training after this step"""
    return step < self._train_steps and cur_time <= self._train_time

  def test_now(self, step, cur_time):
    """Check if we should test values"""
    return (
      (self._test_freq_steps and step % self._test_freq_steps == 0) or \
      (cur_time > self._test_freq)
    )

  ########## Setup functions ##########
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
  
    tf.summary.scalar('learning_rate', learning_rate)
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

  def get_train_op(self, loss):
    self.global_step = tf.train.get_or_create_global_step()
    optimizer = self.build_optimizer(self.global_step)
    train_op = None
    if self.weight_decay > 0 or self.weight_l1_decay > 0:
      wd = self.weight_decay if self.weight_decay > 0 else self.weight_l1_decay
      reg_loss = tf.losses.get_regularization_loss()
      reg_optimizer = tf.train.GradientDescentOptimizer(learning_rate=wd)
      reg_step = reg_optimizer.minimize(reg_loss, global_step=self.global_step)
      with tf.control_dependencies([reg_step]):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [ (tf.clip_by_value(grad, -1., 1.), var)
                       for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
    else:
      gvs = optimizer.compute_gradients(loss)
      capped_gvs = [ (tf.clip_by_value(grad, -1., 1.), var)
                     for grad, var in gvs]
      train_op = optimizer.apply_gradients(capped_gvs)
    return train_op

  def build_session(self):
    checkpoint_dir = self.save_dir
    if self.checkpoint_start_dir is not None:
      checkpoint_dir = self.checkpoint_start_dir
    saver_hook = tf.train.CheckpointSaverHook(self.save_dir,
                                              save_secs=self.save_interval_secs)
    merged = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(output_dir=self.save_dir, 
                                             summary_op=merged,
                                             save_secs=self.save_summaries_secs)
    all_hooks = [saver_hook, summary_hook]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.train.SingularMonitoredSession(
              checkpoint_dir=checkpoint_dir,
              hooks=all_hooks,
              config=config)

  def build_speye(self):
    b, v, p = self.tensor_sizes
    x = np.arange(v*p).reshape(-1,1)
    xb = np.arange(b).reshape(-1,1)
    o = np.ones((v*p,1))
    ob = np.ones((b,1))
    concat_vals = [np.kron(xb ,o)] + [np.kron(ob, x)] * 2
    speye_idxs = np.concatenate(concat_vals, 1).astype(np.int64)
    speye_vals = np.ones(b*v*p).astype(np.float32)
    speye = tf.SparseTensor(indices=tf.convert_to_tensor(speye_idxs),
                            values=tf.convert_to_tensor(speye_vals),
                            dense_shape=[ b, v*p, v*p ])
    return speye

  ########## Training and testing functions ##########
  def get_output_sim(self, out_graph):
    out_shape = [self.batch_size, -1, self.final_embedding_dim]
    output = tf.reshape(out_graph.nodes, out_shape)
    output_sim = tfutils.get_sim(output)
    if self.use_abs_value:
      output_sim = tf.abs(output_sim)
    return output_sim

  def get_geometric_loss(self, sample, output_sim, name='geo_loss'):
    b, v, p = self.tensor_sizes
    # Build rotation and cross product matrices
    R = tf.reshape(tf.tile(sample['Rotations'], [1, 1, p, 1]), [-1, v*p, 3, 3])
    T = tf.reshape(tf.tile(sample['Translations'], [1, 1, p]), [-1, v*p, 3])
    nodes = sample['graph'].nodes
    X = tf.concat([ nodes[...,-4:-2],
                    tf.tile(tf.ones((1,v*p,1)), [ b, 1, 1 ]) ], axis=-1)
    RX = tf.einsum('bvik,bvk->bvi',R,X)
    TcrossRX = tf.cross(T, RX)
    # Build finall Essential matrix distance score
    E_part = tfutils.batch_matmul(RX, tf.transpose(TcrossRX, perm=[0, 2, 1]))
    # npmask = np.kron(1-np.eye(v),np.ones((p,p)))
    # npmask = npmask.reshape(1,v*p,v*p).astype(self.dataset_params.dtype)
    # mask = tf.convert_to_tensor(npmask, name='mask_{}'.format(name))
    # E = tf.multiply(tf.abs(E_part + tf.transpose(E_part, [0, 2, 1])), mask)
    E = tf.abs(E_part + tf.transpose(E_part, [0, 2, 1]))
    # Logging stuff
    tf.summary.image('Geometric matrix {}'.format(name), tf.expand_dims(E, -1))
    tf.summary.histogram('Geometric matrix hist {}'.format(name), E)
    tf.summary.scalar('Geometric matrix norm {}'.format(name), tf.norm(E, ord=np.inf))
    return tf.reduce_mean(tf.multiply(output_sim, E), name=name)

  def get_loss(self, sample, output_sim, test_mode=False, name='train'):
    # Compute loss
    sim_true = sample['true_adj_mat']
    sim = tf.sparse_add(sample['adj_mat'], self.build_speye())
    reconstr_loss = loss_fns[self.loss_type](output_sim, sim)
    if self.geometric_loss > 0:
      geo_loss = self.get_geometric_loss(sample, output_sim, name='geom_loss_{}'.format(name))
      geo_loss_gt = get_geometric_loss(opts, sample, sim_true)
      loss = self.reconstruction_loss * reconstr_loss + \
                self.geometric_loss * geo_loss
    else:
      loss = reconstr_loss
    # Logging everything
    tf.summary.scalar('Total Loss {}'.format(name), loss)
    tf.summary.image('Output Similarity {}'.format(name),
                     tf.expand_dims(output_sim, -1))
    # dense_sim = tf.sparse_tensor_to_dense(sim, [b, v*p, v*p])
    # tf.summary.image('Embedding Similarity {}'.format(name),
    #                  tf.expand_dims(dense_sim, -1))
    tf.summary.scalar('Reconstruction Loss {}'.format(name), reconstr_loss)
    if self.geometric_loss > 0:
      tf.summary.scalar('Geometric Loss {}'.format(name), geo_loss)
      tf.summary.scalar('Geometric Loss GT {}'.format(name), geo_loss_gt)
    # Get additional test logs
    if test_mode:
      gt_l1_loss = loss_fns['l1'](output_sim, sim_true, add_loss=False)
      gt_l2_loss = loss_fns['l2'](output_sim, sim_true, add_loss=False)
      tf.summary.scalar('GT L1 Loss {}'.format(name), gt_l1_loss)
      tf.summary.scalar('GT L2 Loss {}'.format(name), gt_l2_loss)
      test_logs = {}
      test_logs['total_loss'] = loss
      test_logs['reconstr_loss'] = reconstr_loss
      test_logs['gt_l1'] = gt_l1_loss
      test_logs['gt_l2'] = gt_l2_loss
      return test_logs
    else:
      return loss

  def build_test(self, test_sample, network):
    # Save out so when we run tests we don't have to be passing things around
    self._test_values['sample'] = test_sample
    num_batches = self.dataset_params.sizes['test'] / self.batch_size
    self._test_values['nsteps'] = int(num_batches)
    # Values to save out for the first batch
    self._test_save['input_nodes'] = test_sample['graph'].nodes
    self._test_save['adj_mat'] = test_sample['adj_mat']
    self._test_save['true_match'] = test_sample['true_match']
    self._test_save['output'] = network(test_sample)
    # Compute losses
    self._test_values['output_sim'] = \
        self.get_output_sim(self._test_save['output'])
    self._test_logs = self.get_loss(self._test_values['sample'],
                                    self._test_values['output_sim'],
                                    test_mode=True,
                                    name='test')
    # Build saver
    self.test_saver = tf.train.Saver()
    # Display strings for test
    # TODO: Make this... not magical?
    self._test_disp = collections.OrderedDict([
        ('total_loss' , 'Loss'),
        ('reconstr_loss' , 'Reconstr. Loss'),
        ('gt_l1' , 'GT L1 Loss'),
        ('gt_l2' , 'GT L2 Loss'),
    ])

  def run_test(self, sess, verbose=True):
    # Setup
    npsave = {}
    # Build test display string from the losses we want to display
    teststr = " ------------------- "
    for k in self._test_disp:
      if k in self._test_logs:
        teststr += self._test_disp[k] + ': {:4e}, '
    teststr = teststr[:-2]
    teststr += " ({:.03} sec)"
    summed_vals = { k: 0 for k in self._test_logs }
    # Run experiment
    start_time = time.time()
    save_outputs = sess.run({ **self._test_save, **self._test_logs })
    # Start computing the aggregate values for our losses
    for k in self._test_logs:
      summed_vals[k] += save_outputs[k]
    for _ in range(self._test_values['nsteps']-1):
      run_outputs = sess.run(self._test_logs)
      for k in self._test_logs:
        summed_vals[k] += save_outputs[k]
    strargs = ( summed_vals[k] / self._test_values['nsteps']
                for k in self._test_disp )
    np.savez(myutils.next_file(self.save_dir, 'test', '.npz'), **save_outputs)
    ctime = time.time()
    self.log(teststr.format(*strargs, ctime-start_time))
    avg_loss = summed_vals['total_loss'] / self._test_values['nsteps']
    if avg_loss < self.min_test_value:
      fname = os.path.join(self.save_dir, 'best-loss-model')
      self.test_saver.save(sess, fname)


  def train(self):
    dataset = data_util.datasets.get_dataset(opts)
    with tf.device('/cpu:0'):
      sample = dataset.load_batch('train')
      test_sample = dataset.load_batch('test')
    # Build network
    network = model.get_network(opts, self.arch)
    # Get training output
    out_graph = network(sample)
    output_sim = self.get_output_sim(out_graph)
    loss = self.get_loss(sample, output_sim, test_mode=False, name='train')
    train_op = self.get_train_op(loss)
    # Get test output
    self.build_test(test_sample, network)
    # Start training
    trainstr = "local step {}: loss = {} ({:.04} sec/step, time {:.04})"
    step = 0
    tf.logging.set_verbosity(tf.logging.INFO)
    with self.build_session() as sess:
      raw_sess = sess.raw_session()
      stime = time.time()
      ctime = stime
      ttime = stime
      # while step != train_steps and ctime - stime <= train_time:
      while self.keep_training(step, ctime - stime):
        step_time = time.time()
        _, loss_ = sess.run([ train_op, loss ])
        ctime = time.time()
        if (step % opts.log_steps) == 0:
          self.log(trainstr.format(step, loss_, ctime-step_time, ctime-stime))
        if self.test_now(step, ctime - ttime):
          self.run_test(raw_sess)
          ttime = time.time()
        step += 1


if __name__ == "__main__":
  # sys.argv.extend(['--dataset', 'synth3view',
  #                  '--architecture', 'basic0',
  #                  '--datasets_dir', '/data',
  #                  '--rome16k_dir', '/mount/data/Rome16K',
  #                  '--test_freq_steps', '300',
  #                  '--batch_size', '32',
  #                  '--save_dir', 'save/testing'])
  opts = options.get_opts()
  trainer = Trainer(opts)
  trainer.train()


