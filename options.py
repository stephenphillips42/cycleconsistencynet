# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import types
import yaml
import re

arch_params = collections.namedtuple('arch_params', [
  'nlayers', 'layer_lens', 'activ', 'normalize_emb'
])
# synth_dataset_params_vars = [
#   'data_dir', 'sizes', 'dtype', # Meta-parameters
#   'fixed_size', 'views', 'points', # Graph
#   'points_scale', 'knn', 'scale', 'sparse', 'soft_edges',
#   'descriptor_dim', 'descriptor_var', 'descriptor_noise_var', # Descriptor
# ]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  # Directory and dataset options
  parser.add_argument('--save_dir',
                      default=None,
                      help='Directory to save out logs and checkpoints')
  parser.add_argument('--data_dir',
                      default='/NAS/data/stephen/',
                      help='Directory for saving/loading dataset')
  dataset_choices = [
    'synth_3view', 'synth_small', 'synth_4view',
    'noise_3view',
    'noise_gauss', 'noise_symgauss',
    'noise_pairwise', 'noise_pairwise3', 'noise_pairwise5',
    'noise_largepairwise3', 'noise_largepairwise5'
  ]
  # 'synth_noise1', 'synth_noise2'
  parser.add_argument('--dataset',
                      default=dataset_choices[0],
                      choices=dataset_choices,
                      help='Choose which dataset to use')
  parser.add_argument('--datasets_dir',
                      default='/NAS/data/stephen',
                      help='Directory where all the datasets are')
  parser.add_argument('--load_data',
                      default=True,
                      type=str2bool,
                      help='Load data or just generate it on the fly. '
                           'Generating slower but you get infinite data.')
  parser.add_argument('--shuffle_data',
                      default=True,
                      type=str2bool,
                      help='Shuffle the dataset or no?')

  # Architecture parameters
  arch_choices = [
    'vanilla', 'vanilla0', 'vanilla1', 
    'skip', 'skip0', 'skip1', 
  ]
  parser.add_argument('--architecture',
                      default='vanilla',
                      choices=arch_choices,
                      help='Network architecture to use')
  parser.add_argument('--final_embedding_dim',
                      default=12,
                      type=int,
                      help='Dimensionality of the output')
  activation_types = ['relu','leakyrelu','tanh', 'elu']
  parser.add_argument('--activation_type',
                      default=activation_types[0],
                      choices=activation_types,
                      help='What type of activation to use')

  # Machine learning parameters
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
  parser.add_argument('--use_unsupervised_loss',
                      default=False,
                      type=str2bool,
                      help='Use true adjacency or noisy one in loss')
  loss_types = [ 'l2', 'bce' ]
  parser.add_argument('--loss_type',
                      default=loss_types[0],
                      choices=loss_types,
                      help='')
  parser.add_argument('--weight_decay',
                      default=4e-5,
                      type=float,
                      help='Weight decay regularization')
  parser.add_argument('--weight_l1_decay',
                      default=0,
                      type=float,
                      help='L1 weight decay regularization')
  optimizer_types = ['sgd','adam','adadelta','momentum']
  parser.add_argument('--optimizer_type',
                      default=optimizer_types[0],
                      choices=optimizer_types,
                      help='Optimizer type for adaptive learning methods')
  parser.add_argument('--learning_rate',
                      default=1e-3,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--momentum',
                      default=0.6,
                      type=float,
                      help='Learning rate for gradient descent')
  lr_decay_types = ['exponential','fixed','polynomial']
  parser.add_argument('--learning_rate_decay_type',
                      default=lr_decay_types[0],
                      choices=lr_decay_types,
                      help='Learning rate decay policy')
  parser.add_argument('--min_learning_rate',
                      default=1e-5,
                      type=float,
                      help='Minimum learning rate after decaying')
  parser.add_argument('--learning_rate_decay_rate',
                      default=0.95,
                      type=float,
                      help='Learning rate decay rate')
  parser.add_argument('--learning_rate_decay_epochs',
                      default=4,
                      type=int,
                      help='Number of epochs before learning rate decay')

  # Training options
  parser.add_argument('--train_time',
                      default=-1,
                      type=int,
                      help='Time in minutes the training procedure runs')
  parser.add_argument('--num_epochs',
                      default=-1,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--test_freq',
                      default=8,
                      type=int,
                      help='Minutes between running loss on test set')
  parser.add_argument('--test_freq_steps',
                      default=0,
                      type=int,
                      help='Number of steps between running loss on test set')
  parser.add_argument('--num_runs',
                      default=1,
                      type=int,
                      help='Number of times training runs (length determined'
                           'by run_time)')

  # Logging options
  parser.add_argument('--verbose',
                      default=False,
                      type=str2bool,
                      help='Print out everything')
  parser.add_argument('--full_tensorboard',
                      default=True,
                      type=str2bool,
                      help='Display everything on tensorboard?')
  parser.add_argument('--save_summaries_secs',
                      default=120,
                      type=int,
                      help='How frequently in seconds we save training summaries')
  parser.add_argument('--save_interval_secs',
                      default=600,
                      type=int,
                      help='How frequently in seconds we save our model while training')
  parser.add_argument('--log_steps',
                      default=5,
                      type=int,
                      help='How frequently we print training loss')

  # Debugging options
  parser.add_argument('--debug',
                      default=False,
                      type=str2bool,
                      help='Run in debug mode')
  parser.add_argument('--debug_index',
                      default=1,
                      type=int,
                      help='Test data index to experiment with')
  parser.add_argument('--debug_data_path',
                      default='test001.npz',
                      help='Path to test data to experiment with')
  parser.add_argument('--debug_log_dir',
                      default='logs',
                      help='Logs to experiment with')
  plot_options = [ 'none', 'plot', 'unsorted', 'baseline', 'random' ]
  parser.add_argument('--debug_plot',
                      default=plot_options[0],
                      choices=plot_options,
                      help='Plot things in experiment')
  parser.add_argument('--viewer_size',
                      default=8,
                      type=int,
                      help='Run in debug mode')


  opts = parser.parse_args()

  # Get save directory default
  if opts.save_dir is None:
    save_idx = 0
    while os.path.exists('save/save-{:03d}'.format(save_idx)):
      save_idx += 1
    opts.save_dir = 'save/save-{:03d}'.format(save_idx)

  # Determine dataset
  class DatasetParams(object):
    def __init__(self, opts):
      self.data_dir='{}/{}'.format(opts.datasets_dir, opts.dataset)
      self.sizes={ 'train': 40000, 'test': 3000 }
      self.fixed_size=True
      self.views=[3]
      self.points=[25]
      self.points_scale=1
      self.knn=8
      self.scale=3
      self.sparse=False
      self.soft_edges=False
      self.descriptor_dim=12
      self.descriptor_var=1.0
      self.descriptor_noise_var=0
      self.noise_level=0.1
      self.num_repeats=1
      self.dtype='float32'
  dataset_params = DatasetParams(opts)
  if opts.dataset == 'synth_3view':
    pass
  elif opts.dataset == 'noise_3view':
    dataset_params.noise_level = 0.2
  elif opts.dataset == 'synth_small':
    sizes={ 'train': 400, 'test': 300 },
  elif opts.dataset == 'synth_4view':
    pass
  elif opts.dataset == 'noise_gauss':
    dataset_params.noise_level = 0.1
  elif opts.dataset == 'noise_symgauss':
    dataset_params.noise_level = 0.1
    dataset_params.num_repeats = 1
  elif 'noise_pairwise' in opts.dataset:
    dataset_params.noise_level = 0.1
    num_rep = re.search(r'[0-9]+', opts.dataset)
    if num_rep:
      dataset_params.num_repeats = int(num_rep.group(0))
  elif 'noise_largepairwise' in opts.dataset:
    dataset_params.noise_level = 0.1
    dataset_params.sizes['train'] = 400000
    num_rep = re.search(r'[0-9]+', opts.dataset)
    if num_rep:
      dataset_params.num_repeats = int(num_rep.group(0))
  opts.data_dir = dataset_params.data_dir
  setattr(opts, 'dataset_params', dataset_params)

  # Set up architecture
  arch = None 
  if opts.architecture in ['vanilla', 'skip']:
    arch = arch_params(
      nlayers=5,
      layer_lens=[ 2**min(5+k,9) for k in range(5) ],
      activ=opts.activation_type,
      normalize_emb=True)
  elif opts.architecture in ['vanilla0', 'skip0']:
    arch = arch_params(
      nlayers=5,
      layer_lens=[ 2**min(6+k,10) for k in range(5) ],
      activ=opts.activation_type,
      normalize_emb=True)
  elif opts.architecture in ['vanilla1', 'skip1']:
    arch = arch_params(
      nlayers=6,
      layer_lens=[ 2**min(6+k,10) for k in range(6) ],
      activ=opts.activation_type,
      normalize_emb=True)
  setattr(opts, 'arch', arch)

  # Post processing
  if arch.normalize_emb:
    setattr(opts, 'embedding_offset', 1)
  # Save out options
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'w') as yml:
    yml.write(yaml.dump(opts.__dict__))

  # Finished, return options
  return opts



