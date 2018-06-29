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
  # Logging options
  parser.add_argument('--debug',
                      default=False,
                      type=str2bool,
                      help='Run in debug mode')
  parser.add_argument('--print_freq',
                      default=5,
                      type=int,
                      help='Print every n batches')
  parser.add_argument('--viewer_size',
                      default=8,
                      type=int,
                      help='Run in debug mode')

  # Directory and dataset options
  parser.add_argument('--save_dir',
                      default=None,
                      help='Directory to save out logs and checkpoints')
  parser.add_argument('--data_dir',
                      default='/NAS/data/stephen/',
                      help='Directory for saving/loading dataset')
  dataset_choices = [
    'synth_small', 'synth_3view', 'synth_4view',
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
  parser.add_argument('--use_descriptors',
                      default=True,
                      type=str2bool,
                      help='Dimention of the descriptors of the points')
  parser.add_argument('--load_data',
                      default=True,
                      type=str2bool,
                      help='Load data or just generate it on the fly. '
                           'Generating slower but you get infinite data.')

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
  parser.add_argument('--activation_type',
                      default=None,
                      choices=['relu','leakyrelu','tanh', 'elu'],
                      help='What type of activation to use')

  # Machine learning parameters
  parser.add_argument('--run_time',
                      default=-1,
                      type=int,
                      help='Time in minutes the training procedure runs')
  parser.add_argument('--num_epochs',
                      default=-1,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
  parser.add_argument('--use_unsupervised_loss',
                      default=False,
                      type=str2bool,
                      help='Use true adjacency or noisy one in loss')
  parser.add_argument('--embedding_offset',
                      default=10,
                      type=int,
                      help='Offset used for computing the loss on negative examples')
  parser.add_argument('--embedding_distance_weight',
                      default=0.01,
                      type=float,
                      help='Weight put on embedding distance')
  parser.add_argument('--weight_decay',
                      default=4e-5,
                      type=float,
                      help='Weight decay regularization')
  parser.add_argument('--weight_l1_decay',
                      default=3e-5,
                      type=float,
                      help='L1 weight decay regularization')
  parser.add_argument('--optimizer_type',
                      default='sgd',
                      choices=['adam','adadelta','momentum','sgd'],
                      help='Optimizer type for adaptive learning methods')
  parser.add_argument('--learning_rate',
                      default=1e-3,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--momentum',
                      default=0.6,
                      type=float,
                      help='Learning rate for gradient descent')
  parser.add_argument('--learning_rate_decay_type',
                      default='exponential',
                      choices=['fixed','exponential','polynomial'],
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

  # Tensorflow technical options
  parser.add_argument('--full_tensorboard',
                      default=True,
                      type=str2bool,
                      help='Display everything on tensorboard?')
  # parser.add_argument('--test_check_freq',
  #                     default=4000,
  #                     type=int,
  #                     help='Number of steps between running loss on test set')
  parser.add_argument('--num_readers',
                      default=3,
                      type=int,
                      help='Number of parallel threads to read in the dataset')
  parser.add_argument('--num_preprocessing_threads',
                      default=1,
                      type=int,
                      help='How many threads to preprocess data i.e. data augmentation')
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
  parser.add_argument('--save_interval_steps',
                      default=4000,
                      type=int,
                      help='How frequently in seconds we save our model while training')
  parser.add_argument('--shuffle_data',
                      default=True,
                      type=str2bool,
                      help='Shuffle the dataset or no?')

  # Debugging options
  parser.add_argument('--verbose',
                      default=False,
                      type=str2bool,
                      help='Print out everything')
  parser.add_argument('--debug_index',
                      default=1,
                      type=int,
                      help='Test data index to experiment with')
  parser.add_argument('--debug_data_dir',
                      default='logs/np_datasets',
                      help='Test data directory to experiment with')
  parser.add_argument('--debug_log_dir',
                      default='logs',
                      help='Logs to experiment with')
  plot_options = [ 'none', 'plot', 'save', 'baseline' ]
  parser.add_argument('--debug_plot',
                      default=plot_options[0],
                      choices=plot_options,
                      help='Plot things in experiment')


  opts = parser.parse_args()

  # Get save directory default
  if opts.save_dir is None:
    save_idx = 0
    while os.path.exists('save/save-{:03d}'.format(save_idx)):
      save_idx += 1
    opts.save_dir = 'save/save-{:03d}'.format(save_idx)

  # Determine dataset
  dataset_params = types.SimpleNamespace(
    data_dir='/NAS/data/stephen/{}'.format(opts.dataset),
    sizes={ 'train': 40000, 'test': 3000 },
    fixed_size=True,
    views=[3],
    points=[25],
    points_scale=1,
    knn=8,
    scale=3,
    sparse=False,
    soft_edges=False,
    descriptor_dim=12,
    descriptor_var=1.0,
    descriptor_noise_var=0,
    noise_level=0.1,
    num_repeats=1,
    dtype='float32')
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
      activ='relu',
      normalize_emb=True)
  elif opts.architecture in ['vanilla0', 'skip0']:
    arch = arch_params(
      nlayers=5,
      layer_lens=[ 2**min(6+k,10) for k in range(5) ],
      activ='relu',
      normalize_emb=True)
  elif opts.architecture in ['vanilla1', 'skip1']:
    arch = arch_params(
      nlayers=6,
      layer_lens=[ 2**min(6+k,10) for k in range(6) ],
      activ='relu',
      normalize_emb=True)
  setattr(opts, 'arch', arch)

  # Post processing
  if arch.normalize_emb:
    opts.embedding_offset = 1
  # Save out options
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'w') as yml:
    yml.write(yaml.dump(opts.__dict__))

  # Finished, return options
  return opts



