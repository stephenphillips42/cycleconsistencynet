# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import types
import yaml

arch_params = collections.namedtuple('arch_params', [
  'nlayers', 'layer_lens', 'activ', 'normalize_emb'
])
# synth_dataset_params_vars = [
#   'data_dir', 'sizes', 'dtype', # Meta-parameters
#   'fixed_size', 'views', 'points', # Graph
#   'points_scale', 'knn', 'scale', 'sparse', 'soft_edges', 'use_descriptors',
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
    'synth_small','synth_3view', 'synth_4view',
  ]
  # 'synth_noise1', 'synth_noise2'
  parser.add_argument('--dataset',
                      default=True,
                      choices=dataset_choices,
                      help='Choose which dataset to use')

  # Dataset options
  parser.add_argument('--num_gen_train',
                      default=8000,
                      type=int,
                      help='Number of training samples to generate.')
  parser.add_argument('--num_gen_test',
                      default=2000,
                      type=int,
                      help='Number of testing samples to generate.')
  parser.add_argument('--dtype',
                      default='float32',
                      help='dtype to save dataset as (float{16,32,64})')
  parser.add_argument('--fixed_size',
                      default=True,
                      type=str2bool,
                      help='Determine weather or not dataset has fixed or stochastic size')
  parser.add_argument('--min_views',
                      default=25,
                      type=int,
                      help='Minimum number of viewpoints in the graph')
  parser.add_argument('--max_views',
                      default=30,
                      type=int,
                      help='Maximum number of viewpoints in the graph')
  parser.add_argument('--min_points',
                      default=9,
                      type=int,
                      help='Minimum number of points in the world seen in '
                           'the images')
  parser.add_argument('--max_points',
                      default=15,
                      type=int,
                      help='Maximum number of points in the world seen in '
                           'the images')
  parser.add_argument('--points_scale',
                      default=1,
                      type=float,
                      help='Scale of the points in the world')
  parser.add_argument('--knn',
                      default=8,
                      type=int,
                      help='number of neighbors used in graph')
  parser.add_argument('--scale',
                      default=3,
                      type=int,
                      help='Distance from center of sphere to cameras')
  parser.add_argument('--sparse',
                      default=False,
                      type=str2bool,
                      help='Use full graph')
  parser.add_argument('--soft_edges',
                      default=False,
                      type=str2bool,
                      help='Use soft or hard correspondences')
  parser.add_argument('--use_descriptors',
                      default=True,
                      type=str2bool,
                      help='Dimention of the descriptors of the points')
  parser.add_argument('--descriptor_dim',
                      default=12,
                      type=int,
                      help='Dimention of the descriptors of the points')
  parser.add_argument('--descriptor_var',
                      default=1.0,
                      type=float,
                      help='Variance of the true descriptors')
  parser.add_argument('--descriptor_noise_var',
                      default=0,
                      type=float,
                      help='Variance of the noise of the descriptors '
                           'of the projected points')

  # Architecture parameters
  arch_choices = [
    'vanilla', 'vanilla_0', 'vanilla_1', 
  ]
  #  'skip_0', 'skip_1'
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
  parser.add_argument('--num_epochs',
                      default=400,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
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
  parser.add_argument('--test_check_freq',
                      default=4000,
                      type=int,
                      help='Number of steps between running loss on test set')
  parser.add_argument('--num_readers',
                      default=3,
                      type=int,
                      help='Number of parallel threads to read in the dataset')
  parser.add_argument('--num_preprocessing_threads',
                      default=1,
                      type=int,
                      help='How many threads to preprocess data i.e. data augmentation')
  parser.add_argument('--save_summaries_secs',
                      default=300,
                      type=int,
                      help='How frequently we save our model while training')
  parser.add_argument('--save_interval_secs',
                      default=600,
                      type=int,
                      help='How frequently we save our model while training')
  parser.add_argument('--log_steps',
                      default=1,
                      type=int,
                      help='How frequently we save our model while training')
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
  parser.add_argument('--debug_dir',
                      default='logs',
                      help='Test data directory to experiment with')
  parser.add_argument('--debug_plot',
                      default=False,
                      type=str2bool,
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
    data_dir=opts.data_dir,
    sizes={ 'train': 8000, 'test': 2000 },
    dtype='float32',
    fixed_size=False,
    views=[25, 30],
    points=[9, 15],
    points_scale=1,
    knn=8,
    scale=3,
    sparse=False,
    soft_edges=False,
    use_descriptors=True,
    descriptor_dim=12,
    descriptor_var=1.0,
    descriptor_noise_var=0)
  if opts.dataset == 'synth_3view':
    dataset_params.data_dir = '/NAS/data/stephen/synth_3view'
    dataset_params.fixed_size=True
    dataset_params.views=[3]
    dataset_params.points=[25]
    dataset_params.sizes = { 'train': 40000, 'test': 3000 }
  elif opts.dataset == 'synth_small':
    dataset_params.data_dir = '/NAS/data/stephen/synth_small'
    dataset_params.fixed_size=True
    dataset_params.views=[3]
    dataset_params.points=[25]
    dataset_params.sizes = { 'train': 400, 'test': 300 }
  elif opts.dataset == 'synth_4view':
    dataset_params.data_dir = '/NAS/data/stephen/synth_4view'
    dataset_params.fixed_size=True
    dataset_params.views=[4]
    dataset_params.points=[25]
    dataset_params.sizes = { 'train': 40000, 'test': 3000 }
  opts.data_dir = dataset_params.data_dir
  setattr(opts, 'dataset_params', dataset_params)

  # Set up architecture
  arch = None 
  if opts.architecture == 'vanilla':
    arch = arch_params(
      nlayers=5,
      layer_lens=[ 2**min(5+k,9) for k in range(5) ],
      activ='relu',
      normalize_emb=True)
  elif opts.architecture == 'vanilla_0':
    arch = arch_params(
      nlayers=5,
      layer_lens=[ 2**min(6+k,10) for k in range(5) ],
      activ='relu',
      normalize_emb=True)
  elif opts.architecture == 'vanilla_1':
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



