# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import yaml

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
  parser.add_argument('--dataset',
                      default=True,
                      choices=['cycle_small', 'cycle_large', 'custom'],
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

  ## TODO: Implement these for graphs
  # Architecture parameters
  # parser.add_argument('--network_type',
  #                     default='heading_network',
  #                     choices=['heading_network'],
  #                     help='Network architecture to use')
  parser.add_argument('--nlayers',
                      default=4,
                      type=int,
                      help='Number of layers in the architecture')
  parser.add_argument('--final_embedding_dim',
                      default=12,
                      type=int,
                      help='Dimensionality of the output')
  parser.add_argument('--nclasses',
                      default=24,
                      type=int,
                      help='Number of classes')
  parser.add_argument('--activation_type',
                      default=None,
                      choices=['relu','leakyrelu','tanh','relusq'],
                      help='What type of activation to use')
  parser.add_argument('--use_fully_connected',
                      default=False,
                      type=str2bool,
                      help='Use fully connected layer at the end')
  parser.add_argument('--fully_connected_size',
                      default=1024,
                      type=int,
                      help='Size of the last fully connected layer')
  parser.add_argument('--use_batch_norm',
                      default=True,
                      type=str2bool,
                      help='Decision whether to use batch norm or not')
  parser.add_argument('--architecture',
                      default=None,
                      help='Helper variable for building the architecture type from network_type')
  parser.add_argument('--normalize_embedding',
                      default=False,
                      type=str2bool,
                      help='Helper variable for building the architecture type from network_type')

  # Machine learning parameters
  parser.add_argument('--num_epochs',
                      default=400,
                      type=int,
                      help='Number of epochs to run training')
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
  # parser.add_argument('--noise_level',
  #                     default=1e-2,
  #                     type=float,
  #                     help='Standard devation of white noise to add to input')
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
                      default='adam',
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
                      default=False,
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

  opts = parser.parse_args()

  # Get save directory default
  if opts.save_dir is None:
    save_idx = 0
    while os.path.exists('save/save-{:03d}'.format(save_idx)):
      save_idx += 1
    opts.save_dir = 'save/save-{:03d}'.format(save_idx)

  # Determine dataset
  setattr(opts, 'load', True)
  if opts.dataset == 'cycle_large':
    opts.fixed_size=True
    opts.max_views=3
    opts.max_points=25
    opts.num_gen_test=3000
    opts.num_gen_train=40000
    opts.data_dir = '/NAS/data/stephen/cycle_large'
  elif opts.dataset == 'cycle_small':
    opts.fixed_size=True
    opts.max_views=3
    opts.max_points=25
    opts.num_gen_test=300
    opts.num_gen_train=400
    opts.data_dir = '/NAS/data/stephen/cycle_small'
  elif opts.dataset == 'custom':
    opts.load = False
  setattr(opts, 'sample_sizes', {'train': opts.num_gen_train,
                                 'test': opts.num_gen_test})
  # Post processing
  if opts.normalize_embedding:
    opts.embedding_offset = 1
  # Save out options
  if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'w') as yml:
    yml.write(yaml.dump(opts.__dict__))

  # Finished, return options
  return opts



