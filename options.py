# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import argparse
import collections
import types
import yaml
import re

import myutils

class YamlLoader(object):
  def __init__(self, file_load):
    if not os.path.exists(file_load):
      errstr = "ERROR: File {} does not exist, cannot find configuration"
      print(errstr.format(file_load))
      sys.exit(1)
    with open(file_load, 'r') as f:
      yamldict = yaml.load(f)
    self.__dict__.update(yamldict)
    self.yamldict = yamldict

class DatasetParams(YamlLoader):
  def __init__(self, opts):
    fname = os.path.join('config', 'datasets', opts.dataset + '.yaml')
    super().__init__(fname)
    self.data_dir=os.path.join(opts.datasets_dir, opts.dataset)
    default_sizes={ 'train': 40000, 'test': 3000 }
    if 'sizes' not in self.__dict__:
      self.sizes = default_sizes
    else:
      for k in default_sizes:
        if k not in self.sizes:
          self.sizes[k] = default_sizes[k]

config_len = len('config/datasets/')
yaml_len = len('.yaml')
dataset_files = sorted(glob.glob(os.path.join('config','datasets','*.yaml')))
dataset_choices = [ x[config_len:-yaml_len] for x in dataset_files ]

class ArchParams(YamlLoader):
  def __init__(self, opts):
    fname = os.path.join('config', 'architectures', opts.architecture+'.yaml')
    super().__init__(fname)

arch_files = sorted(glob.glob(os.path.join('config','architectures','*.yaml')))
config_len = len('config/architectures/')
arch_choices = [ x[config_len:-yaml_len] for x in arch_files ]

loss_types = [ 'l2', 'l1', 'l1l2' ]
optimizer_types = ['sgd','adam','adadelta','momentum','adamw']
lr_decay_types = ['exponential','fixed','polynomial']


def get_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  # Directory and dataset options
  parser.add_argument('--save_dir',
                      default='',
                      help='Directory to save out logs and checkpoints')
  parser.add_argument('--checkpoint_start_dir',
                      default=None,
                      help='Place to load from if not loading from save_dir')
  parser.add_argument('--data_dir',
                      default='/NAS/data/stephen/',
                      help='Directory for saving/loading dataset')
  parser.add_argument('--rome16k_dir',
                      default='/NAS/data/stephen/Rome16K',
                      help='Directory for storing Rome16K dataset (Very specific)')
  # 'synth_noise1', 'synth_noise2'
  parser.add_argument('--dataset',
                      default=dataset_choices[0],
                      choices=dataset_choices,
                      help='Choose which dataset to use')
  parser.add_argument('--datasets_dir',
                      default='/NAS/data/stephen',
                      help='Directory where all the datasets are')
  parser.add_argument('--shuffle_data',
                      default=True,
                      type=myutils.str2bool,
                      help='Shuffle the dataset or no?')

  # Architecture parameters
  parser.add_argument('--architecture',
                      default=arch_choices[0],
                      choices=arch_choices,
                      help='Network architecture to use')
  parser.add_argument('--final_embedding_dim',
                      default=None,
                      type=int,
                      help='Dimensionality of the output')

  # Machine learning parameters
  parser.add_argument('--batch_size',
                      default=32,
                      type=int,
                      help='Size for batches')
  # TODO: Combine next two to add post-processing option
  parser.add_argument('--use_clamping',
                      default=False,
                      type=myutils.str2bool,
                      help='Use clamping to [0, 1] on the output similarities')
  parser.add_argument('--use_abs_value',
                      default=False,
                      type=myutils.str2bool,
                      help='Use absolute value on the output similarities')
  parser.add_argument('--loss_type',
                      default=loss_types[0],
                      choices=loss_types,
                      help='Loss function to use for training')
  parser.add_argument('--reconstruction_loss',
                      default=1.0,
                      type=float,
                      help='Use true adjacency or noisy one in loss')
  parser.add_argument('--geometric_loss',
                      default=-1,
                      type=float,
                      help='Weight to use on the geometric loss')
  parser.add_argument('--weight_decay',
                      default=4e-5,
                      type=float,
                      help='Weight decay regularization')
  parser.add_argument('--weight_l1_decay',
                      default=0,
                      type=float,
                      help='L1 weight decay regularization')
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
  parser.add_argument('--learning_rate_continuous',
                      default=False,
                      type=myutils.str2bool,
                      help='Number of epochs before learning rate decay')
  parser.add_argument('--learning_rate_decay_epochs',
                      default=4,
                      type=float,
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
                      default=-1,
                      type=int,
                      help='Minutes between running loss on test set.'
                           'If less than zero, does not check time for testing')
  parser.add_argument('--test_freq_steps',
                      default=-1,
                      type=int,
                      help='Number of steps between running loss on test set'
                           'If less than zero, does not check steps for testing')

  # Logging options
  parser.add_argument('--verbose',
                      default=False,
                      type=myutils.str2bool,
                      help='Print out everything')
  parser.add_argument('--save_summaries_secs',
                      default=120,
                      type=int,
                      help='How frequently in seconds we save training summaries')
  parser.add_argument('--save_interval_secs',
                      default=600,
                      type=int,
                      help='Frequency in seconds to save model while training')
  parser.add_argument('--log_steps',
                      default=5,
                      type=int,
                      help='How frequently we print training loss')

  # Debugging options
  parser.add_argument('--debug',
                      default=False,
                      type=myutils.str2bool,
                      help='Run in debug mode')


  opts = parser.parse_args()

  ##### Post processing
  # Dataset
  dataset_params = DatasetParams(opts)
  opts.data_dir = dataset_params.data_dir
  if opts.final_embedding_dim is None:
    opts.final_embedding_dim = dataset_params.points[-1]
  setattr(opts, 'dataset_params', dataset_params)

  # Set up architecture
  arch = ArchParams(opts)
  setattr(opts, 'arch', arch)

  # Save out directory
  if opts.save_dir == '':
    print(''.join([ '=' ] * 20), file=sys.stderr)
    print('WARNING: save_dir not set, '
          'going to default /tmp/discard_dir', file=sys.stderr)
    print(''.join([ '=' ] * 20), file=sys.stderr)
    opts.save_dir = '/tmp/discard_dir'
    if os.path.exists(opts.save_dir):
      import shutil
      shutil.rmtree(opts.save_dir)
    os.makedirs(opts.save_dir)
  elif not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)

  # Checkpoint loading
  if opts.checkpoint_start_dir and not os.path.exists(opts.checkpoint_start_dir):
    print("ERROR: Checkpoint Directory {} does not exist".format(opts.checkpoint_start_dir))
    return

  yaml_fname = os.path.join(opts.save_dir, 'options.yaml')
  if not os.path.exists(yaml_fname):
    with open(yaml_fname, 'w') as yml:
      yml.write(yaml.dump(opts.__dict__))

  # Finished, return options
  return opts

def parse_yaml_opts(opts):
  with open(os.path.join(opts.save_dir, 'options.yaml'), 'r') as yml:
    yaml_opts = yaml.load(yml)
  opts.__dict__.update(yaml_opts)
  return opts


