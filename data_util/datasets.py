# -*- coding: utf-8 -*-
import numpy as np
import os

from data_util import synth_dataset
from data_util import real_dataset
from data_util import spgraph_dataset


def get_dataset(opts):
  """Getting the dataset with all the correct attributes"""
  if opts.dataset in [ 'synth_small', 'synth_3view', 'synth_4view', \
                       'synth_5view', 'synth_6view' ]:
    return synth_dataset.GraphSimDataset(opts, opts.dataset_params)
  elif 'synth_pts' in opts.dataset:
    return synth_dataset.GraphSimDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'noise_3view' ]:
    return synth_dataset.GraphSimNoisyDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'noise_gauss' ]:
    return synth_dataset.GraphSimGaussDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'noise_symgauss' ]:
    return synth_dataset.GraphSimSymGaussDataset(opts, opts.dataset_params)
  elif 'noise_largepairwise' in opts.dataset or \
        'noise_pairwise' in opts.dataset:
    return synth_dataset.GraphSimPairwiseDataset(opts, opts.dataset_params)
  elif 'noise_outlier' in opts.dataset:
    return synth_dataset.GraphSimOutlierDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'rome16kknn0' ]:
    return real_dataset.KNNRome16KDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'rome16kgeom0', 'rome16kgeom4view0' ]:
    return real_dataset.GeomKNNRome16KDataset(opts, opts.dataset_params)
  elif opts.dataset in [ 'spsynth0', 'spsynth_test' ]:
    return spgraph_dataset.SpSynthGraphDataset(opts, opts.dataset_params)

