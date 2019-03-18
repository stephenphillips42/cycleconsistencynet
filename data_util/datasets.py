# -*- coding: utf-8 -*-
import numpy as np
import os

# from data_util import synth_dataset
# from data_util import real_dataset
from data_util import synthgraph_dataset as synthgraph
from data_util import spreal_dataset as spreal


def get_dataset(opts):
  """Getting the dataset with all the correct attributes"""
  dataset_type = opts.dataset_params.dataset_type
  if dataset_type in 'synth':
    return synthgraph.SynthGraphDataset(opts, opts.dataset_params)
  elif dataset_type in 'synthnoise':
    return synthgraph.SynthNoiseGraphDataset(opts, opts.dataset_params)
  elif dataset_type in 'synthoutlier':
    return synthgraph.SynthOutlierGraphDataset(opts, opts.dataset_params)
  elif dataset_type in 'rome16kgeom':
    return spreal.GeomKNNRome16KDataset(opts, opts.dataset_params)
  else:
    print("ERROR: Dataset type {} not implemented yet".format(dataset_type))
    sys.exit(1)

