# -*- coding: utf-8 -*-
import numpy as np
import os

from data_util import synth_dataset
from data_util import real_dataset
from data_util import spgraph_dataset


def get_dataset(opts):
  """Getting the dataset with all the correct attributes"""
  dataset_type = opts.dataset_params.dataset_type
  if dataset_type in 'synth':
    return spgraph_dataset.SpSynthGraphDataset(opts, opts.dataset_params)
  else:
    print("ERROR: Dataset type {} not implemented yet".format(dataset_type))
    sys.exit(1)

