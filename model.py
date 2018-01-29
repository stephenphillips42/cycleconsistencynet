# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
from tqdm import tqdm

import tensorflow as tf
import sonnet as snt

import nputils
import options

# TODO: Better documentation
class DenseGraphLayer(snt.AbstractModule):
  """Implements graph layer from GCN, using dense matrices."""
  def __init__(self, n_in, n_out, name="dense_graph_layer"):
    """Build graph layer and specify input and output size."""
    super(DenseGraph, self).__init__(name=name)
    self._in = n_in
    self._out = n_out

  def _build(self, inputs):
    """Compute next layer of embeddings."""



