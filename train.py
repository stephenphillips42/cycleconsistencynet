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

import torch

import nputils
import options
import model

def MyModel(torch.nn.Module):
  def __init__(self, opts):
    super(MyModel, self).__init__()
    lens = [ opts.descriptor_dim ] + \
           [ 15, 20, 25, 20 ] + \
           [ opts.final_embedding_dim ]
    self._linear = [ torch.nn.Linear(les[i], lens[i+1])
                     for i in range(len(lens)-1) ]
    self._activ = [ torch.nn.ReLu() 
                    for i in range(len(lens)-1) ]

  def forward(self, x):
    out = x[0]
    lap = x[1]
    for i in range(len(self._linear)-1):
      out = self._activ[i](torch.matmul(lap, self._linear[i](out)))
    return self._linear[-1](out)

def train(opts):
  pass

if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

