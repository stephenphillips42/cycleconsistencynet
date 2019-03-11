# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import graphnet_networks

def get_network(opts, arch):
  regularizers = None
  if arch.architecture_type == 'basic':
    network = graphnet_networks.GraphBasicNetwork(opts, arch)
  elif arch.architecture_type == 'skip':
    network = graphnet_networks.GraphSkipNetwork(opts, arch)
  elif arch.architecture_type == 'skiphop':
    network = graphnet_networks.GraphSkipHopNetwork(opts, arch)
  else:
    print("ERROR: {} not implemented yet".format(arch.architecture_type))
    sys.exit(1)
  return network

if __name__ == "__main__":
  import data_util
  opts = options.get_opts()






