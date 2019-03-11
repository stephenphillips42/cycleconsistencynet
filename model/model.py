# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import graph_net_networks
from model import adjmat_networks

def get_network(opts, arch):
  regularizers = None
  if arch.architecture_type == 'basic':
    network = graph_net_networks.GraphBasicNetwork(opts, arch)
  elif arch.architecture_type == 'skip':
    network = graph_net_networks.GraphSkipNetwork(opts, arch)
  elif arch.architecture_type == 'skiphop':
    network = graph_net_networks.GraphSkipHopNetwork(opts, arch)
  elif arch.architecture_type == 'adjmat0':
    network = graph_net_networks.GraphSkipHopNetwork(opts, arch)
  elif arch.architecture_type == 'adjmat0':
    network = graph_net_networks.GraphSkipHopNetwork(opts, arch)
  # elif arch.architecture_type == 'normedhop':
  #   pass
  else:
    print("ERROR: {} not implemented yet".format(arch.architecture_type))
    sys.exit(1)
  return network

if __name__ == "__main__":
  import data_util
  opts = options.get_opts()






