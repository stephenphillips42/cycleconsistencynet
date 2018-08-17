# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import networks

def get_regularizers(opts):
  regularizer_fn = None
  if opts.weight_decay <= 0 and opts.weight_l1_decay <= 0:
    return None
  elif opts.weight_decay > 0 and opts.weight_l1_decay <= 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l2_regularizer(1.0)
  elif opts.weight_decay <= 0 and opts.weight_l1_decay > 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l1_regularizer(1.0)
  elif opts.weight_decay <= 0 and opts.weight_l1_decay > 0:
    regularizer_fn = \
        lambda r_l2, r_l1: tf.contrib.layers.l1_l2_regularizer(r_l1/r_l2, 1.0)
  if opts.architecture in ['vanilla', 'vanilla0', 'vanilla1']:
    return {
          "w" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "b" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay),
      }
  elif opts.architecture in ['skip', 'skip0', 'skip1', 'longskip0', 'longskip1']:
    return {
          "w" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "u" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "b" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay),
          "c" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay),
      }
  elif opts.architecture in ['attn0', 'attn1', 'attn2', \
                             'spattn0', 'spattn1', 'spattn2']:
    return {
          "w" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "u" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "f1" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "f2" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "b" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay),
          "c" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay),
          "d1" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
          "d2" : regularizer_fn(opts.weight_decay, opts.weight_l1_decay), 
      }

def get_network(opts, arch):
  regularizers = None
  if opts.architecture in ['vanilla', 'vanilla0', 'vanilla1']:
    network = networks.GraphConvLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['skip', 'skip0', 'skip1']:
    network = networks.GraphSkipLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['longskip0', 'longskip1']:
    network = networks.GraphLongSkipLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  elif opts.architecture in ['attn0', 'attn1', 'attn2', \
                             'spattn0', 'spattn1', 'spattn2']:
    network = networks.GraphAttentionLayerNetwork(
                    opts,
                    arch,
                    regularizers=get_regularizers(opts))
  return network

if __name__ == "__main__":
  import data_util
  opts = options.get_opts()






