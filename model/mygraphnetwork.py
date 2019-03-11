# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import graph_nets # import blocks

import tfutils
import myutils
import options

from model import layers

class MyGraphNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super().__init__(custom_getter=custom_getter, name=name)
    self.opts = opts
    self.arch = arch
    self.use_bias = use_bias
    self.regularizers = self.build_regularizers()
    self.initializers = self.build_initializers()
    self.custom_getter = custom_getter
    self.final_embedding_dim = opts.final_embedding_dim

  def build_initializers(self, initializers=None):
    """Get initializers based on opts"""
    # For now we just stick with the defaults
    return initializers

  def build_regularizers(self, regularizers=None):
    """Get regularizers based on opts"""
    if regularizers is not None:
      return regularizers
    reg = { 'w' : None }
    # Set up weight decay (and/or weight l1 decay)
    wd2 = self.opts.weight_decay
    wd1 = self.opts.weight_l1_decay
    # We use gradient descent optimizer separate from regular optimization
    #   process, so we don't set weight decay terms here
    if wd2 > 0 and wd1 <= 0:
      reg['w'] = tf.contrib.layers.l2_regularizer(1.0)
    elif wd2 <= 0 and wd1 > 0:
      reg['w'] = tf.contrib.layers.l1_regularizer(1.0)
    elif wd2 > 0 and wd1 > 0:
      reg['w'] = tf.contrib.layers.l1_l2_regularizer(wd1/wd2,1.0)
    # Bias has no regularization
    if self.use_bias:
      bias_reg = tf.contrib.layers.l2_regularizer(0.0)
      reg['b'] = bias_reg
    return reg

  # Various layer builders
  def SingleLayerMLP(self, layer_lens):
     return lambda: snt.nets.MLP(
                      layer_lens,
                      activate_final=True,
                      regularizers=self.regularizers,
                      initializers=self.initializers,
                      custom_getter=self.custom_getter,
                      use_bias=self.use_bias,
                      activation=tfutils.get_tf_activ(self.arch.activ))

  def SkipMLP(self, layer_len):
     return lambda: layers.SkipLayer(
                      layer_len,
                      activation=self.arch.activ,
                      regularizers=self.regularizers,
                      initializers=self.initializers,
                      custom_getter=self.custom_getter,
                      use_bias=self.use_bias)

  # Simple layers
  def GraphLinear(self, layer_len, name):
     return snt.Linear(layer_len,
                       regularizers=self.regularizers,
                       initializers=self.initializers,
                       custom_getter=self.custom_getter,
                       use_bias=self.use_bias,
                       name=name)


