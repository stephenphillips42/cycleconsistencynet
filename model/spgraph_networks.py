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
                      activate_final=True,
                      regularizers=self.regularizers,
                      initializers=self.initializers,
                      custom_getter=self.custom_getter,
                      use_bias=self.use_bias,
                      activation=tfutils.get_tf_activ(self.arch.activ))

  # Simple layers
  def LinearFinal(self, name):
     return snt.Linear(self.final_embedding_dim,
                       regularizers=self.regularizers,
                       initializers=self.initializers,
                       custom_getter=self.custom_getter,
                       use_bias=self.use_bias,
                       name=name)


class GraphBasicNetwork(MyGraphNetwork):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super().__init__(opts, arch,
                     use_bias=use_bias,
                     initializers=initializers,
                     regularizers=regularizers,
                     custom_getter=custom_getter,
                     name=name)
    self._nlayers = len(arch.layer_lens)
    self._layer_lens = arch.layer_lens
    self.normalize_emb = arch.normalize_emb
    #     graph_nets.modules.InteractionNetwork(
    #       edge_model_fn=self.SingleLayerMLP([ layer_len ]),
    #       node_model_fn=self.SingleLayerMLP([ layer_len ]),
    #       reducer=tf.unsorted_segment_mean,
    #       name="layer",
    #     )
    with self._enter_variable_scope():
      self._layers = [
        graph_nets.modules.InteractionNetwork(
          edge_model_fn=lambda: snt.Linear(layer_len,
                      regularizers=self.regularizers,
                      initializers=self.initializers,
                      custom_getter=self.custom_getter,
                      use_bias=self.use_bias),
          node_model_fn=lambda: snt.Linear(layer_len,
                      regularizers=self.regularizers,
                      initializers=self.initializers,
                      custom_getter=self.custom_getter,
                      use_bias=self.use_bias),
          reducer=tf.unsorted_segment_mean,
          name="layer",
        )
        for layer_len in arch.layer_lens
      ]
      self.final_layer = self.LinearFinal(name="final_block") 

  def _build(self, graph):
    """Applying this graph network to sample"""
    ingraph = graph
    for layer in self._layers:
      graph = layer(graph)
    graph = graph.replace(nodes=self.final_layer(graph.nodes))
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    return graph


class GraphSkipNetwork(MyGraphNetwork):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super().__init__(opts, arch,
                     use_bias=use_bias,
                     initializers=initializers,
                     regularizers=regularizers,
                     custom_getter=custom_getter,
                     name=name)
    self._nlayers = len(arch.layer_lens)
    self._layer_lens = arch.layer_lens
    self.normalize_emb = arch.normalize_emb
    with self._enter_variable_scope():
      self._layers = [
        graph_nets.modules.InteractionNetwork(
          edge_model_fn=self.SkipMLP(layer_len),
          node_model_fn=self.SkipMLP(layer_len),
          reducer=tf.unsorted_segment_mean,
          name="layer",
        )
        for layer_len in arch.layer_lens
      ]
      self.final_layer = self.LinearFinal(name="final_block") 

  def _build(self, graph):
    """Applying this graph network to sample"""
    ingraph = graph
    for layer in self._layers:
      graph = layer(graph)
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    graph = graph.replace(nodes=self.final_layer(graph.nodes))
    return graph

class GraphSkipHopNetwork(MyGraphNetwork):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super().__init__(opts, arch,
                     use_bias=use_bias,
                     initializers=initializers,
                     regularizers=regularizers,
                     custom_getter=custom_getter,
                     name=name)
    self._nlayers = len(arch.layer_lens)
    self._layer_lens = arch.layer_lens
    self._hop_layers = arch.hop_layers
    self._nhops = len(arch.hop_layers)
    self.normalize_emb = arch.normalize_emb
    with self._enter_variable_scope():
      self._layers = [
        graph_nets.modules.InteractionNetwork(
          edge_model_fn=self.SkipMLP(layer_len),
          node_model_fn=self.SkipMLP(layer_len),
          reducer=tf.unsorted_segment_mean,
          name="layer",
        )
        for layer_len in arch.layer_lens
      ]
      self.final_layer = self.LinearFinal(name="final_block") 
      self._hops = {
        hop_idx : self.GraphLinear(self._layer_lens[hop_idx], name="hop")
        for hop_idx in self._hop_layers
      }

  def _build(self, graph):
    """Applying this graph network to sample"""
    ingraph = graph
    prev_graph = graph
    for i, layer in enumerate(self._layers):
      graph_next = layer(graph_prev)
      if i in self._hop_layers:
        hop_layer = self._hop[i](ingraph)
        graph_next += hop_layer
      graph_prev = graph_next
    graph = graph_prev.replace(nodes=self.final_layer(graph_prev.nodes))
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    return graph

