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

from model import mygraphnetwork
from model import layers

class GraphBasicNetwork(mygraphnetwork.MyGraphNetwork):
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
          edge_model_fn=self.SingleLayerMLP([ layer_len ]),
          node_model_fn=self.SingleLayerMLP([ layer_len ]),
          reducer=tf.unsorted_segment_mean,
          name="layer",
        )
        for layer_len in arch.layer_lens
      ]
      self.final_linear = self.GraphLinear(self.final_embedding_dim,
                                          name="final_block")

  def _build(self, sample):
    """Applying this graph network to sample"""
    graph = sample['graph']
    ingraph = graph
    for layer in self._layers:
      graph = layer(graph)
    graph = graph.replace(nodes=self.final_linear(graph.nodes))
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    return graph


class GraphSkipNetwork(mygraphnetwork.MyGraphNetwork):
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
      self.final_layer = self.GraphLinear(self.final_embedding_dim,
                                          name="final_block")

  def _build(self, sample):
    """Applying this graph network to sample"""
    graph = sample['graph']
    ingraph = graph
    for layer in self._layers:
      graph = layer(graph)
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    graph = graph.replace(nodes=self.final_layer(graph.nodes))
    return graph

# TODO: Describe difference between 'hop' and 'skip'
# As of now: Hop: Linear connection from intermediate layers to later layers
# As of now: Skip: Linear connection from first layer to later layers
# Probably should switch the two... it makes way more sense
class GraphSkipHopNetwork(mygraphnetwork.MyGraphNetwork):
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
    self._skip_layers = arch.skip_layers
    self._nskips = len(arch.skip_layers)
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
      self.final_linear = self.GraphLinear(self.final_embedding_dim,
                                          name="final_block")
      self._skips = {
        skip_idx : self.GraphLinear(self._layer_lens[skip_idx], name="skip")
        for skip_idx in self._skip_layers
      }
      self._hops = {
        skip_idx : self.GraphLinear(self._layer_lens[skip_idx], name="hop")
        for skip_idx in self._skip_layers[1:]
      }

  def _build(self, sample):
    """Applying this graph network to sample"""
    ingraph = sample['graph']
    graph_prev = ingraph
    last_skip = None
    for i, layer in enumerate(self._layers):
      graph_next = layer(graph_prev)
      if i in self._hops:
        hop_layer = self._hops[i](last_skip.nodes)
        graph_next = graph_next.replace(nodes=graph_next.nodes + hop_layer)
      if i in self._skips:
        skip_layer = self._skips[i](ingraph.nodes)
        graph_next = graph_next.replace(nodes=graph_next.nodes + skip_layer)
        last_skip = graph_next
      graph_prev = graph_next
    graph = graph_prev.replace(nodes=self.final_linear(graph_prev.nodes))
    if self.normalize_emb:
      norm_nodes = tf.nn.l2_normalize(graph.nodes, axis=1)
      graph = graph.replace(nodes=norm_nodes)
    return graph

