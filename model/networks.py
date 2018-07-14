# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options

from model import layers

class GraphConvLayerNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphConvLayerNetwork, self).__init__(custom_getter=custom_getter, name=name)
    self._nlayers = arch.nlayers
    self._layers = [
      layers.GraphConvLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_conv".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/embed_lin".format(name))
    ]

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    output = tf.nn.l2_normalize(output, axis=2)
    return output


class GraphSkipLayerNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphSkipLayerNetwork, self).__init__(custom_getter=custom_getter, name=name)
    self._nlayers = arch.nlayers
    final_regularizers = None
    if regularizers is not None:
      final_regularizers = { k:v
                             for k, v in regularizers.items()
                             if k in ["w", "b"] }
    self._layers = [
      layers.GraphSkipLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_skip".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=final_regularizers,
        name="{}/embed_lin".format(name))
    ]

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphAttentionLayerNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphAttentionLayerNetwork, self).__init__(custom_getter=custom_getter,
                                                     name=name)
    self._nlayers = arch.nlayers
    final_regularizers = None
    if regularizers is not None:
      final_regularizers = { k:v
                             for k, v in regularizers.items()
                             if k in ["w", "b"] }
    self._layers = [
      layers.GraphAttentionLayer(
        output_size=layer_len,
        activation=arch.activ,
        initializers=initializers,
        regularizers=regularizers,
        name="{}/graph_attn".format(name))
      for layer_len in arch.layer_lens
    ] + [
      layers.EmbeddingLinearLayer(
        output_size=opts.final_embedding_dim,
        initializers=initializers,
        regularizers=final_regularizers,
        name="{}/embed_lin".format(name))
    ]

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    output = tf.nn.l2_normalize(output, axis=2)
    return output


