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
    self._nlayers = len(arch.layer_lens)
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
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    if self.normalize_emb:
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
    self._nlayers = len(arch.layer_lens)
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
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphLongSkipLayerNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphLongSkipLayerNetwork, self).__init__(custom_getter=custom_getter,
                                                    name=name)
    self._nlayers = len(arch.layer_lens)
    final_regularizers = None
    if regularizers is not None:
      lin_regularizers = { k:v
                           for k, v in regularizers.items()
                           if k in ["w", "b"] }
    else:
      lin_regularizers = None
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
        regularizers=lin_regularizers,
        name="{}/embed_lin".format(name))
    ]
    self._skip_layer_idx = arch.skip_layers
    self._skip_layers = [
      layers.EmbeddingLinearLayer(
        output_size=arch.layer_lens[skip_idx],
        initializers=initializers,
        regularizers=lin_regularizers,
        name="{}/skip".format(name))
      for skip_idx in self._skip_layer_idx
    ]
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    sk = 0
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        output = layer(laplacian, output) + self._skip_layers[sk](laplacian, output)
        sk += 1
      else:
        output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

class GraphLongSkipNormedNetwork(snt.AbstractModule):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super(GraphLongSkipLayerNetwork, self).__init__(custom_getter=custom_getter,
                                                    name=name)
    self._nlayers = len(arch.layer_lens)
    self.start_normed = arch.start_normed
    final_regularizers = None
    if regularizers is not None:
      lin_regularizers = { k:v
                           for k, v in regularizers.items()
                           if k in ["w", "b"] }
    else:
      lin_regularizers = None
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
        regularizers=lin_regularizers,
        name="{}/embed_lin".format(name))
    ]
    self._skip_layer_idx = arch.skip_layers
    self._skip_layers = [
      layers.EmbeddingLinearLayer(
        output_size=arch.layer_lens[skip_idx],
        initializers=initializers,
        regularizers=lin_regularizers,
        name="{}/skip".format(name))
      for skip_idx in self._skip_layer_idx
    ]
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    sk = 0
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        output = layer(laplacian, output) + self._skip_layers[sk](laplacian, init_embeddings)
        sk += 1
      else:
        output = layer(laplacian, output)
      if i >= self.start_normed:
        output = tf.contrib.layers.group_norm(output)
    if self.normalize_emb:
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
    self._nlayers = len(arch.layer_lens)
    final_regularizers = None
    if regularizers is not None:
      final_regularizers = { k:v
                             for k, v in regularizers.items()
                             if k in ["w", "b"] }
    self._layers = [
      layers.GraphAttentionLayer(
        output_size=layer_len,
        activation=arch.activ,
        sparse=arch.sparse,
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
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output


