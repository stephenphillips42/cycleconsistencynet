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

class LaplacianSkipNetwork(mygraphnetwork.MyGraphNetwork):
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
    with self._enter_variable_scope():
      self._layers = [
        layers.LaplacianGraphSkipLayer(
          output_size=layer_len,
          activation=arch.activ,
          initializers=self.initializers,
          regularizers=self.regularizers,
          name="{}/graph_skip".format(name))
        for layer_len in arch.layer_lens
      ]
      self.final_linear = [
        layers.EmbeddingLinearLayer(
          output_size=opts.final_embedding_dim,
          initializers=self.initializers,
          regularizers=self.regularizers,
          name="{}/embed_lin".format(name))
      ]
      self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    laplacian_ = tf.sparse_tensor_to_dense(laplacian)
    output = init_embeddings
    for layer in self._layers:
      output = layer(laplacian_, output)
    output = self.final_linear(output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output


class LaplacianLongSkipNetwork(mygraphnetwork.MyGraphNetwork):
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
    with self._enter_variable_scope():

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    laplacian_ = tf.sparse_tensor_to_dense(laplacian)
    output = init_embeddings
    sk = 0
    for i, layer in enumerate(self._layers):
      if i in self._skip_layer_idx:
        skip_val = self._skip_layers[sk](laplacian_, output)
        output = layer(laplacian_, output) + skip_val
        sk += 1
      else:
        output = layer(laplacian, output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output

# TODO: Rename things for skip/hop
class GraphSkipHopLayerNetwork(LaplacianLongSkipNetwork):
  def __init__(self,
               opts,
               arch,
               use_bias=True,
               initializers=None,
               regularizers=None,
               custom_getter=None,
               name="graphnn"):
    super().__init__(opts,
                     arch,
                     use_bias=use_bias,
                     initializers=initializers,
                     regularizers=regularizers,
                     custom_getter=custom_getter,
                     name=name)
    with self._enter_variable_scope():
      self._layers = [
        layers.GraphSkipLayer(
          output_size=layer_len,
          activation=arch.activ,
          initializers=initializers,
          regularizers=regularizers,
          name="{}/graph_skip".format(name))
        for layer_len in arch.layer_lens
      ]
      self.final_linear = [
        layers.EmbeddingLinearLayer(
          output_size=opts.final_embedding_dim,
          initializers=self.initializers,
          regularizers=self.regularizers,
          name="{}/embed_lin".format(name))
      ]
      self._skip_layer_idx = arch.skip_layers
      self._skip_layers = {
        skip_idx: layers.EmbeddingLinearLayer(
          output_size=arch.layer_lens[skip_idx],
          initializers=self.initializers,
          regularizers=self.regularizers,
          name="{}/skip".format(name))
        for skip_idx in self._skip_layer_idx
      }
      self._hop_layers = {
        skip_idx: layers.EmbeddingLinearLayer(
          output_size=arch.layer_lens[skip_idx],
          initializers=self.initializers,
          regularizers=self.regularizers,
          name="{}/hop".format(name))
        for skip_idx in self._skip_layer_idx[1:]
      }
    self.normalize_emb = arch.normalize_emb

  def _build(self, laplacian, init_embeddings):
    """Applying this graph network to sample"""
    laplacian_ = tf.sparse_tensor_to_dense(laplacian)
    output = init_embeddings
    sk = 0
    last_skip = None
    for i, layer in enumerate(self._layers):
      output = layer(laplacian_, output)
      if i in self._skip_layers:
        skip_add = self._skip_layers[i](laplacian_, init_embeddings)
        output = output + skip_add
        if last_skip is not None:
          hop_add = self._hop_layers[i](laplacian_, last_skip)
          output = output + hop_add
        last_skip = output
        sk += 1
    output = self.final_linear(output)
    if self.normalize_emb:
      output = tf.nn.l2_normalize(output, axis=2)
    return output



