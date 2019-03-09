# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options


# TODO: Make a version of this that does not average accross images
class GraphGroupNorm(snt.AbstractModule):
  """Group Norm for Graph-based inputs module 

  Implemented since the tensorflow one does not work with unknown batch size.
  Assumed input dimensions is [ Batch, Nodes, Features ]
  """
  def __init__(self, group_size=32, name='group_norm'):
    super().__init__(custom_getter=None, name=name)
    self.group_size = 32
    self.possible_keys = self.get_possible_initializer_keys()
    self._initializers = {
      'gamma' : tf.ones_initializer(),
      'beta'  : tf.zeros_initializer()
    }
    self._gamma = None
    self._beta = None
    self._input_shape = None

  def get_possible_initializer_keys(cls, use_bias=True):
    return {"gamma", "beta"}

  @property
  def gamma(self):
    """Returns the Variable containing the scale parameter, gamma.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._gamma

  @property
  def beta(self):
    """Returns the Variable containing the center parameter, beta.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._beta

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  def clone(self, name=None):
    """Returns a cloned `GraphGroupNorm` module.
    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Returns:
      Cloned `GraphGroupNorm` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphGroupNorm(group_size=self.group_size)

  def _build(self, inputs):
    # TODO: Reference https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))
    self._input_shape = input_shape
    dtype = inputs.dtype
    group_sizes = [ self.group_size, self._input_shape[2] // self.group_size ]
    broadcast_shape = [ 1, 1 ] + group_sizes
    self._gamma = tf.get_variable("gamma",
                                  shape=(self._input_shape[2]),
                                  dtype=dtype,
                                  initializer=self._initializers["gamma"])
    if self._gamma not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._gamma)
    self._gamma = tf.reshape(self._gamma, broadcast_shape)

    self._beta = tf.get_variable("beta",
                                 shape=(self._input_shape[2],),
                                 dtype=dtype,
                                 initializer=self._initializers["beta"])
    if self._beta not in tf.get_collection('biases'):
      tf.add_to_collection('biases', self._beta)
    self._beta = tf.reshape(self._beta, broadcast_shape)

    ##### Actually perform operations
    # Reshape input
    original_shape = [ -1, self._input_shape[1], self._input_shape[2] ]
    inputs_shape = [ -1, self._input_shape[1] ] + group_sizes
                     
    inputs = tf.reshape(inputs, inputs_shape)

    # Normalize
    mean, variance = tf.nn.moments(inputs, [1, 3], keep_dims=True)
    gain = tf.rsqrt(variance + 1e-7) * self._gamma
    offset = -mean * gain + self._beta
    outputs = inputs * gain + offset

    # Reshape back to output
    outputs = tf.reshape(outputs, original_shape)

    return outputs

class SkipLayer(snt.AbstractModule):
  """MLP + Linear layer for convenience
  
  More Specifically this creates an MLP with layers [output_size, output_size]
  and a linear layer with layer output_size and adds their results
  """
  def __init__(self,
               output_size,
               activation='relu',
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="graph_skip"):
    super().__init__(custom_getter=custom_getter, name=name)
    with self._enter_variable_scope():
      self._linear = snt.Linear(output_size,
                                use_bias=use_bias,
                                initializers=initializers,
                                partitioners=partitioners,
                                regularizers=regularizers,
                                custom_getter=custom_getter)
      self._mlp = snt.nets.MLP([ output_size, output_size ],
                                use_bias=use_bias,
                                initializers=initializers,
                                partitioners=partitioners,
                                regularizers=regularizers,
                                custom_getter=custom_getter)

  def _build(self, x):
    return self._linear(x) + self._mlp(x)



if __name__ == "__main__":
  pass


