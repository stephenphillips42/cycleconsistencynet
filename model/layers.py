# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options


# AdjMat network based layers
class LaplacianGraphLayer(snt.AbstractModule):
  """Transformation on an graphe node embedding.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="embed_lin"):
    super().__init__(custom_getter=None, name=name)
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = snt.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = snt.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = snt.check_regularizers(
        regularizers, self.possible_keys)
    self._weight = {}
    self._bias = {}

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    ret_val = set(self._weight.keys())
    if use_bias:
      ret_val = ret_val.union(self._bias.keys())
    return ret_val

  @property
  def output_size(self):
    """Returns the module output size."""
    if callable(self._output_size):
      self._output_size = self._output_size()
    return self._output_size

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  @property
  def regularizers(self):
    """Returns the regularizers dictionary."""
    return self._regularizers

  @property
  def weights(self):
    """Returns the Variable containing the weight matrix.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._weight

  @property
  def biases(self):
    """Returns the Variable containing the bias.
    Returns:
      Variable object containing the bias, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._bias


class EmbeddingLinearLayer(LaplacianGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="embed_lin"):
    super().__init__(output_size,
                     use_bias=use_bias,
                     initializers=initializers,
                     partitioners=partitioners,
                     regularizers=regularizers,
                     custom_getter=custom_getter,
                     name=name)
    self._weight = { 'w': None }
    self._bias = { 'b': None }
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  def _build(self, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and input_shape[2] != self._input_shape[2]:
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  input_shape[2],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))

    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._weight['w'] = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._w)
    outputs = tfutils.matmul(inputs, self._w)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._bias['b'] = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._b)
      outputs += self._b

    return outputs


class LaplacianGraphConvLayer(LaplacianGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
  """
  def __init__(self,
               output_size,
               activation='relu',
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="graph_conv"):
    super(GraphConvLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._activ = tfutils.get_tf_activ(activation)
    self._weight = { 'w' : None }
    self._bias = { 'b' : None }
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    # TODO: Add shape constraints to laplacian

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise snt.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  self._input_shape[1],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))


    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._weight['w'] = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._w)
    outputs_ = tfutils.matmul(inputs, self._w)
    outputs = tfutils.batch_matmul(laplacian, outputs_)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._bias['b'] = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._b)
      outputs += self._b


    return self._activ(outputs)


class LaplacianGraphSkipLayer(LaplacianGraphLayer):
  """Linear transformation on an embedding, each independently.
  
  This functions almost exactly like snt.Linear except it is for tensors of
  size batch_size x nodes x input_size. Acts by matrix multiplication on the
  left side of each nodes x input_size matrix.
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
    super().__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._activ = tfutils.get_tf_activ(activation)
    self._weight = {
      "w" : None
      "u" : None
    }
    self._bias = {
      "b" : None
      "c" : None
    }
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  def _build(self, laplacian, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise snt.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise snt.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise snt.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise snt.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}, {}] not: [batch_size, {}, {}]"
          .format(self.scope_name,
                  self._input_shape[1],
                  self._input_shape[2],
                  input_shape[1],
                  input_shape[2]))


    self._input_shape = input_shape
    dtype = inputs.dtype

    if "w" not in self._initializers:
      self._initializers["w"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)
    if "u" not in self._initializers:
      self._initializers["u"] = tfutils.create_linear_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)
    if "c" not in self._initializers and self._use_bias:
      self._initializers["c"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
                                          self._output_size,
                                          dtype)

    weight_shape = (self._input_shape[2], self.output_size)
    self._weight['w'] = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    if self._w not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._weight['w'])
    self._weight['u'] = tf.get_variable("u",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["u"],
                              partitioner=self._partitioners.get("u", None),
                              regularizer=self._regularizers.get("u", None))
    if self._u not in tf.get_collection('weights'):
      tf.add_to_collection('weights', self._u)
    preactiv_ = tfutils.matmul(inputs, self._w)
    preactiv = tfutils.batch_matmul(laplacian, preactiv_)
    skip = tfutils.matmul(inputs, , self._weight['u'])

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._bias['b'] = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      if self._b not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._bias['b'])
      self._bias['c'] = tf.get_variable("c",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["c"],
                                partitioner=self._partitioners.get("c", None),
                                regularizer=self._regularizers.get("c", None))
      if self._c not in tf.get_collection('biases'):
        tf.add_to_collection('biases', self._bias['c'])
      preactiv += self._bias['b']
      skip += self._bias['c']

    activ = self._activ(preactiv) + skip

    return activ


# Sparse graph related layers
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

# Group norm - currently not used
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
    # Reference: https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py
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



if __name__ == "__main__":
  pass


