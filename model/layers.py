# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import tensorflow as tf
import sonnet as snt

import tfutils
import myutils
import options


class AbstractGraphLayer(snt.AbstractModule):
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
    super(AbstractGraphLayer, self).__init__(custom_getter=custom_getter, name=name)
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

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    raise NotImplemented("Need to overwrite in subclass")

  def _build(self, laplacian, inputs):
    return inputs

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

  def clone(self, name=None):
    """Returns a cloned `Linear` module.
    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Returns:
      Cloned `Linear` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return AbstractGraphLayer(output_size=self.output_size,
                              use_bias=self._use_bias,
                              initializers=self._initializers,
                              partitioners=self._partitioners,
                              regularizers=self._regularizers,
                              name=name)



# TODO: Organize this into multiple files
class EmbeddingLinearLayer(AbstractGraphLayer):
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
    super(EmbeddingLinearLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

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
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    outputs = tfutils.matmul(inputs, self._w)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      outputs += self._b

    return outputs

  @property
  def w(self):
    """Returns the Variable containing the weight matrix.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
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
    return self._b

  def clone(self, name=None):
    """Returns a cloned `Linear` module.
    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Returns:
      Cloned `Linear` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return EmbeddingRightLinear(output_size=self.output_size,
                                use_bias=self._use_bias,
                                initializers=self._initializers,
                                partitioners=self._partitioners,
                                regularizers=self._regularizers,
                                name=name)

class GraphConvLayer(AbstractGraphLayer):
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
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

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
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    outputs_ = tfutils.matmul(inputs, self._w)
    outputs = tfutils.batch_matmul(laplacian, outputs_)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      outputs += self._b


    return self._activ(outputs)

  @property
  def w(self):
    """Returns the Variable containing the weight matrix.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
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
    return self._b

  def clone(self, name=None):
    """Returns a cloned `Linear` module.
    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Returns:
      Cloned `Linear` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphConvLayer(output_size=self.output_size,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           name=name)

class GraphSkipLayer(AbstractGraphLayer):
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
    super(GraphSkipLayer, self).__init__(
                 output_size,
                 use_bias=use_bias,
                 initializers=initializers,
                 partitioners=partitioners,
                 regularizers=regularizers,
                 custom_getter=custom_getter,
                 name=name)
    self._activ = tfutils.get_tf_activ(activation)
    self._w = None
    self._u = None
    self._b = None
    self._c = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "u", "b", "c"} if use_bias else {"w", "u"}

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
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    self._u = tf.get_variable("u",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["u"],
                              partitioner=self._partitioners.get("u", None),
                              regularizer=self._regularizers.get("u", None))
    preactiv_ = tfutils.matmul(inputs, self._w)
    preactiv = tfutils.batch_matmul(laplacian, preactiv_)
    skip = tfutils.matmul(inputs, self._u)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      self._c = tf.get_variable("c",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["c"],
                                partitioner=self._partitioners.get("c", None),
                                regularizer=self._regularizers.get("c", None))
      preactiv += self._b
      skip += self._c

    activ = self._activ(preactiv) + skip

    return activ

  @property
  def w(self):
    """Returns the Variable containing the weight matrix.
    Returns:
      Variable object containing the weights, from the most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def u(self):
    """Returns the Variable containing the skip connection weight matrix.
    Returns:
      Variable object containing the skip weights, from most recent __call__.
    Raises:
      snt.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._u

  @property
  def b(self):
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
    return self._b

  @property
  def c(self):
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
    return self._c

  def clone(self, name=None):
    """Returns a cloned `Linear` module.
    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.
    Returns:
      Cloned `Linear` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return GraphSkipLayer(output_size=self.output_size,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           name=name)




