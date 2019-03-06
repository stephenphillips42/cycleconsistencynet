import tensorflow as tf
import math

def matmul(x,y):
  return tf.einsum('bik,kj->bij', x, y)

def batch_matmul(x,y):
  return tf.einsum('bik,bkj->bij', x, y)

def get_sim(x):
  if isinstance(x, tf.SparseTensor):
    x_T = tf.sparse_transpose(x, perm=[0, 2, 1])
    return batch_matmul(x, x_T)
  else:
    x_T = tf.transpose(x, perm=[0, 2, 1])
    return batch_matmul(x, x_T)

def get_tf_activ(activ):
  if activ == 'relu':
    return tf.nn.relu
  elif activ == 'leakyrelu':
    return tf.nn.leaky_relu
  elif activ == 'tanh':
    return tf.nn.tanh
  elif activ == 'elu':
    return tf.nn.elu

def create_linear_initializer(input_size, output_size, dtype=tf.float32):
  """Returns a default initializer for weights of a linear module."""
  stddev = math.sqrt((1.3 * 2.0) / (input_size + output_size))
  return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

def create_bias_initializer(unused_in, unused_out, dtype=tf.float32):
  """Returns a default initializer for the biases of a linear/AddBias module."""
  return tf.zeros_initializer(dtype=dtype)

def bce_loss(labels, logits, add_loss=True):
  bce_elements = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  bce_ = tf.reduce_mean(bce_elements)
  if add_loss:
    tf.losses.add_loss(bce_)
  return bce_

# Standard losses
def l1_loss(x, y, add_loss=True):
  l1_ = tf.reduce_mean(tf.abs(x - y))
  if add_loss:
    tf.losses.add_loss(l1_)
  return l1_

def l2_loss(x, y, add_loss=True):
  l2_ = tf.reduce_mean(tf.square(x - y))
  if add_loss:
    tf.losses.add_loss(l2_)
  return l2_

def l1_l2_loss(x, y, add_loss=True):
  l1_ = tf.reduce_mean(tf.abs(x - y))
  l2_ = tf.reduce_mean(tf.square(x - y))
  l1l2_ = l1_ + l2_
  if add_loss:
    tf.losses.add_loss(l1l2_)
  return l1l2_

# Sparse losses
def l1_loss_sp(x, y, add_loss=True):
  """L1 loss, x should be a Tensor, y SparseTensor"""
  diff = tf.sparse_add(-x,y)
  l1_ = tf.reduce_mean(tf.abs(diff))
  if add_loss:
    tf.losses.add_loss(l1_)
  return l1_

def l2_loss_sp(x, y, add_loss=True):
  """L2 loss, x should be a Tensor, y SparseTensor"""
  diff = tf.sparse_add(-x,y)
  l2_ = tf.reduce_mean(tf.square(diff))
  if add_loss:
    tf.losses.add_loss(l2_)
  return l2_

def l1_l2_loss_sp(x, y, add_loss=True):
  """L1 + L2 loss, x should be a Tensor, y SparseTensor"""
  diff = tf.sparse_add(-x,y)
  l1_ = tf.reduce_mean(tf.abs(diff))
  l2_ = tf.reduce_mean(tf.square(diff))
  l1l2_ = l1_ + l2_
  if add_loss:
    tf.losses.add_loss(l1l2_)
  return l1l2_

