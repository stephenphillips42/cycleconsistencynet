import tensorflow as tf
import math

def matmul(x,y):
  return tf.einsum('bik,kj->bij', x, y)

def batch_matmul(x,y):
  return tf.einsum('bik,bkj->bij', x, y)

def get_sim(x):
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
  bce_ = tf.reduce_sum(bce_elements)
  tf.losses.add_loss(bce_)
  return bce_

def l1_loss(x, y, add_loss=True):
  l1_ = tf.reduce_mean(tf.abs(x - y))
  tf.losses.add_loss(l1_)
  return l1_

def l2_loss(x, y, add_loss=True):
  l2_ = tf.reduce_mean(tf.square(x - y))
  tf.losses.add_loss(l2_)
  return l2_

