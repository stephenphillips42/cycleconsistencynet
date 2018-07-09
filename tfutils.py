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

def create_linear_initializer(input_size, dtype=tf.float32):
  """Returns a default initializer for weights of a linear module."""
  stddev = 1 / math.sqrt(input_size)
  return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

def create_bias_initializer(unused_bias_shape, dtype=tf.float32):
  """Returns a default initializer for the biases of a linear/AddBias module."""
  return tf.zeros_initializer(dtype=dtype)

