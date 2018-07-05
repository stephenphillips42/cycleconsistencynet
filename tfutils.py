import tensorflow as tf

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
