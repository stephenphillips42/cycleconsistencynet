import tensorflow as tf

def matmul(x,y):
  return tf.einsum('bik,kj->bij', x, y)

def batch_matmul(x,y):
  return tf.einsum('bik,bkj->bij', x, y)

def get_sim(x):
  x_T = tf.transpose(x, perm=[0, 2, 1])
  return batch_matmul(x, x_T)
