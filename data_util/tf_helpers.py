import tensorflow as tf

class MyFeature(object):
  """Custom class used for decoding a serialized int64 value."""
  def __init__(self, key, description, shape=[], dtype='float32'):
    super().__init__()
    self._key = key
    self.description = description
    self.shape = shape
    self.dtype = dtype

  def get_placeholder(self):
    return tf.placeholder(self.dtype, shape=[None] + self.shape)

  def _get_feature_write(self, value):
    return value

  def _get_feature_read(self):
    return tf.FixedLenFeature([], self.dtype)

  def get_feature_write(self, value):
    feat_write = self._get_feature_write(value)
    if type(feat_write) == dict:
      return feat_write
    else:
      return { self._key : feat_write }

  def get_feature_read(self):
    feat_read = self._get_feature_read()
    if type(feat_read) == dict:
      return feat_read
    else:
      return { self._key : feat_read }

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    return tensor


class Int64Feature(MyFeature):
  """Custom class used for decoding a serialized int64 value."""
  def __init__(self, key, description, dtype='int64'):
    super().__init__(key, description, shape=[], dtype='int64')
    self.convert_to = dtype

  def _get_feature_write(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    if self.convert_to != 'int64':
      return tf.cast(tensor, dtype=self.convert_to)
    else:
      return tf.cast(tensor, dtype=tf.int64)


class TensorFeature(MyFeature):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super().__init__(key, description, shape=shape, dtype=dtype)

  def _get_feature_write(self, value):
    v = value.astype(self.dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def _get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self.dtype)
    return tf.reshape(tensor, self.shape)


class VarLenIntListFeature(MyFeature):
  """Custom class used for decoding variable length int64 lists."""
  def __init__(self, key, dtype, description):
    super().__init__(key, description, shape=[None], dtype=dtype)

  def _get_feature_write(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _get_feature_read(self):
    return tf.VarLenFeature(tf.int64)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    return tf.cast(tensor, self.dtype)


class VarLenFloatFeature(MyFeature):
  """Custom class used for decoding variable length float tensors."""
  def __init__(self, key, shape, description):
    super().__init__(key, description, shape=shape, dtype='float32')

  def _get_feature_write(self, value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _get_feature_read(self):
    return tf.VarLenFeature(tf.float32)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    return tensor


class SparseTensorFeature(MyFeature):
  """Custom class used for decoding serialized sparse float tensors."""
  def __init__(self, key, shape, description):
    super().__init__(key, description, shape=shape, dtype='float32')

  def _get_feature_write(self, value):
    idx, value = value[0], value[1]
    sptensor_feature = { '{}_{:02d}'.format(self._key,i) :
                           tf.train.Feature(
                              int64_list=tf.train.Int64List(value=idx[i]))
                         for i in range(len(self.shape)) }
    sptensor_feature['{}_value'.format(self._key)] = \
      tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return sptensor_feature

  def _get_feature_read(self):
    feat_write = { '{}_{:02d}'.format(self._key,i) :
                     tf.VarLenFeature(tf.int64)
                   for i in range(len(self.shape)) }
    feat_write['{}_value'.format(self._key)] = tf.VarLenFeature(self.dtype)
    return feat_write

  def tensors_to_item(self, keys_to_tensors):
    indeces_sp = [ keys_to_tensors['{}_{:02d}'.format(self._key,i)]
                     for i in range(len(self.shape)) ]
    indeces = [ tf.sparse_tensor_to_dense(inds) for inds in indeces_sp ]
    values_sp = keys_to_tensors['{}_value'.format(self._key)]
    values = tf.sparse_tensor_to_dense(values_sp)
    tensor = tf.SparseTensor(indeces, values, self.shape)
    return tensor


