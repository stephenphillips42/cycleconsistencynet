import sys
import os
import glob
import tensorflow as tf
import collections
import sonnet as snt

import options 
import model
import tfutils

class Int64Feature(object):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, description):
    super(Int64Feature, self).__init__()
    self._key = key
    self.shape = []
    self._description = description

  def get_placeholder(self):
    return tf.placeholder(tf.int64, shape=[None])

  def get_feature_write(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.int64)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    return tf.cast(tensor, dtype=tf.int64)

class TensorFeature(object):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super(TensorFeature, self).__init__()
    self._key = key
    self.shape = shape
    self._dtype = dtype
    self._description = description

  def get_placeholder(self):
    return tf.placeholder(self._dtype, shape=[None] + self.shape)

  def get_feature_write(self, value):
    v = value.astype(self._dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self._dtype)
    return tf.reshape(tensor, self.shape)


class GraphSimDataset(object):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, params):
    self.opts = opts
    self.dataset_params = params
    self.data_dir = params.data_dir
    self.dtype = params.dtype
    if params.fixed_size:
      self.n_views = params.views[-1]
      self.n_pts = params.points[-1]
    else:
      self.n_views = np.random.randint(params.views[0], params.views[1]+1)
      self.n_pts = np.random.randint(params.points[0], params.points[1]+1)
    d = self.n_pts*self.n_views
    e = params.descriptor_dim
    p = params.points[-1]
    f = opts.final_embedding_dim
    self.features = {
      'InitEmbeddings':
           TensorFeature(key='InitEmbeddings',
                         shape=[d, e],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'AdjMat':
           TensorFeature(key='AdjMat',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Adjacency matrix for graph'),
      'Degrees':
           TensorFeature(key='Degrees',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Degree matrix for graph'),
      'Laplacian':
           TensorFeature(key='Laplacian',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Alternate Laplacian matrix for graph'),
      'Mask':
           TensorFeature(key='Mask',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Mask for valid values of matrix'),
      'MaskOffset':
           TensorFeature(key='MaskOffset',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Mask offset for loss'),
      'TrueEmbedding':
           TensorFeature(key='TrueEmbedding',
                         shape=[d, p],
                         dtype=self.dtype,
                         description='True values for the low dimensional embedding'),
      'NumViews':
           Int64Feature(key='NumViews',
                         description='Number of views used in this example'),
      'NumPoints':
           Int64Feature(key='NumPoints',
                         description='Number of points used in this example'),
    }

  def process_features(self, loaded_features):
    features = {}
    for k, feat in self.features.items():
      features[k] = feat.get_feature_write(loaded_features[k])
    return features

  def augment(self, keys, values):
    return keys, values

  def gen_sample(self):
    # Pose graph and related objects
    params = self.dataset_params
    pose_graph = sim_graphs.PoseGraph(self.dataset_params,
                                      n_pts=self.n_pts,
                                      n_views=self.n_views)
    sz = (pose_graph.n_pts, pose_graph.n_pts)
    sz2 = (pose_graph.n_views, pose_graph.n_views)
    if params.sparse:
      mask = np.kron(pose_graph.adj_mat,np.ones(sz))
    else:
      mask = np.kron(np.ones(sz2)-np.eye(sz2[0]),np.ones(sz))

    perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
               for i in range(pose_graph.n_views) ]
    # Embedding objects
    TrueEmbedding = np.concatenate(perms_, 0)
    InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                      for i in range(pose_graph.n_views) ], 0)

    # Graph objects
    if not params.soft_edges:
      if params.descriptor_noise_var == 0:
        AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
        if params.sparse:
          AdjMat = AdjMat * mask
        else:
          AdjMat = AdjMat - np.eye(len(AdjMat))
        Degrees = np.diag(np.sum(AdjMat,0))
    else:
      if params.sparse and params.descriptor_noise_var > 0:
        AdjMat = pose_graph.get_feature_matching_mat()
        Degrees = np.diag(np.sum(AdjMat,0))

    # Laplacian objects
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))

    # Mask objects
    neg_offset = np.kron(np.eye(sz2[0]),np.ones(sz)-np.eye(sz[0]))
    Mask = AdjMat - neg_offset
    MaskOffset = neg_offset
    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'Mask': Mask.astype(self.dtype),
      'MaskOffset': MaskOffset.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': pose_graph.n_views,
      'NumPoints': pose_graph.n_pts,
    }

  def get_placeholders(self):
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    params = self.dataset_params
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(params.sizes[mode])):
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer: writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.gen_sample()
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    batch_size = opts.batch_size
    data_source_name = mode + '-[0-9][0-9].tfrecords'
    print((self.data_dir, mode, data_source_name))
    data_sources = glob.glob(os.path.join(self.data_dir, mode, data_source_name))
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read()
                         for k, v in self.features.items() }
    items_to_descriptions = { k: v._description
                              for k, v in self.features.items() }
    def parser_op(record):
      example = tf.parse_single_example(record, keys_to_features)
      return { k : v.tensors_to_item(example) for k, v in self.features.items() }
    dataset = tf.data.TFRecordDataset(data_sources)
    dataset = dataset.map(parser_op)
    dataset = dataset.shuffle(buffer_size=5*opts.batch_size)
    dataset = dataset.batch(opts.batch_size)

    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()
    return sample

class EmbeddingRightLinear(snt.AbstractModule):
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
               name="lin"):
    super(EmbeddingRightLinear, self).__init__(custom_getter=custom_getter, name=name)
    self._output_size = output_size
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = snt.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = snt.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = snt.check_regularizers(
        regularizers, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise base.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[2] is None:
      raise base.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and input_shape[2] != self._input_shape[2]:
      raise base.IncompatibleShapeError(
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
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
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
      base.NotConnectedError: If the module has not been connected to the
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
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

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
    return EmbeddingRightLinear(output_size=self.output_size,
                                use_bias=self._use_bias,
                                initializers=self._initializers,
                                partitioners=self._partitioners,
                                regularizers=self._regularizers,
                                name=name)

class GraphConvLayer(snt.AbstractModule):
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
               name="lin"):
    super(GraphConvLayer, self).__init__(custom_getter=custom_getter, name=name)
    self._output_size = output_size
    self._activ = tfutils.get_tf_activ(activation)
    self._use_bias = use_bias
    self._input_shape = None
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = snt.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = snt.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = snt.check_regularizers(
        regularizers, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs, laplacian):
    input_shape = tuple(inputs.get_shape().as_list())
    if len(input_shape) != 3:
      raise base.IncompatibleShapeError(
          "{}: rank of shape must be 3 not: {}".format(
              self.scope_name, len(input_shape)))

    # TODO: Add shape constraints to laplacian

    if input_shape[2] is None:
      raise base.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if input_shape[1] is None:
      raise base.IncompatibleShapeError(
          "{}: Number of nodes must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and \
        (input_shape[2] != self._input_shape[2] or \
         input_shape[1] != self._input_shape[1]):
      raise base.IncompatibleShapeError(
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
                                          dtype)

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = tfutils.create_bias_initializer(
                                          self._input_shape[2],
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
      base.NotConnectedError: If the module has not been connected to the
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
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

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
    return GraphConvLayer(output_size=self.output_size,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           name=name)

if __name__ == "__main__":
  opts = options.get_opts()
  data = GraphSimDataset(opts, opts.dataset_params)
  sample = data.load_batch('test')
  net = GraphConvLayer(opts.dataset_params.points[-1])
  output = net(sample["InitEmbeddings"], sample["Laplacian"])
  print(sample["InitEmbeddings"])
  print(output)
  print("")
  


