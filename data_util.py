# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import os
import sys
import glob
import datetime
import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim

import options
import myutils
import sim_graphs

# Tensorflow features
def _bytes_feature(value):
  """Create arbitrary tensor Tensorflow feature."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class Int64Feature(slim.tfexample_decoder.ItemHandler):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, description):
    super(Int64Feature, self).__init__(key)
    self._key = key
    self._description = description

  def get_feature_write(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.int64)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    return tf.cast(tensor, dtype=tf.int64)

class TensorFeature(slim.tfexample_decoder.ItemHandler):
  """Custom class used for decoding serialized tensors."""
  def __init__(self, key, shape, dtype, description):
    super(TensorFeature, self).__init__(key)
    self._key = key
    self._shape = shape
    self._dtype = dtype
    self._description = description

  def get_feature_write(self, value):
    v = value.astype(self._dtype).tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

  def get_feature_read(self):
    return tf.FixedLenFeature([], tf.string)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._key]
    tensor = tf.decode_raw(tensor, out_type=self._dtype)
    return tf.reshape(tensor, self._shape)

class GraphSimDataset(object):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=7000

  def __init__(self, opts, n_pts=None, n_views=None):
    self.opts = opts
    self.data_dir = opts.data_dir
    self.n_pts = n_pts
    self.n_views = n_views
    self.dtype = opts.dtype
    d = n_pts*n_views
    e = opts.descriptor_dim
    p = opts.max_points
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
    pose_graph = sim_graphs.PoseGraph(self.opts,
                                      n_pts=self.n_pts,
                                      n_views=self.n_views)
    sz = (pose_graph.n_pts, pose_graph.n_pts)
    sz2 = (pose_graph.n_views, pose_graph.n_views)
    if self.opts.sparse:
      mask = np.kron(pose_graph.adj_mat,np.ones(sz))
    else:
      mask = np.kron(np.ones(sz2)-np.eye(sz2[0]),np.ones(sz))

    perms_ = [ np.eye(pose_graph.n_pts)[:,pose_graph.get_perm(i)]
               for i in range(pose_graph.n_views) ]
    # Embedding objects
    TrueEmbedding = np.concatenate(perms_, 0)
    InitEmbeddings = np.ones((pose_graph.n_pts*pose_graph.n_views, \
                              self.opts.descriptor_dim))
    if self.opts.use_descriptors:
      InitEmbeddings = np.concatenate([ pose_graph.get_proj(i).d
                                        for i in range(pose_graph.n_views) ], 0)

    # Graph objects
    if not self.opts.soft_edges:
      if self.opts.descriptor_noise_var == 0:
        AdjMat = np.dot(TrueEmbedding,TrueEmbedding.T)
        if self.opts.sparse:
          AdjMat = AdjMat * mask
        else:
          AdjMat = AdjMat - np.eye(len(AdjMat))
        Degrees = np.diag(np.sum(AdjMat,0))
    else:
      if self.opts.sparse and self.opts.descriptor_noise_var > 0:
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
    
  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(self.opts.sample_sizes[mode])):
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

  def create_np_dataset(self, out_dir, num_entries):
    """Create npz files to store dataset"""
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample()
      np.savez(outfile(index), **features)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('Numpy Dataset created {}'.format(str(datetime.datetime.now())))

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    assert mode in self.opts.sample_sizes, "Mode {} not supported".format(mode)
    batch_size = self.opts.batch_size
    data_source_name = mode + '-[0-9][0-9].tfrecords'
    data_sources = glob.glob(os.path.join(self.data_dir, mode, data_source_name))
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read()
                         for k, v in self.features.items() }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      self.features)
    items_to_descriptions = { k: v._description
                              for k, v in self.features.items() }
    dataset = slim.dataset.Dataset(
                data_sources=data_sources,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=self.opts.sample_sizes[mode],
                items_to_descriptions=items_to_descriptions)
    provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=self.opts.num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size,
                shuffle=self.opts.shuffle_data)
    # Extract features
    keys = list(self.features.keys())
    values = provider.get(keys)
    keys, values = self.augment(keys, values)
    # Flow preprocessing here?
    values = tf.train.batch(
                values,
                batch_size=batch_size,
                num_threads=self.opts.num_preprocessing_threads,
                capacity=5 * batch_size)
    return dict(zip(keys, values))


def get_dataset(opts):
  """Getting the dataset with all the correct attributes"""
  if opts.fixed_size:
    n_pts = opts.max_points
    n_views = opts.max_views
  else:
    n_pts = None
    n_views = None
  return GraphSimDataset(opts, n_pts=n_pts, n_views=n_views)
 
if __name__ == '__main__':
  opts = options.get_opts()
  if opts.fixed_size:
    n_pts = opts.max_points
    n_views = opts.max_views
  else:
    n_pts = None
    n_views = None
  print("Generating Pose Graphs")
  if not os.path.exists(opts.data_dir):
    os.makedirs(opts.data_dir)

  sizes = {
    'train' : opts.num_gen_train,
    'test' : opts.num_gen_test
  }
  for t, sz in sizes.items():
    dname = os.path.join(opts.data_dir,t)
    if not os.path.exists(dname):
      os.makedirs(dname)
    dataset = GraphSimDataset(opts, n_pts, n_views)
    dataset.convert_dataset(dname, t)

  # Generate numpy test
  out_dir = os.path.join(opts.data_dir,'np_test')
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  dataset = GraphSimDataset(opts, n_pts, n_views)
  dataset.create_np_dataset(out_dir, sizes['test'])


