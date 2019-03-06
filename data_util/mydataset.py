# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import glob
import datetime
import tqdm

import tensorflow as tf

from data_util import tf_helpers

class MyDataset(object):
  """Dataset for Cycle Consistency graphs"""
  MAX_IDX=500

  def __init__(self, opts, params):
    self.opts = opts
    self.dataset_params = params
    self.data_dir = params.data_dir
    self.dtype = params.dtype
    self.n_views = params.views[-1]
    self.n_pts = params.points[-1]
    d = self.n_pts*self.n_views
    e = params.descriptor_dim
    p = params.points[-1]
    f = opts.final_embedding_dim
    self.features = {
      'InitEmbeddings':
           tf_helpers.TensorFeature(
                         key='InitEmbeddings',
                         shape=[d, e],
                         dtype=self.dtype,
                         description='Initial embeddings for optimization'),
      'AdjMat':
           tf_helpers.TensorFeature(
                         key='AdjMat',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Adjacency matrix for graph'),
      'Degrees':
           tf_helpers.TensorFeature(
                         key='Degrees',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Degree matrix for graph'),
      'Laplacian':
           tf_helpers.TensorFeature(
                         key='Laplacian',
                         shape=[d, d],
                         dtype=self.dtype,
                         description='Alternate Laplacian matrix for graph'),
      'TrueEmbedding':
           tf_helpers.TensorFeature(
                         key='TrueEmbedding',
                         shape=[d, p],
                         dtype=self.dtype,
                         description='True values for the low dimensional embedding'),
      'NumViews':
           tf_helpers.Int64Feature(
                         key='NumViews',
                         description='Number of views used in this example'),
      'NumPoints':
           tf_helpers.Int64Feature(
                         key='NumPoints',
                         description='Number of points used in this example'),
    }

  def process_features(self, loaded_features):
    features = {}
    for k, feat in self.features.items():
      features.update(feat.get_feature_write(loaded_features[k]))
    return features

  def augment(self, keys, values):
    return keys, values

  def gen_sample(self):
    print("Error: Unable to generate sample - Not Implemented")
    sys.exit(1)

  def get_placeholders(self):
    return { k:v.get_placeholder() for k, v in self.features.items() }

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    params = self.dataset_params
    fname = '{}-{:03d}.tfrecords'
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

  def gen_batch(self, mode):
    """Return batch loaded from this dataset"""
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    batch_size = opts.batch_size
    keys = sorted(list(self.features.keys()))
    shapes = [ self.features[k].shape for k in keys ]
    types = [ self.features[k].dtype for k in keys ]
    tfshapes = [ tuple([batch_size] + s) for s in shapes ]
    tftypes = [ tf.as_dtype(t) for t in types ]
    def generator_fn():
      while True:
        vals = [ np.zeros([batch_size] + s, types[i])
                 for i, s in enumerate(shapes) ]
        for b in range(batch_size):
          s = self.gen_sample()
          for i, k in enumerate(keys):
            vals[i][b] = s[k]
        yield tuple(vals)
    dataset = tf.data.Dataset.from_generator(generator_fn,
                                             tuple(tftypes),
                                             tuple(tfshapes))
    batches = dataset.prefetch(2 * batch_size)

    iterator = batches.make_one_shot_iterator()
    values = iterator.get_next()
    return dict(zip(keys, values))

  def load_batch(self, mode):
    """Return batch loaded from this dataset"""
    params = self.dataset_params
    opts = self.opts
    assert mode in params.sizes, "Mode {} not supported".format(mode)
    data_source_name = mode + '-*.tfrecords'
    data_sources = glob.glob(os.path.join(self.data_dir, mode, data_source_name))
    if opts.shuffle_data and mode != 'test':
      np.random.shuffle(data_sources) # Added to help the shuffle
    # Build dataset provider
    keys_to_features = { k: v.get_feature_read()
                         for k, v in self.features.items() }
    items_to_descriptions = { k: v.description
                              for k, v in self.features.items() }
    def parser_op(record):
      example = tf.parse_single_example(record, keys_to_features)
      return { k : v.tensors_to_item(example) for k, v in self.features.items() }
    dataset = tf.data.TFRecordDataset(data_sources)
    dataset = dataset.map(parser_op)
    dataset = dataset.repeat(None)
    if opts.shuffle_data and mode != 'test':
      dataset = dataset.shuffle(buffer_size=5*opts.batch_size)
    if opts.batch_size > 1:
      dataset = dataset.batch(opts.batch_size)
      # TODO: Is this the optimal buffer_size?
      dataset = dataset.prefetch(buffer_size=opts.batch_size)

    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()
    return sample


