import numpy as np
import os
import sys
import glob
import datetime
import tqdm
import pickle

import tensorflow as tf

import sim_graphs
from data_util import parent_dataset
from data_util import tf_helpers

# Format:
# dict: {'train', 'test', 'np_dataset'}
#   -> dict: rome16k_name -> (ntriplets, ncams)
bundle_file_info = {
  'train' : {
    '5.1.0.0': (507, 44),
    '20.0.0.0': (566, 48),
    '55.0.0.0': (644, 58),
    '38.0.0.0': (663, 70),
    '26.1.0.0': (744, 86),
    '74.0.0.0': (1050, 33),
    '49.0.0.0': (1053, 37),
    '36.0.0.0': (1204, 34),
    '12.0.0.0': (1511, 63),
    '60.0.0.0': (2057, 47),
    '54.0.0.0': (2068, 60),
    '57.0.0.0': (2094, 40),
    '167.0.0.0': (2119, 42),
    '4.11.0.0': (2714, 115),
    '38.3.0.0': (3248, 39),
    '135.0.0.0': (3476, 46),
    '4.8.0.0': (3980, 63),
    '110.0.0.0': (4075, 86),
    '4.3.0.0': (4442, 81),
    '29.0.0.0': (4849, 50),
    '97.0.0.0': (4967, 87),
    '4.6.0.0': (5409, 99),
    '84.0.0.0': (5965, 59),
    '9.1.0.0': (6536, 58),
    '33.0.0.0': (6698, 125),
    '15.0.0.0': (9950, 59),
    '26.5.0.0': (12913, 54),
    '122.0.0.0': (15269, 93),
    '10.0.0.0': (16709, 101),
    '11.0.0.0': (16871, 262),
    '0.0.0.0': (22632, 186),
    '17.0.0.0': (28333, 117),
    '16.0.0.0': (35180, 93),
    '4.1.0.0': (36460, 163),
    '26.2.0.1': (75225, 135),
    '4.5.0.0': (79259, 251)
  },
  'test' : {
    '73.0.0.0': (26, 32),
    '33.0.0.1': (31, 13),
    '5.11.0.0': (93, 50),
    '0.3.0.0': (170, 31),
    '46.0.0.0': (205, 67),
    '26.4.0.0': (239, 30),
    '82.0.0.0': (256, 56),
    '65.0.0.0': (298, 35),
    '40.0.0.0': (340, 36),
    '56.0.0.0': (477, 30),
    '5.9.0.0': (481, 88),
    '34.1.0.0': (487, 38),
  },
  'np_dataset' : {
    '11.2.0.0': (12, 40),
    '125.0.0.0': (21, 34),
    '41.0.0.0': (22, 54),
    '37.0.0.0': (25, 46),
  }
}


class Rome16KTripletDataset(parent_dataset.GraphSimDataset):
  """Abstract base class for Rome16K cycle consistency graphs"""

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)
    self.rome16k_dir = opts.rome16k_dir
    del self.features['Mask']
    del self.features['MaskOffset']
    self.dataset_params.sizes['train'] = \
        sum([x[0] for _, x in bundle_file_info['train'].items()])
    self.dataset_params.sizes['test'] = \
        sum([x[0] for _, x in bundle_file_info['test'].items()])
    self.np_dataset_size = \
        sum([x[0] for _, x in bundle_file_info['np_dataset'].items()])

  def gen_sample(self):
    print("ERROR: Cannot generate sample - need to load data")
    sys.exit(1)

  def gen_sample_from_triplet(self, scene, triplet):
    print("ERROR: Not implemented in abstract base class")
    sys.exit(1)

  def triplet_fname(name):
    return os.path.join(self.rome16k_dir, 'triplets.{}.pkl'.format(name))

  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    params = self.dataset_params
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    scene = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1

    pbar = tqdm.tqdm(total=params.sizes[mode])
    for name in bundle_file_info[mode]:
      with open(triplet_fname(name), 'rb') as f:
        triplets = pickle.load(f)
      for triplet in triplets:
        if file_idx > self.MAX_IDX:
          file_idx = 0
          if writer: writer.close()
          writer = tf.python_io.TFRecordWriter(outfile(record_idx))
          record_idx += 1
        loaded_features = self.gen_sample_from_triplet(scene, triplet)
        features = self.process_features(loaded_features)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
        file_idx += 1
        pbar.update()

    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def create_np_dataset(self, out_dir, num_entries):
    """Create npz files to store dataset"""
    del num_entries
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    pbar = tqdm.tqdm(total=self.np_dataset_size)
    for name in bundle_file_info['np_dataset']:
      with open(triplet_fname(name), 'rb') as f:
        triplets = pickle.load(f)
      for triplet in triplets:
        loaded_features = self.gen_sample_from_triplet(scene, triplet)
        features = self.process_features(loaded_features)
        np.savez(outfile(index), **features)
        pbar.update()

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


