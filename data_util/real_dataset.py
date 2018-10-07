import numpy as np
import scipy.linalg as la
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
from data_util.rome16k import parse

class Rome16KTripletDataset(parent_dataset.GraphSimDataset):
  """Abstract base class for Rome16K cycle consistency graphs"""

  def __init__(self, opts, params):
    parent_dataset.GraphSimDataset.__init__(self, opts, params)
    self.rome16k_dir = opts.rome16k_dir
    del self.features['Mask']
    del self.features['MaskOffset']
    self.dataset_params.sizes['train'] = \
        sum([x[0] for _, x in parse.bundle_file_info['train'].items()])
    self.dataset_params.sizes['test'] = \
        sum([x[0] for _, x in parse.bundle_file_info['test'].items()])
    self.np_dataset_size = \
        sum([x[0] for _, x in parse.bundle_file_info['np_dataset'].items()])

  def gen_sample(self):
    print("ERROR: Cannot generate sample - need to load data")
    sys.exit(1)

  def gen_sample_from_triplet(self, scene, triplet):
    print("ERROR: Not implemented in abstract base class")
    sys.exit(1)

  def triplet_fname(self, bundle_file):
    return os.path.join(self.rome16k_dir, parse.triplets_name(bundle_file))

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
    for bundle_file in parse.bundle_file_info[mode]:
      scene_name = '{}/{}'.format(self.rome16k_dir, parse.scene_name(bundle_file))
      scene = parse.load_scene(scene_name)
      with open(self.triplet_fname(bundle_file), 'rb') as f:
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
    index = 0
    for bundle_file in parse.bundle_file_info['np_dataset']:
      scene_name = '{}/{}'.format(self.rome16k_dir, parse.scene_name(bundle_file))
      scene = parse.load_scene(scene_name)
      with open(self.triplet_fname(bundle_file), 'rb') as f:
        triplets = pickle.load(f)
      for triplet in triplets:
        features = self.gen_sample_from_triplet(scene, triplet)
        np.savez(outfile(index), **features)
        index += 1
        pbar.update()

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

class KNNRome16KDataset(Rome16KTripletDataset):
  def __init__(self, opts, params):
    Rome16KTripletDataset.__init__(self, opts, params)

  def gen_sample_from_triplet(self, scene, triplet):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]
    mask = np.kron(np.ones((v,v))-np.eye(v),np.ones((n,n)))
    cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
    point_set = cam_pt(triplet[0]) & cam_pt(triplet[1]) & cam_pt(triplet[2])
    # Build features
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [] 
    for camid in triplet:
      fset = [ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ]
      fset = sorted(fset, key=lambda x: x.id)
      features.append([ fset[x] for x in feat_perm ])
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    rids = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in rids ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))
    desc_norms = np.sum(descs**2, 1).reshape(-1, 1)
    ndescs = descs / desc_norms
    Dinit = np.dot(ndescs,ndescs.T)
    # Rescaling
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin)
    L = np.copy(D)
    for i in range(L.shape[0]):
      L[i,L[i].argsort()[:-k]] = 0
    LLT = np.maximum(L,L.T)

    # Build dataset options
    InitEmbeddings = ndescs
    AdjMat = LLT*mask
    Degrees = np.diag(np.sum(AdjMat,0))
    TrueEmbedding = np.concatenate(perm_mats,axis=0)
    Ahat = AdjMat + np.eye(*AdjMat.shape)
    Dhat_invsqrt = np.diag(1/np.sqrt(np.sum(Ahat,0)))
    Laplacian = np.dot(Dhat_invsqrt, np.dot(Ahat, Dhat_invsqrt))

    return {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': v,
      'NumPoints': n,
    }

