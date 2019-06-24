import numpy as np
import scipy.linalg as la
import os
import sys
import glob
import datetime
import random
import tqdm
import networkx as nx

import tensorflow as tf
from graph_nets import utils_np

from data_util import graph_dataset
from data_util import tf_helpers
from data_util.rome16k import parse


class Rome16KGraphDataset(graph_dataset.GraphDataset):
  def __init__(self, opts, params):
    super().__init__(opts, params)
    self.rome16k_dir = opts.rome16k_dir
    self.tuple_size = self.n_views
    self.tuple_file_sizes = self.get_tuple_file_sizes()
    p = self.n_pts
    v = self.n_views
    d = p*v
    self.features.update({
      'adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='adj_mat',
                         shape=[d, d],
                         description='Sparse adjacency matrix of graph'),
      'true_adj_mat':
           tf_helpers.SparseTensorFeature(
                         key='true_adj_mat',
                         shape=[d, d],
                         description='Sparse ground truth adjacency of graph'),
      'true_match':
           tf_helpers.VarLenIntListFeature(
                         key='true_match',
                         dtype='int64',
                         description='Ground truth matches of graph'),
    })

  # Scene/Bundle file functions
  def scene_fname(self, bundle_file):
    return os.path.join(self.rome16k_dir, 'scenes', parse.scene_fname(bundle_file))

  def tuples_fname(self, bundle_file):
    return os.path.join(self.rome16k_dir, 'scenes', parse.tuples_fname(bundle_file))

  # TODO: NEED TO FIX ALL OF THIS
  def get_tuple_file_sizes(self):
    print("Extracting tuples...")
    tuple_file_sizes = { 'train': {}, 'test': {} }
    for mode, bundle_files in parse.bundle_file_split.items():
      for bundle_file in bundle_files:
        fname = self.tuples_fname(bundle_file)
        with open(fname, 'rb') as f:
          ld = np.load(f)
        tuple_file_sizes[mode][bundle_file] = \
            { i+2: len(x) for i, x in enumerate(ld) }
    print(tuple_file_sizes)
    return tuple_file_sizes

  def get_bundle_tuples(self, mode):
    tsize = self.tuple_size
    rndseed = (self.dataset_params.rndseed * hash(mode)) % 2**32
    np.random.seed(rndseed)
    random.seed(rndseed)
    selmode = mode if mode != 'np_test' else 'test' # TODO: Hack...
    bundle_files = [ 
      tname for tname, sizes in self.tuple_file_sizes[selmode].items()
      if tsize in sizes and sizes[tsize] > 0
    ]
    all_tuples = []
    for bundle_file in bundle_files:
      with open(self.tuples_fname(bundle_file), 'rb') as f:
        tuples_ = np.load(f)
      all_tuples.extend(
        [ (bundle_file, tupl) for tupl in tuples_[tsize-2] ]
      )
    print(len(all_tuples))
    sel_random = random.sample(all_tuples, self.dataset_params.sizes[mode])
    bundle_tuples = { bundle_file: [] for bundle_file in bundle_files }
    for bundle_file, tupl in sel_random:
      bundle_tuples[bundle_file].append(tupl)
    return bundle_tuples

  # New Convert dataset function - works quite differently than synth
  def convert_dataset(self, out_dir, mode):
    """Writes synthetic flow data in .mat format to a TF record file."""
    params = self.dataset_params
    fname = '{}-{:02d}.tfrecords'
    outfile = lambda idx: os.path.join(out_dir, fname.format(mode, idx))
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
    # Select tuples to generate
    bundle_tuples = self.get_bundle_tuples(mode)
    # Begin generation
    print('Writing dataset to {}/{}'.format(out_dir, mode))
    writer = None
    scene = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    pbar = tqdm.tqdm(total=params.sizes[mode])
    for bundle_file, tuples in bundle_tuples.items():
      scene_name = self.scene_fname(bundle_file)
      np.random.seed(hash(scene_name) % 2**32)
      scene = parse.load_scene(scene_name)
      for tupl in tuples:
        if file_idx > self.MAX_IDX:
          file_idx = 0
          if writer: writer.close()
          writer = tf.python_io.TFRecordWriter(outfile(record_idx))
          record_idx += 1
        loaded_features = self.gen_sample_from_tuple(scene, tupl)
        features = self.process_features(loaded_features)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
        file_idx += 1
        pbar.update()
    pbar.close()
    if writer: writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(mode)
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('TFrecord created {}'.format(str(datetime.datetime.now())))

  def create_np_dataset(self, out_dir):
    """Create npz files to store dataset"""
    mode = 'np_test'
    fname = 'np_test-{:04d}.npz'
    outfile = lambda idx: os.path.join(out_dir, fname.format(idx))
    print('Writing dataset to {}'.format(out_dir))
    record_idx = 0
    # Select tuples to generate
    bundle_tuples = self.get_bundle_tuples(mode)
    pbar = tqdm.tqdm(total=self.dataset_params.sizes[mode])
    index = 0
    for bundle_file, tuples in bundle_tuples.items():
      scene_name = self.scene_fname(bundle_file)
      np.random.seed(hash(scene_name) % 2**32)
      scene = parse.load_scene(scene_name)
      for tupl in tuples:
        features = self.gen_sample_from_tuple(scene, tupl)
        np_features = {}
        for k, v in self.features.items():
          np_features.update(v.npz_value(features[k]))
        np.savez(outfile(index), **np_features)
        index += 1
        pbar.update()
    pbar.close()
    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(out_dir, timestamp_file), 'w') as date_file:
      date_file.write('Numpy Dataset created {}'.format(str(datetime.datetime.now())))

  # Abstract functions
  def gen_sample(self):
    print("ERROR: Cannot generate sample - need to load data")
    sys.exit(1)

  def gen_sample_from_tuple(self, scene, tupl):
    print("ERROR: Not implemented in abstract base class")
    sys.exit(1)


class GeomKNNRome16KDataset(Rome16KGraphDataset):
  def __init__(self, opts, params):
    super().__init__(opts, params)
    self.features.update({
      'rots':
           tf_helpers.TensorFeature(
                         key='rots',
                         shape=[self.tuple_size, 3, 3],
                         dtype=self.dtype,
                         description='Rotations of the cameras'),
      'trans':
           tf_helpers.TensorFeature(
                         key='trans',
                         shape=[self.tuple_size, 3],
                         dtype=self.dtype,
                         description='Translations of the cameras'),
    })

  def gen_sample_from_tuple(self, scene, tupl):
    # Parameters
    k = self.dataset_params.knn
    n = self.dataset_params.points[-1]
    v = self.dataset_params.views[-1]
    cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
    point_set = set.intersection(*[ cam_pt(t) for t in tupl ])
    # Build features
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [] 
    for camid in tupl:
      fset = [ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ]
      fset = sorted(fset, key=lambda x: x.id)
      features.append([ fset[x] for x in feat_perm ])
    # Build descriptors
    xy_pos_ = [ np.array([ f.pos for f in feats ]) for feats in features ]
    scale_ = [ np.array([ f.scale for f in feats ]) for feats in features ]
    orien_ = [ np.array([ f.orien for f in feats ]) for feats in features ]
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    # Apply permutation to features
    perms = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in perms ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))
    xy_pos = np.dot(perm,np.concatenate(xy_pos_))
    # We have to manually normalize these values as they are much larger than the others
    logscale = np.dot(perm, np.log(np.concatenate(scale_)) - 1.5).reshape(-1,1)
    orien = np.dot(perm,np.concatenate(orien_)).reshape(-1,1) / np.pi
    desc_norms = np.sqrt(np.sum(descs**2, 1).reshape(-1, 1))
    ndescs = descs / desc_norms
    # Finish the inital node embeddings
    init_emb = np.concatenate([ndescs,xy_pos,logscale,orien], axis=1)
    # Build Graph
    Dinit = np.dot(ndescs,ndescs.T)
    Dmin = Dinit.min()
    Dmax = Dinit.max()
    D = (Dinit - Dmin)/(Dmax-Dmin) # Rescaling
    A_ = np.zeros_like(D)
    for i in range(v):
      for j in range(v):
        if i == j:
          continue
        # Perhaps not the most efficient... oh well
        Asub = np.copy(D[n*i:n*(i+1),n*j:n*(j+1)])
        for u in range(n):
          Asub[u,Asub[u].argsort()[:-k]] = 0
        A_[n*i:n*(i+1),n*j:n*(j+1)] = Asub
    adj_mat = np.maximum(A_, A_.T)
    # Build dataset ground truth
    true_emb = np.concatenate(perm_mats,axis=0)
    gt_adj_mat = np.dot(true_emb, true_emb.T)
    matches_ = np.concatenate(perms)
    rots = np.stack([ scene.cams[i].rot.T for i in tupl ], axis=0)
    trans = np.stack([ -np.dot(scene.cams[i].rot.T, scene.cams[i].trans)
                       for i in tupl ], axis=0)
    # Build spart graph representation
    G_nx = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
    node_attrs = { i : init_emb[i].astype(np.float32)
                   for i in range(len(G_nx)) }
    edges_attrs = { (i, j) : np.array([ adj_mat[i,j] ]).astype(np.float32)
                    for (i,j) in G_nx.edges }
    nx.set_node_attributes(G_nx, node_attrs, 'features')
    nx.set_edge_attributes(G_nx, edges_attrs, 'features')
    # Make to dictionary, with additional graph attributes
    G = utils_np.networkx_to_data_dict(G_nx)
    G['globals'] = np.array([0,0])
    G['adj_mat'] = graph_dataset.np_dense_to_sparse(adj_mat)
    G['true_adj_mat'] = graph_dataset.np_dense_to_sparse(gt_adj_mat)
    G['true_match'] = matches_
    G['rots'] = rots
    G['trans'] = trans
    return G



