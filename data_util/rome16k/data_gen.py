import numpy as np 
import scipy.linalg as la
import argparse
import pickle

from data_util.rome16k import scenes
from data_util.rome16k import parse

def generate_samples_fixed_knn(opts, bundle_file):
  scene = parse.load_scene(opts.load)
  cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
  print("Loading triples...")
  triplets_fname = parse.triplets_name(opts.load, lite=opts.triplets_lite)
  with open(triplets_fname, 'rb') as f:
    triplets = np.array(pickle.load(f))
  print(len(triplets))
  print("Building similiarty matrices...")
  k = opts.knn
  n = opts.points[-1]
  v = opts.views[-1]
  mask = np.kron(np.ones((v,v))-np.eye(v),np.ones((n,n)))
  for triplet in triplets:
    point_set = cam_pt(triplet[0]) & cam_pt(triplet[1]) & cam_pt(triplet[2])
    feat_perm = np.random.permutation(len(point_set))[:n]
    features = [ 
        sorted([ ([ f for f in p.features if f.cam.id == camid  ])[0] for p in point_set ],
               key=lambda x: x.id)[feat_perm]
        for camid in triplet ]
    descs_ = [ np.array([ f.desc for f in feats ]) for feats in features ]
    rids = [ np.random.permutation(len(ff)) for ff in descs_ ]
    perm_mats = [ np.eye(len(perm))[perm] for perm in rids ]
    perm = la.block_diag(*perm_mats)
    descs = np.dot(perm,np.concatenate(descs_))
    desc_norms = np.sum(descs**2, 1).reshape(-1, 1)
    ndescs = descs / desc_norms
    Dinit = np.dot(ndescs,ndescs.T)
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

    # Mask objects
    yield {
      'InitEmbeddings': InitEmbeddings.astype(self.dtype),
      'AdjMat': AdjMat.astype(self.dtype),
      'Degrees': Degrees.astype(self.dtype),
      'Laplacian': Laplacian.astype(self.dtype),
      'TrueEmbedding': TrueEmbedding.astype(self.dtype),
      'NumViews': v,
      'NumPoints': n,
    }

  



