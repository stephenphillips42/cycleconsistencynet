import numpy as np 
import os
import sys
import argparse
import pickle
import tqdm

from data_util.rome16k import scenes
from data_util.rome16k import parse
from data_util.rome16k import myutils

def get_build_scene_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  parser.add_argument('--build_triplets',
                      type=myutils.str2bool,
                      default=False,
                      help='Name of pickle file to load - None if no loading')
  parser.add_argument('--save',
                      choices=bundle_files,
                      default=bundle_files[0],
                      help='Save out bundle file to pickle file')

  opts = parser.parse_args()
  return opts


opts = get_build_scene_opts()
scene = parse.parse_bundle(opts.save)
parse.save_scene(scene, opts.save)
# Triplet managment
n = len(scene.cams)
cam_pt = lambda i: set([ f.point for f in scene.cams[i].features ])
idx_lst = np.array([ (i,j,k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n) ])
triplets = []
if opts.build_triplets:
  print("Building triples...")
  lst = []
  for i, j, k in tqdm.tqdm(idx_lst):
    p = len(cam_pt(i) & cam_pt(j) & cam_pt(k))
    lst.append(p)
  lst = np.array(lst)
  print("Done")
  print("Selecting triplets...")
  triplets = idx_lst[(lst >= 80) & (lst <= 150)]
  with open(triplets_name(opts.save),'wb') as f:
    pickle.dump(triplets.tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
  with open(triplets_name(opts.save, lite=True),'wb') as f:
    pickle.dump(triplets[:100].tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

print("Done")





