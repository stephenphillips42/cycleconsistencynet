import numpy as np 
import os
import sys
import argparse
import pickle
import tqdm
import time
import itertools as it

from data_util.rome16k import scenes
from data_util.rome16k import parse
import myutils

def get_build_scene_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  parser.add_argument('--build_ktuples',
                      type=myutils.str2bool,
                      default=True,
                      help='Name of pickle file to load - None if no loading')
  parser.add_argument('--top_dir',
                      default='/NAS/data/stephen/Rome16K',
                      help='Storage location for pickle files')
  parser.add_argument('--save',
                      choices=parse.bundle_files,
                      default=parse.bundle_files[0],
                      help='Save out bundle file to pickle file')
  parser.add_argument('--min_points',
                      default=80,
                      type=int,
                      help='Minimum overlap of points for connection')
  parser.add_argument('--max_points',
                      default=150,
                      type=int,
                      help='Minimum overlap of points for connection')
  parser.add_argument('--max_k',
                      default=4,
                      type=int,
                      help='Maximum tuple size')

  opts = parser.parse_args()
  return opts

def factorial(n, stop=0):
  o = 1
  while n > stop:
    o *= n
    n -= 1
  return o

def choose(n, k):
  return factorial(n, stop=k) // factorial(n - k)

opts = get_build_scene_opts()
scene = parse.parse_bundle(opts.save, opts.top_dir)
filename=os.path.join(opts.top_dir,'scenes','scene.{}.pkl'.format(opts.save))
parse.save_scene(scene, filename)
# Triplet managment
if not opts.build_ktuples:
  sys.exit(0)

n = len(scene.cams)
cam_pts = lambda i: set([ f.point for f in scene.cams[i].features ])
ktuples_full = []
ktuples_sizes = []
# Length 2 is a special case
print("Building pairs...")
start_time = time.time()
pairs, sizes = [], []
for x in tqdm.tqdm(it.combinations(range(n),2), total=choose(n,2)):
  p = len(cam_pts(x[0]) & cam_pts(x[1]))
  if p >= opts.min_points:
    pairs.append(x)
    sizes.append(p)
ktuples_full.append(pairs)
ktuples_sizes.append(sizes)
end_time = time.time()
print("Done with pairs ({} sec)".format(end_time-start_time))
# Length 3 and above
for k in range(3,opts.max_k+1):
  print("Selecting {}-tuples...".format(k))
  start_time = time.time()
  klist, ksizes = [], []
  kvals = ktuples_full[-1]
  for (i, x) in tqdm.tqdm(enumerate(kvals), total=len(kvals)):
    xpts = cam_pts(x[0])
    for xx in x[1:]:
      xpts = xpts & cam_pts(xx)
    for j in range(x[-1]+1,n):
      p = len(cam_pts(j) & xpts)
      if p >= opts.min_points:
        klist.append(x + (j,))
        ksizes.append(p)
  ktuples_full.append(klist)
  ktuples_sizes.append(ksizes)
  end_time = time.time()
  print("Done with {}-tuples ({} sec)".format(k, end_time-start_time))

ktuples = [ [ x for i, x in enumerate(ktups) if ksizes[i] <= opts.max_points ]
            for ktups, ksizes in zip(ktuples_full, ktuples_sizes) ]

print("Saving tuples...")
ktuples_fname=os.path.join(opts.top_dir,'scenes',parse.ktuples_name(opts.save))
with open(ktuples_fname,'wb') as f:
  pickle.dump(ktuples, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(triplets_name(opts.save, lite=True),'wb') as f:
#   pickle.dump(triplets[:100].tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
print("Done")

print("Finished")





