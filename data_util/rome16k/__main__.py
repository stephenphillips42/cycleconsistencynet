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
                      choices=parse.bundle_files + [ 'all' ],
                      default='all',
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
  parser.add_argument('--verbose',
                      default=True,
                      type=myutils.str2bool,
                      help='Print out everything')

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

def silent(x):
  pass

def process_scene_bundle(opts, bundle_file, verbose=False):
  if verbose:
    myprint = lambda x: print(x)
  else:
    myprint = lambda x: silent(x)
  ######### Build and save out scene file ###########
  scene = parse.parse_bundle(bundle_file, opts.top_dir)
  filename=os.path.join(opts.top_dir,'scenes','scene.{}.pkl'.format(bundle_file))
  parse.save_scene(scene, filename, verbose)

  if not opts.build_ktuples:
    return 

  ######### Build and save out k-tuples ###########
  n = len(scene.cams)
  cam_pts = lambda i: set([ f.point for f in scene.cams[i].features ])
  ktuples_full = []
  ktuples_sizes = []
  # Length 2 is a special case
  myprint("Building pairs...")
  start_time = time.time()
  pairs, sizes = [], []
  for x in tqdm.tqdm(it.combinations(range(n),2), total=choose(n,2), disable=not verbose):
    p = len(cam_pts(x[0]) & cam_pts(x[1]))
    if p >= opts.min_points:
      pairs.append(x)
      sizes.append(p)
  ktuples_full.append(pairs)
  ktuples_sizes.append(sizes)
  end_time = time.time()
  myprint("Done with pairs ({} sec)".format(end_time-start_time))
  # Length 3 and above
  for k in range(3,opts.max_k+1):
    myprint("Selecting {}-tuples...".format(k))
    start_time = time.time()
    klist, ksizes = [], []
    kvals = ktuples_full[-1]
    for (i, x) in tqdm.tqdm(enumerate(kvals), total=len(kvals), disable=not verbose):
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
    myprint("Done with {}-tuples ({} sec)".format(k, end_time-start_time))

  ktuples = [ [ x for i, x in enumerate(ktups) if ksizes[i] <= opts.max_points ]
              for ktups, ksizes in zip(ktuples_full, ktuples_sizes) ]

  myprint("Saving tuples...")
  ktuples_fname=os.path.join(opts.top_dir,'scenes',parse.ktuples_name(bundle_file))
  with open(ktuples_fname,'wb') as f:
    pickle.dump(ktuples, f, protocol=pickle.HIGHEST_PROTOCOL)
  # with open(triplets_name(bundle_file, lite=True),'wb') as f:
  #   pickle.dump(triplets[:100].tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)
  myprint("Done")



opts = get_build_scene_opts()
if opts.save == 'all':
  N = len(parse.bundle_files)
  for i, bundle_file in enumerate(parse.bundle_files):
    if opts.verbose:
      print('Computing {} ({} of {})...'.format(bundle_file,i,N))
    start_time = time.time()
    process_scene_bundle(opts, bundle_file, verbose=opts.verbose)
    end_time = time.time()
    if opts.verbose:
      print('Finished {} ({:0.3f} sec)'.format(bundle_file,end_time-start_time))
else:
  process_scene_bundle(opts, opts.save, verbose=opts.verbose)

