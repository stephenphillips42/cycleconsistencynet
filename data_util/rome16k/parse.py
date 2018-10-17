import numpy as np 
import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gzip
import pickle
import yaml

from data_util.rome16k import scenes

# Format:
# dict: {'train', 'test'}
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
    # These two comprize ~37% of the total training data
    # '26.2.0.1': (75225, 135),
    # '4.5.0.0': (79259, 251)
  },
  'test' : {
    '11.2.0.0': (12, 40),
    '125.0.0.0': (21, 34),
    '41.0.0.0': (22, 54),
    '37.0.0.0': (25, 46),
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
  }
}
bundle_files = sorted([ k for k in bundle_file_info['test'].keys() ] + \
                      [ k for k in bundle_file_info['train'].keys() ])

def check_valid_name(bundle_file):
  return bundle_file in bundle_files

def scene_name(bundle_file):
  if not check_valid_name(bundle_file):
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  return 'scene.{}.pkl'.format(bundle_file)

def triplets_name(bundle_file, lite=False):
  if not check_valid_name(bundle_file):
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  if lite:
    return 'triplets_lite.{}.pkl'.format(bundle_file)
  else:
    return 'triplets.{}.pkl'.format(bundle_file)

def ktuples_name(bundle_file):
  if not check_valid_name(bundle_file):
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  else:
    return 'ktuples.{}.pkl'.format(bundle_file)

def parse_sift_gzip(fname):
  with gzip.open(fname) as f:
    f_list = f.read().decode().split('\n')[:-1]
  n = (len(f_list)-1)//8
  meta = f_list[0]
  feature_list = []
  for k in range(n):
    sift_ = [ [ float(z) for z in x.split(' ') if z != '' ] for x in f_list[(8*k+1):(8*k+9)] ]
    feature = scenes.Feature(0) # To fill in ID later
    feature.pos = np.array(sift_[0][:2])
    feature.scale = np.array(sift_[0][2])
    feature.orien = np.array(sift_[0][3])
    feature.desc = np.array(sum(sift_[2:], sift_[1]))
    feature_list.append(feature)
  return feature_list

def parse_bundle(bundle_file, top_dir, max_load=-1):
  bundle_dir = os.path.join(top_dir, 'bundle', 'components')
  txtname = os.path.join(bundle_dir, 'bundle.{}.txt'.format(bundle_file))
  outname = os.path.join(bundle_dir, 'bundle.{}.out'.format(bundle_file))
  # Load files
  with open(outname, 'r') as f:
    out_lines = []
    for i, line in enumerate(f.readlines()):
      parsed_line = line[:-1].split(' ')
      if parsed_line[0] == '#':
        continue
      out_lines.append([ float(x) for x in parsed_line ])

  with open(txtname, 'r') as list_file:
    txt_lines = list_file.readlines()
  # Load all SIFT features
  print("Getting feature lists...")
  feature_lists = []
  for k, f in enumerate(txt_lines):
    if k == max_load:
      break
    parse = f[:-1].split(' ')
    fname = parse[0][len('images/'):-len('.jpg')] + ".key.gz"
    db_file = os.path.join(top_dir, 'db/{}'.format(fname))
    if os.path.exists(db_file):
      feature_list = parse_sift_gzip(db_file)
    else:
      query_file = os.path.join(top_dir, 'query/{}'.format(fname))
      feature_list = parse_sift_gzip(query_file)
    feature_lists.append(feature_list)
  print("Done")

  meta = out_lines[0]
  num_cams = int(meta[0])
  num_points = int(meta[1])
  # Extract features
  print("Getting cameras...")
  cams = []
  for i in range(num_cams):
    cam_lines = out_lines[(1+5*i):(1+5*(i+1))]
    cam = scenes.Camera(i)
    cam.focal = cam_lines[0][0]
    cam.rot = np.array(cam_lines[1:4])
    cam.trans = np.array(cam_lines[4])
    cam.features = []
    err = (np.linalg.norm(np.dot(cam.rot.T, cam.rot)-np.eye(3)))
    if err > 1e-9:
      print((i,err))
    cams.append(cam)
  print("Done")
  # Extract points/features
  print("Getting points and features...")
  points = []
  features = []
  start = 1+5*num_cams
  for i in range(num_points):
    lines = out_lines[(start+3*i):(start+3*(i+1))]
    # Construct point
    point = scenes.Point(i)
    point.pos = np.array(lines[0])
    point.color = np.array(lines[1])
    point.features = []
    # Construct feature links
    cam_list = [ int(x) for x in lines[2][1::4] ]
    feat_list = [ int(x) for x in lines[2][2::4] ]
    for cam_id, feat_id in zip(cam_list, feat_list):
      # Create feature
      # TODO: Figure this out
      # # There was an recurring theme that came up in some of the files that
      # # They referened feature ids that simply didn't exist... I fixed this 
      # # by just skipping them but I don't know why it happened and it is
      # # not documented online
      # if feat_id > len(feature_lists[cam_id]):
      #   print('feat_id: {}'.format(feat_id))
      #   print('cam_id: {} (len: {})'.format(cam_id, len(feature_lists[cam_id])))
      #   continue
      feature = feature_lists[cam_id][feat_id]
      feature.cam = cams[cam_id]
      feature.point = point
      feature.id = len(features)
      # Connect feature to camera and point
      cams[cam_id].features.append(feature)
      point.features.append(feature)
      features.append(feature)
    points.append(point)
  print("Done")

  # Create save
  scene = scenes.Scene()
  scene.cams = cams
  scene.points = points
  scene.features = features

  return scene

def save_scene(scene, filename, verbose=False):
  scene_dict = scene.save_out_dict()
  if verbose:
    print("Saving scene...")
  with open(filename, 'wb') as f:
    pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
  if verbose:
    print("Done")

def load_scene(filename, verbose=False):
  scene = scenes.Scene(0)
  if verbose:
    print("Loading pickle file...")
  with open(filename, 'rb') as f:
    scene_dict = pickle.load(f)
  if verbose:
    print("Done")
    print("Parsing pickle file...")
  scene.load_dict(scene_dict)
  if verbose:
    print("Done")
  return scene




