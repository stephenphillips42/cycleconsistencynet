import numpy as np 
import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gzip
import pickle
import yaml

from data_util import scenes

bundle_files =  [
    '0.0.0.0', '0.1.0.0', '0.2.0.0', '0.3.0.0', '4.0.0.0', '4.1.0.0',
    '4.2.0.0', '4.3.0.0', '4.4.0.0', '4.5.0.0', '4.6.0.0', '4.7.0.0',
    '4.8.0.0', '5.0.0.0', '5.1.0.0', '5.2.0.0', '5.3.0.0', '5.3.0.0',
    '5.8.0.0', '5.9.0.0', '6.0.0.0', '9.0.0.0', '9.1.0.0', '10.0.0.0',
    '11.0.0.0', '11.1.0.0', '11.2.0.0', '12.0.0.0', '15.0.0.0', '16.0.0.0',
    '17.0.0.0', '20.0.0.0', '26.1.0.0', '26.2.0.0', '26.4.0.0', '26.5.0.0',
    '29.0.0.0', '33.0.0.0', '34.0.0.0', '34.1.0.0', '36.0.0.0', '37.0.0.0',
    '38.0.0.0', '38.3.0.0', '40.0.0.0', '41.0.0.0', '46.0.0.0', '49.0.0.0',
    '54.0.0.0', '54.0.0.0', '55.0.0.0', '56.0.0.0', '57.0.0.0', '60.0.0.0',
    '65.0.0.0', '73.0.0.0', '74.0.0.0', '82.0.0.0', '84.0.0.0', '97.0.0.0'
    '110.0.0.0', '122.0.0.0', '125.0.0.0', '135.0.0.0', '167.0.0.0'
  ]

def scene_name(bundle_file):
  if bundle_file not in bundle_files:
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  return 'scene.{}.pkl'.format(bundle_file)

def triplets_name(bundle_file, lite=False):
  if bundle_file not in bundle_files:
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  if lite:
    return 'triplets_lite.{}.pkl'.format(bundle_file)
  else:
    return 'triplets.{}.pkl'.format(bundle_file)

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

def parse_bundle(bundle_file, max_load=-1):
  txtname = '../bundle/components/bundle.{}.txt'.format(bundle_file)
  outname = '../bundle/components/bundle.{}.out'.format(bundle_file)
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
    if os.path.exists("../db/{}".format(fname)):
      feature_list = parse_sift_gzip("../db/{}".format(fname))
    else:
      feature_list = parse_sift_gzip("../query/{}".format(fname))
    feature_lists.append(feature_list)
  print("Done")
  # TODO: Make this ID system work
  # TODO: But really though
  # fcumlens = np.cumsum(feature_lists_lens)[1:].tolist()
  # fcumlens.insert(0,0)

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
      # feature.id = fcumlens[cam_id] + feat_id
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

def save_scene(scene, bundle_file):
  scene_dict = scene.save_out_dict(scene_name(bundle_file))
  print("Saving scene...")
  with open(scene_name(bundle_file), 'wb') as f:
    pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

def load_scene(bundle_file):
  scene = scenes.Scene(0)
  print("Loading pickle file...")
  with open(scene_name(bundle_file), 'rb') as f:
    scene_dict = pickle.load(f)
  print("Done")
  print("Parsing pickle file...")
  scene.load_dict(scene_dict)
  print("Done")
  return scene




