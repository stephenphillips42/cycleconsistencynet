import numpy as np 
import os
import sys
import gzip
import pickle
import tqdm

import requests
from PIL import Image
import html.parser
import io

from data_util.rome16k import scenes

# Format:
# dict: {'train', 'test'}
#   -> dict: rome16k_name -> (ntriplets, ncams)
bundle_file_info = {
  'train' : {
    '5.1.0.0': (507,),
    '20.0.0.0': (566,),
    '55.0.0.0': (644,),
    '38.0.0.0': (663,),
    '26.1.0.0': (744,),
    '74.0.0.0': (1050,),
    '49.0.0.0': (1053,),
    '36.0.0.0': (1204,),
    '12.0.0.0': (1511,),
    '60.0.0.0': (2057,),
    '54.0.0.0': (2068,),
    '57.0.0.0': (2094,),
    '167.0.0.0': (2119,),
    '4.11.0.0': (2714,),
    '38.3.0.0': (3248,),
    '135.0.0.0': (3476,),
    '4.8.0.0': (3980,),
    '110.0.0.0': (4075,),
    '4.3.0.0': (4442,),
    '29.0.0.0': (4849,),
    '97.0.0.0': (4967,),
    '4.6.0.0': (5409,),
    '84.0.0.0': (5965,),
    '9.1.0.0': (6536,),
    '33.0.0.0': (6698,),
    '15.0.0.0': (9950,),
    '26.5.0.0': (12913,),
    '122.0.0.0': (15269,),
    '10.0.0.0': (16709,),
    '11.0.0.0': (16871,),
    '0.0.0.0': (22632,),
    '17.0.0.0': (28333,),
    '16.0.0.0': (35180,),
    '4.1.0.0': (36460,),
    '4.4.0.0': (-1,),
    # These two comprize ~37% of the total training data
    # '26.2.0.1': (75225,),
    # '4.5.0.0': (79259,)
  },
  'test' : {
    '11.2.0.0': (12,),
    '125.0.0.0': (21,),
    '41.0.0.0': (22,),
    '37.0.0.0': (25,),
    '73.0.0.0': (26,),
    '33.0.0.1': (31,),
    '5.11.0.0': (93,),
    '0.3.0.0': (170,),
    '46.0.0.0': (205,),
    '26.4.0.0': (239,),
    '82.0.0.0': (256,),
    '65.0.0.0': (298,),
    '40.0.0.0': (340,),
    '56.0.0.0': (477,),
    '5.9.0.0': (481,),
    '34.1.0.0': (487,),
  }
}
bundle_files = sorted([ k for k in bundle_file_info['test'].keys() ] + \
                      [ k for k in bundle_file_info['train'].keys() ])

# Methods for getting image size
URL_STR = 'http://www.flickr.com/photo_zoom.gne?id={}'
class ImageSizeHTMLParser(html.parser.HTMLParser):
  def __init__(self):
    super().__init__()
    self.image_url = None
  
  def handle_starttag(self, tag, attrs):
    attr_keys = [ at[0] for at in attrs ]
    attr_vals = [ at[1] for at in attrs ]
    if tag == 'a' and 'href' in attr_keys:
      link = attr_vals[attr_keys.index('href')]
      if '.jpg' in link:
        self.image_url = link
PARSER = ImageSizeHTMLParser()

def get_image_size(imname):
  # photoid = imname[:-len('.jpg')].split('/')[-1].split('_')[-1]
  photoid = imname[:-len('.jpg')].split('_')[-1]
  url_ = URL_STR.format(photoid)
  ret = requests.get(url_)
  PARSER.feed(ret.text)
  im_ = requests.get(PARSER.image_url)
  image = Image.open(io.BytesIO(im_.content))
  return image.size


# Methods for getting filenames
def check_valid_name(bundle_file):
  return bundle_file in bundle_files

def scene_fname(bundle_file):
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

def tuples_fname(bundle_file):
  if not check_valid_name(bundle_file):
    print("ERROR: Specified bundle file does not exist: {}".format(bundle_files))
    sys.exit(1)
  else:
    return 'tuples.{}.pkl'.format(bundle_file)

# Main parsing functions
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

def parse_bundle(bundle_file, top_dir, get_imsize=True, max_load=-1, verbose=False):
  if verbose:
    myprint = lambda x: print(x)
  else:
    myprint = lambda x: 0
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
  myprint("Getting feature lists...")
  feature_lists = []
  imsize_list = []
  for k, f in tqdm.tqdm(enumerate(txt_lines), total=len(txt_lines), disable=not verbose):
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
    if get_imsize:
      imsize_list.append(get_image_size(parse[0]))
  myprint("Done")

  meta = out_lines[0]
  num_cams = int(meta[0])
  num_points = int(meta[1])
  # Extract features
  myprint("Getting cameras...")
  cams = []
  for i in range(num_cams):
    cam_lines = out_lines[(1+5*i):(1+5*(i+1))]
    cam = scenes.Camera(i)
    cam.focal = cam_lines[0][0]
    cam.k1 = cam_lines[0][1]
    cam.k2 = cam_lines[0][2]
    if get_imsize:
      cam.imsize = imsize_list[i]
    cam.rot = np.array(cam_lines[1:4])
    cam.trans = np.array(cam_lines[4])
    cam.features = []
    err = (np.linalg.norm(np.dot(cam.rot.T, cam.rot)-np.eye(3)))
    if err > 1e-9:
      myprint((i,err))
    cams.append(cam)
  myprint("Done")
  # Extract points/features
  myprint("Getting points and features...")
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
      #   myprint('feat_id: {}'.format(feat_id))
      #   myprint('cam_id: {} (len: {})'.format(cam_id, len(feature_lists[cam_id])))
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
  myprint("Done")

  # Create save
  scene = scenes.Scene()
  scene.cams = cams
  scene.points = points
  scene.features = features

  return scene

# Main outward facing functions
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




