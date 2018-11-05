import numpy as np 

class Saver(object):
  def __init__(self, fields, id_num, values=None):
    for f in fields:
      setattr(self, f, None)
    self.fields = fields
    self.id = id_num
    if values is not None:
      for f in values.keys() & fields:
        self.__dict__[f] = values[f]

  def get_dict(self):
    dict_ = { f: self.output(self.__dict__[f]) for f in self.fields }
    dict_['id'] = self.id
    return dict_

  def output(self, obj):
    if type(obj) is np.ndarray:
      return obj.tolist()
    elif type(obj) is list:
      return [ self.output(x) for x in obj ]
    elif isinstance(obj, Saver):
      return obj.id
    else:
      return obj


class Feature(Saver):
  def __init__(self, id_num, values=None):
    fields = [ 'pos', 'pos_uncal',
               'desc', 'scale', 'orien',
               'point', 'cam']
    Saver.__init__(self, fields, id_num, values)
    self.pos = np.array(self.pos or np.zeros(3))
    self.pos_uncal = np.array(self.pos_uncal or np.zeros(2))
    self.desc = np.array(self.desc or np.zeros(128))
    self.scale = self.scale or 0
    self.orien = self.orien or 0

  def get_uncentered_calib(self):
    return np.array([ self.pos_uncal[1], -self.pos_uncal[0] ])

  def get_proj_pt(self):
    P = np.dot(self.cam.rot, self.point.pos) + self.cam.trans
    p = -P / P[-1]
    return self.cam.focal*p[:2]

class Camera(Saver):
  def __init__(self, id_num, values=None):
    fields = [ 'rot', 'trans',
               'focal', 'k1', 'k2',
               'imsize', 'center',
               'features' ]
    Saver.__init__(self, fields, id_num, values)
    self.rot = np.array(self.rot or np.eye(3))
    self.trans = np.array(self.trans or np.zeros(3))
    self.focal = self.focal or 1.0
    self.k1 = self.k1 or 0.0
    self.k2 = self.k2 or 0.0
    self.imsize = self.imsize or (-1,-1)
    self.center = np.array(self.center or np.zeros(2))

  def center_points(self):
    p_uncal, p_proj = [], []
    for f in self.features:
      p_uncal.append(f.get_uncentered_calib())
      p_proj.append(f.get_proj_pt())
    p_uncal, p_proj = np.array(p_uncal), np.array(p_proj)
    self.center = np.mean(p_uncal, 0) - np.mean(p_proj, 0)
    for f in self.features:
      f.pos = -(f.get_uncentered_calib() - self.center)/self.focal

class Point(Saver):
  def __init__(self, id_num, values=None):
    fields = ['pos', 'color', 'features']
    Saver.__init__(self, fields, id_num, values)
    self.pos = np.array(self.pos or np.zeros(3))
    self.color = np.array(self.color or np.zeros(3, dtype='int32'))

class Scene(Saver):
  def __init__(self, id_num=0):
    fields = ['points', 'cams', 'features']
    Saver.__init__(self, fields, id_num)

  def get_dict(self):
    # cams = [ c.get_dict() for c in self.cams ]
    # points = [ p.get_dict() for p in self.points ]
    # features = [ f.get_dict() for f in self.features ]
    cams = []
    for c in self.cams:
      cams.append(c.get_dict())
    points = []
    for p in self.points:
      points.append(p.get_dict())
    features = []
    for f in self.features:
      features.append(f.get_dict())
    return {
      'cams' : cams, 'points': points, 'features': features
    }

  def save_out_dict(self):
    points = []
    for p in self.points:
      p_d = p.get_dict()
      p_d['features'] = None
      points.append(p_d)
    cams = []
    for c in self.cams:
      c_d = c.get_dict()
      c_d['features'] = None
      cams.append(c_d)
    features = []
    for f in self.features:
      features.append(f.get_dict())
    return {
      'cams' : cams, 'points': points, 'features': features
    }

  def load_dict(self, scene_dict):
    self.cams = [ Camera(c['id'], values=c) for c in scene_dict['cams'] ]
    self.points = [ Point(p['id'], values=p) for p in scene_dict['points'] ]
    self.features = []
    for f in scene_dict['features']:
      feature = Feature(f['id'], f)
      c_idx = feature.cam
      p_idx = feature.point
      feature.cam = self.cams[c_idx]
      feature.point = self.points[p_idx]
      self.features.append(feature)
      if self.cams[c_idx].features is None:
        self.cams[c_idx].features = [ feature ]
      else:
        self.cams[c_idx].features.append(feature)
      if self.points[p_idx].features is None:
        self.points[p_idx].features = [ feature ]
      else:
        self.points[p_idx].features.append(feature)
    for c in self.cams:
      c.center_points()

