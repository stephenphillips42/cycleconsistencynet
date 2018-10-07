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
    Saver.__init__(self, ['desc', 'pos', 'scale', 'orien', 'point', 'cam'], id_num, values)

class Camera(Saver):
  def __init__(self, id_num, values=None):
    Saver.__init__(self, ['rot', 'trans', 'focal', 'features' ], id_num, values)
    self.rot = np.eye(3)
    self.trans = np.zeros(3)

class Point(Saver):
  def __init__(self, id_num, values=None):
    Saver.__init__(self, [ 'pos', 'color', 'features' ], id_num, values)
    self.pos = np.zeros(3)
    self.color = np.zeros(3, dtype='int32')

class Scene(Saver):
  def __init__(self, id_num=0):
    Saver.__init__(self, [ 'points', 'cams', 'features' ], id_num)

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
    self.cams = [ Camera(c['id'], c) for c in scene_dict['cams'] ]
    self.points = [ Point(p['id'], p) for p in scene_dict['points'] ]
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

