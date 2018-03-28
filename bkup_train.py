# -*- coding: utf-8 -*-
import numpy as np 
import os
import sys
import collections
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
from tqdm import tqdm

import torch
import torch.utils.data as tdata
from torch.autograd import Variable
import torch.nn.functional as functional

import myutils
import options
import data_util
# import model

class MyLogger(object):
  def __init__(self, logfile_name):
    self.logfile = open(logfile_name, 'w')

  def log(self, message):
    print(message)
    self.logfile.write(message + '\n')

  def __del__(self):
    self.logfile.close()

class GCNModel(torch.nn.Module):
  def __init__(self, opts):
    super(GCNModel, self).__init__()
    lens = [ opts.descriptor_dim ] + \
           [ 2**5, 2**6, 2**7, 2**8 ] + \
           [ opts.final_embedding_dim ]
    self._linear = []
    for i in range(len(lens)-1):
      name = '_linear{:02d}'.format(i)
      layer = torch.nn.Linear(lens[i], lens[i+1])
      self.__setattr__(name, layer)
      self._linear.append(layer)
    self.normalize = opts.normalize_embedding

  def save(self, directory, prefix):
    for i in range(len(self._linear)):
      wi = self._linear[i]._parameters['weight'].data.numpy()
      bi = self._linear[i]._parameters['bias'].data.numpy()
      np.save(os.path.join(directory, "{}_w{:02d}.npy".format(prefix,i)), wi)

  def forward(self, x):
    out = Variable(x[0])
    lap = Variable(x[1])
    for i in range(len(self._linear)-1):
      out = torch.matmul(lap, self._linear[i](out)).clamp(min=0)
    out = self._linear[-1](out)
    if self.normalize:
      return functional.normalize(out,dim=-1)
    else:
      return out

class Criterion(object):
  def __init__(self, opts, debug_dir='figs/', debug_exit=False, debug_show=True):
    self.offset = opts.embedding_offset
    self.dist_w = opts.embedding_distance_weight
    self.normalize = opts.normalize_embedding
    self.debug_dir = debug_dir
    self.prefix = "e0i0"
    self.debug_exit = debug_exit 

  def eval(self, output, sample):
    dists = myutils.pairwise_distances(output)
    weight_mask = Variable(sample['Mask'][0])
    weight_offset = Variable(sample['MaskOffset'][0])
    err = self.offset*(weight_offset) + self.dist_w*(dists*weight_mask)
    if sample['debug']:
      self.debug(output, sample)
    normalizer = (len(weight_mask)**2) 
    if self.normalize: # DEBUG - remove eventually
      normalizer = 1.0
    return torch.sum(err.clamp(min=0))/normalizer

  def debug(self, output, sample):
    wm = weight_mask.data.numpy()
    wo = weight_offset.data.numpy()
    emb = output.data.numpy()
    d = dists.data.numpy()
    tt = sample['TrueEmbedding'][0].numpy()
    td = 2*(1-np.dot(tt,tt.T))
    import matplotlib.pyplot as plt
    def debug_out(name, x): 
      plt.imshow(x)
      plt.savefig(os.path.join(self.debug_dir,
                  "{}_{}.png".format(self.prefix, name)))
      np.save("{}_{}.npy".format(self.prefix,name), x)

    debug_out("weight_mask", wm)
    debug_out("dists", d)
    debug_out("true_dists", td)
    debug_out("true_sims", np.dot(tt.T,tt))
    debug_out("embedding", np.dot(emb.T,emb))
    debug_out("true_embed_corr", np.dot(tt.T,emb))
    debug_out("true_emb", tt)
    debug_out("embedding", emb)
    if self.debug_exit:
      sys.exit()
    

class SimilarityCriterion(object):
  def __init__(self, opts, debug_dir='figs/', debug_exit=False, debug_show=True):
    self.loss = torch.nn.MSELoss()
    self.debug_dir = debug_dir
    self.prefix = "e0i0"
    self.debug_exit = debug_exit 

  def eval(self, output, sample):
    sims_est = torch.mm(output, torch.transpose(output, 0, 1))
    sims_ = sample['AdjMat'][0] + torch.eye(len(sample['AdjMat'][0]))
    sims = Variable(sims_)
    err = self.loss(sims_est, sims)
    if sample['debug']:
      self.debug(output, sample)
    return err

  def debug(self, output, sample):
    emb = output.data.numpy()
    tt = sample['TrueEmbedding'][0].numpy()
    ts = np.dot(tt,tt.T)
    initemb = sample['InitEmbeddings'][0]
    import matplotlib.pyplot as plt
    def debug_out(name, x): 
      plt.imshow(x)
      plt.savefig(os.path.join(self.debug_dir,
                  "{}_{}.png".format(self.prefix, name)))
      np.save("{}_{}.npy".format(self.prefix,name), x)
    debug_out("true_emb", tt)
    debug_out("embedding", emb)
    debug_out("initemb", initemb)
    if self.debug_exit:
      sys.exit()

def train(opts):
  logger = MyLogger('stdout.log')
  # Get data
  train_dir = os.path.join(opts.data_dir, 'train')
  test_dir = os.path.join(opts.data_dir, 'test')
  dataset = data_util.GraphSimDataset(opts,
                                      opts.num_gen_train,
                                      n_pts=opts.min_points,
                                      n_poses=opts.min_views)
  loader = tdata.DataLoader(dataset, batch_size=1,shuffle=True)
  testset = data_util.GraphSimDataset(opts,
                                      opts.num_gen_test,
                                      n_pts=opts.min_points,
                                      n_poses=opts.min_views)
  test_loader = tdata.DataLoader(testset, batch_size=1,shuffle=True)
  # Get model and optimizer
  # model = GCNModel(opts)
  model = GCNModel(opts)
  criterion = SimilarityCriterion(opts,
                                  debug_dir=opts.save_dir,
                                  debug_exit=False, debug_show=False)
  optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
  optimizer.zero_grad()
  l = 0
  for epoch in range(opts.num_epochs):
    l = 0
    tl = 0
    with tqdm(total=len(test_loader),ncols=79) as pbar:
      for idx, sample in enumerate(test_loader):
        pbar.update(1)
        sample['debug'] = False
        lap = torch.eye(len(sample['Degrees'][0])) + \
              torch.diag(sample['Degrees'][0]) - sample['AdjMat'][0]
        output = model.forward((sample['InitEmbeddings'][0], lap))
        loss_ = criterion.eval(output,sample)
        l += loss_.data[0]
        # tl += criterion.eval_true(sample)
    logger.log("\n\nTest Loss: {}\n\n".format(l / len(test_loader)))
    l = 0
    for idx, sample in enumerate(loader):
      sample['debug'] = False
      if (epoch == 0 and idx == 3) or (epoch == 2 and idx == len(loader)-2):
        sample['debug'] = True
        criterion.prefix = 'e{}i{}'.format(epoch,idx)
        if idx == len(loader)-2:
          criterion.debug_exit = True
        model.save(criterion.debug_dir, criterion.prefix)
      lap = torch.eye(len(sample['Degrees'][0])) + \
            torch.diag(sample['Degrees'][0]) - sample['AdjMat'][0]
      output = model.forward((sample['InitEmbeddings'][0], lap))
      loss_ = criterion.eval(output,sample)
      loss_.backward()
      l += loss_.data[0]
      if idx > 0 and (idx % opts.batch_size) == 0:
        if ((idx // opts.batch_size) % opts.print_freq) == 0:
          logger.log("Loss {:08d}: {}".format(idx, l / opts.batch_size))
        optimizer.step()
        optimizer.zero_grad()
        l = 0
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
  opts = options.get_opts()
  train(opts)

