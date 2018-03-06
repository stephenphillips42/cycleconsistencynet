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
import torch.nn.functional as F

import nputils
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

def meanstd(x):
  return (np.mean(x), np.std(x))

# class GCNNModel(torch.nn.Module):
#   # def __init__(self, opts):
#   #   super(MyModel, self).__init__()
#   #   lens = [ opts.descriptor_dim ] + \
#   #          [ 15, 20, 25, 20 ] + \
#   #          [ opts.final_embedding_dim ]
#   #   self._linear = [ torch.nn.Linear(lens[i], lens[i+1])
#   #                    for i in range(len(lens)-1) ]
#   #   self._activ = [ torch.nn.ReLU() 
#   #                   for i in range(len(lens)-2) ]
# 
#   def __init__(self, opts):
#     super(GCNModel, self).__init__()
#     lens = [ opts.descriptor_dim ] + \
#            [ 2**5, 2**6, 2**7, 2**8 ] + \
#            [ opts.final_embedding_dim ]
#     self._linear0 = torch.nn.Linear(lens[0], lens[1])
#     self._linear1 = torch.nn.Linear(lens[1], lens[2])
#     self._linear2 = torch.nn.Linear(lens[2], lens[3])
#     self._linear3 = torch.nn.Linear(lens[3], lens[4])
#     self._linear4 = torch.nn.Linear(lens[4], lens[5])
#     self._linear = [ 
#         self._linear0,
#         self._linear1,
#         self._linear2,
#         self._linear3,
#         self._linear4,
#     ]
# 
#   def forward(self, x):
#     out = Variable(x[0])
#     lap = Variable(x[1])
#     for i in range(len(self._linear)-1):
#       out = torch.matmul(lap, self._linear[i](out)).clamp(min=0)
#     return self._linear[-1](out)

# TODO: Make this opts independent...
class GCNModelOld(torch.nn.Module):
  def __init__(self, opts):
    super(GCNModelOld, self).__init__()
    lens = [ opts.descriptor_dim ] + \
           [ 2**5, 2**6, 2**7, 2**8 ] + \
           [ opts.final_embedding_dim ]
    self._linear0 = torch.nn.Linear(lens[0], lens[1])
    self._linear1 = torch.nn.Linear(lens[1], lens[2])
    self._linear2 = torch.nn.Linear(lens[2], lens[3])
    self._linear3 = torch.nn.Linear(lens[3], lens[4])
    self._linear4 = torch.nn.Linear(lens[4], lens[5])
    self._linear = [ 
        self._linear0,
        self._linear1,
        self._linear2,
        self._linear3,
        self._linear4,
    ]

  def forward(self, x):
    out = Variable(x[0])
    lap = Variable(x[1])
    for i in range(len(self._linear)-1):
      out = torch.matmul(lap, self._linear[i](out)).clamp(min=0)
    return self._linear[-1](out)

class GCNModel(torch.nn.Module):
  def __init__(self, opts, layer_lens=None, use_normalization=False):
    super(GCNModel, self).__init__()
    if layer_lens is None:
      layer_lens = [ 2**5, 2**6, 2**7, 2**8 ]
    lens = [ opts.descriptor_dim ] + layer_lens + [ opts.final_embedding_dim ]
    self._use_normalization = use_normalization
    self._linear = []
    for i in range(len(lens)-1):
      name = '_linear{:02d}'.format(i)
      layer = torch.nn.Linear(lens[i], lens[i+1])
      self.__setattr__(name, layer)
      self._linear.append(layer)

  def forward(self, x):
    out = Variable(x[0])
    lap = Variable(x[1])
    for i in range(len(self._linear)-1):
      out = torch.matmul(lap, self._linear[i](out)).clamp(min=0)
    final_out = self._linear[-1](out)
    if self._use_normalization:
      return F.normalize(final_out,dim=-1)
    else:
      return final_out

class Criterion(object):
  def __init__(self, opts):
    self.offset = opts.embedding_offset
    print('Embedding offset: {}'.format(self.offset))

  def eval(self, output, sample):
    dists = pairwise_distances(output)
    weight_mask = Variable(sample['Mask'][0])
    weight_offset = Variable(sample['MaskOffset'][0])
    err = self.offset*weight_offset + dists*weight_mask
    return torch.sum(err.clamp(min=0))/(len(weight_mask)**2)

def pairwise_distances(x, y=None, normalized=False):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is None:
      y = x
    if normalized:
      x_norm = 1
      y_norm = 1
    else:
      x_norm = (x**2).sum(1).view(-1, 1)
      if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
      else:
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def train(opts):
  logger = MyLogger('stdout.log')
  # Get data
  train_dir = os.path.join(opts.data_dir, 'train')
  test_dir = os.path.join(opts.data_dir, 'test')
  # dataset = data_util.CycleConsitencyGraphDataset(train_dir)
  dataset = data_util.GraphSimDataset(opts, opts.num_gen_train, n_pts=12, n_poses=25)
  loader = tdata.DataLoader(dataset, batch_size=1,shuffle=True)
  # testset = data_util.CycleConsitencyGraphDataset(test_dir)
  testset = data_util.GraphSimDataset(opts, opts.num_gen_test, n_pts=12, n_poses=25)
  test_loader = tdata.DataLoader(testset, batch_size=1,shuffle=True)
  # Get model and optimizer
  # model = GCNModel(opts)
  model = GCNModel(opts, use_normalization=True)
  criterion = Criterion(opts)
  # print([ x for x in model.parameters()])
  optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
  optimizer.zero_grad()
  l = 0
  for epoch in range(opts.num_epochs):
    l = 0
    with tqdm(total=len(test_loader),ncols=79) as pbar:
      for idx, sample in enumerate(test_loader):
        pbar.update(1)
        lap = torch.eye(len(sample['Degrees'][0])) + \
              torch.diag(sample['Degrees'][0]) - sample['AdjMat'][0]
        output = model.forward((sample['InitEmbeddings'][0], lap))
        loss_ = criterion.eval(output,sample)
        l += loss_.data[0]
    logger.log("\n\nTest Loss: {}\n\n".format(l / len(test_loader)))
    l = 0
    for idx, sample in enumerate(loader):
      lap = torch.eye(len(sample['Degrees'][0])) + \
            torch.diag(sample['Degrees'][0]) - sample['AdjMat'][0]
      output = model.forward((sample['InitEmbeddings'][0], lap))
      loss_ = criterion.eval(output,sample)
      loss_.backward()
      l += loss_.data[0]
      if (idx % opts.batch_size) == 0:
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

