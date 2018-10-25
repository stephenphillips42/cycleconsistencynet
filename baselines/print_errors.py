import os
import sys
import numpy as np


for fname in sys.argv[1:]:
  f = open(fname, 'r')
  ids = []
  overlap = []
  precision = []
  recall = []
  l1 = []
  l2 = []
  for line in f:
    l = line.split(' ')
    ids.append(int(l[0]))
    overlap.append(float(l[3][:-1]))
    precision.append(float(l[5][:-1]))
    recall.append(float(l[7][:-1]))
    l1.append(float(l[9][:-1]))
    l2.append(float(l[12][:-1]))

  f.close()

  overlap = np.array(overlap)
  precision = np.array(precision)
  recall = np.array(recall)
  l1 = np.array(l1)
  l2 = np.array(l2)

  print(fname)
  print("L1: {:.6e} +/- {:.6e}".format(l1.mean(), l1.std()))
  print("L2: {:.6e} +/- {:.6e}".format(l2.mean(), l2.std()))

