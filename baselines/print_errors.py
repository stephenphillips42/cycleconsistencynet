import os
import sys
import numpy as np

import matplotlib.pyplot as plt

def myformat(x):
  y = "{:.03e}".format(x).split('e')
  return "{}e-{}".format(y[0], y[1][-1])

for fname in sys.argv[1:]:
  f = open(fname, 'r')
  ids = []
  overlap = []
  precision = []
  recall = []
  l1 = []
  l2 = []
  bce = []
  for line in f:
    l = line.split(' ')
    ids.append(int(l[0]))
    l1.append(float(l[3][:-1]))
    l2.append(float(l[5][:-1]))
    bce.append(float(l[7]))

  f.close()

  overlap = np.array(overlap)
  precision = np.array(precision)
  recall = np.array(recall)
  l1 = np.array(l1)
  l2 = np.array(l2)
  bce = np.abs(np.array(bce))
  # plt.scatter(l1, l2)
  # plt.show()
  # plt.scatter(l2, bce)
  # plt.show()

#   print("{} ({} points)".format(fname, len(l1)))
#   print("L1 : {:.6e} +/- {:.6e} (min: {:.6e}, max: {:.6e})".format(l1.mean(), l1.std(), l1.min(), l1.max()))
#   print("L2 : {:.6e} +/- {:.6e} (min: {:.6e}, max: {:.6e})".format(l2.mean(), l2.std(), l2.min(), l2.max()))
#   print("BCE: {:.6e} +/- {:.6e} (min: {:.6e}, max: {:.6e})".format(bce.mean(), bce.std(), bce.min(), bce.max()))

  # Latex
  fstr = "{}           & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {}"
  print(fstr.format(fname, 
                    myformat(l1.mean()), myformat(l1.std()),
                    myformat(l2.mean()), myformat(l2.std()),
                    myformat(bce.mean()), myformat(bce.std())))
