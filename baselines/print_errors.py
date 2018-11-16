import os
import sys
import numpy as np
import re

import matplotlib.pyplot as plt

egstr = '000000 Errors: L1: 5.297e-02, L2: 1.678e-02, BCE: 1.008e-01, Same sim: 8.776e-01 +/- 9.812e-02, Diff sim: 5.209e-02 +/- 1.182e-01, '

def stdagg(x):
  return np.sqrt(np.mean(np.array(x)**2))

def myformat(x):
  return '{:.05e}'.format(x)

def myformat2(x):
  return '{:.03f}'.format(x)

def myformat_old(x):
  y = "{:.03e}".format(x).split('e')
  return "{}e-{}".format(y[0], y[1][-1])

efmt = '[-+]?\d+\.\d*e[-+]\d+'
disp_match = re.compile(efmt)
names = ['l1', 'l2', 'bce', 'ssame_m', 'ssame_s', 'sdiff_m', 'sdiff_s']
aggregators = [ np.mean, np.mean, np.mean, np.mean, stdagg, np.mean, stdagg ]
def parse(line):
  return dict(zip(names, [ float(x) for x in disp_match.findall(line) ]))

agg_names = [ 'l1', 'l2', 'bce', 'ssame', 'sdiff' ]
def agg(vals):
  aggs = dict(zip(agg_names, [ None for nm in agg_names ]))
  for k in [ 'l1', 'l2', 'bce' ]:
    aggs[k] = (np.mean(vals[k]), np.std(vals[k]))
  for k in [ 'ssame', 'sdiff' ]:
    aggs[k] = ( np.mean(vals[k + '_m']), stdagg(vals[k + '_s']) )
  return aggs

def disp_val(aggs):
  # fstr = "{:40}, L1: {} +/- {} , L2: {} +/- {} , BCE: {} +/- {}"
  # print(fstr.format(fname, 
  #                   myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
  #                   myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
  #                   myformat(aggs['bce'][0]), myformat(aggs['bce'][1])))
  # return 
  fstr = "{:40} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {}"
  print(fstr.format(fname, 
                    myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
                    myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
                    myformat(aggs['bce'][0]), myformat(aggs['bce'][1])))



for fname in sys.argv[1:]:
  vals = dict(zip(names, [ [] for nm in names ]))
  f = open(fname, 'r')
  for line in f:
    vals_ = parse(line)
    for k, v in vals_.items():
      vals[k].append(v)

  f.close()

  # Latex
  aggs = agg(vals)
  disp_val(aggs)
