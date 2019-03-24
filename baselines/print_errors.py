import os
import sys
import collections
import copy
import numpy as np
import yaml
import tqdm

import matplotlib.pyplot as plt

egstr = '000008 Errors: L1: 1.430e-02, L2: 5.834e-03, BCE: 5.971e-02, ' \
        'Same sim: 4.281e-01 +/- 1.905e-01, Diff sim: 7.239e-03 +/- 3.317e-02, ' \
        'Area under ROC: 9.161e-01, Area under P-R: 7.879e-01, ' \
        'Time: 2.072e+00'

def stdagg(x):
  return np.sqrt(np.mean(np.array(x)**2))

def myformat2(x):
  return '{:.05e}'.format(x)

def myformat(x):
  return '{:.03f}'.format(x)

def myformat_old(x):
  y = "{:.03e}".format(x).split('e')
  return "{}e-{}".format(y[0], y[1][-1])

# TODO: Error checking
def append_to(vals, v):
  if type(vals) == dict:
    for key in vals:
      append_to(vals[key], v[key])
  elif type(vals) == list:
    vals.append(v)

def gen_agg_dict(default_value=[]):
  return {
    **{ k: copy.deepcopy(default_value)
        for k in ['l1', 'l2', 'roc', 'p_r', 'time'] },
    **{ k: { 'm': copy.deepcopy(default_value),
             'std': copy.deepcopy(default_value) }
        for k in ['ssame', 'sdiff'] },
  }
#   ['l1', 'l2', 'bce', 'roc', 'p_r', 'time']
#   'ssame_m', 'ssame_s', 'sdiff_m', 'sdiff_s', \
def agg(vals):
  aggs = gen_agg_dict(0)
  for k in [ 'l1', 'l2', 'roc', 'p_r', 'time' ]:
    aggs[k] = (np.mean(vals[k]), np.std(vals[k]))
  for k in [ 'ssame', 'sdiff' ]:
    aggs[k] = ( np.mean(vals[k]['m']), stdagg(vals[k]['std']) )
  return aggs

def disp_str(fname, aggs):
  fstr = "{:60s}, L1: {} +/- {} , L2: {} +/- {} , ROC: {} +/- {} , P-R: {} +/- {} , Time: {} +/- {}"
  fmtstr = fstr.format(fname, 
                       myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
                       myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
                       myformat(aggs['roc'][0]), myformat(aggs['roc'][1]),
                       myformat(aggs['p_r'][0]), myformat(aggs['p_r'][1]),
                       myformat(aggs['time'][0]), myformat(aggs['time'][1]))
  return fmtstr

def latex_str(fname, aggs):
  # Latex
  fstr = "{:40} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} & {} $\pm$ {} \\\\ \\hline"
  fmtstr = fstr.format(fname, 
                       myformat(aggs['l1'][0]), myformat(aggs['l1'][1]),
                       myformat(aggs['l2'][0]), myformat(aggs['l2'][1]),
                       myformat(aggs['roc'][0]), myformat(aggs['roc'][1]),
                       myformat(aggs['p_r'][0]), myformat(aggs['p_r'][1]),
                       myformat(aggs['time'][0]), myformat(aggs['time'][1]))
  return fmtstr


use_latex = True
disp_strings = collections.OrderedDict()
for fname in tqdm.tqdm(sys.argv[1:]):
  agg_dict = gen_agg_dict()
  with open(fname, 'r') as f:
    yml = yaml.load(f)
  it = 0
  for num, row in yml.items():
    vals_ = row
    append_to(agg_dict, row)
  # Latex
  aggs = agg(agg_dict)
  if use_latex:
    disp_strings[fname] = latex_str(fname, aggs)
  else:
    disp_strings[fname] = disp_str(fname, aggs)

topstr_latex = \
    "Method                                   &" \
    " $L_1$             &" \
    " $L_2$             &" \
    " Area under ROC    &" \
    " Area Prec.-Recall &" \
    " Time (sec)        \\ \hline"
if use_latex:
  print(topstr_latex)
for fname, dstr in disp_strings.items():
  print(dstr)

