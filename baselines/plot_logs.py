import os
import sys
import numpy as np
import re

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Plot the downloaded model training/testing curves')
parser.add_argument('files', metavar='files', nargs='+',
                    help='Files to plot. If concatenating files in list put between parenthesis')

def errorfill(x, y, yerr, color=None, label=None, alpha_fill=0.3, semilogy=None, ax=None):
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = ax._get_lines.color_cycle.next()
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
  if semilogy:
    ax.semilogy(x, y, color=color)
  else:
    ax.plot(x, y, color=color)
  ax.fill_between(x, ymax, ymin, color=color, label=label, alpha=alpha_fill)

def myord(x):
  if x == 'M':
    return 'A'
  elif x == 'P':
    return 'B'
  elif x == 'S':
    return 'C'
  elif x == 'N':
    return 'D'
  else:
    return 'Z'
    
def stdagg(x, axis=0):
  return np.sqrt(np.mean(np.array(x)**2, axis=axis))

def myformat(x):
  return '{:.05e}'.format(x)

def myformat2(x):
  return '{:.03f}'.format(x)

def myformat_old(x):
  y = "{:.03e}".format(x).split('e')
  return "{}e-{}".format(y[0], y[1][-1])

def get_name(nms):
  nm = nms[0]
  if nm[-len('.log'):] == '.log':
    nm = nm[:-len('.log')]
  if nm[-len('TestErrors'):] == 'TestErrors':
    nm = nm[:-len('TestErrors')]
  if nm[len('Iter'):] == 'Iter':
    nm = nm[:-len('000Iter')]
  return nm

dfmt = '\d+'
d_match = re.compile(dfmt)
def get_iters(nms):
  return [ int(d_match.findall(nm)[0]) for nm in nms ]

efmt = '[-+]?\d+\.\d*e[-+]\d+'
disp_match = re.compile(efmt)
names = ['l1', 'l2', 'bce', 'ssame_m', 'ssame_s', 'sdiff_m', 'sdiff_s', 'time']
def parse(line):
  return dict(zip(names, [ float(x) for x in disp_match.findall(line) ]))

def get_values(files):
  vls = dict(zip(names, [ [] for nm in names ]))
  for fname in files:
    vls_ = dict(zip(names, [ [] for nm in names ]))
    f = open(fname, 'r')
    for line in f:
      vv = parse(line)
      for k, v in vv.items():
        vls_[k].append(v)
    for k, v in vls_.items():
      vls[k].append(v)
  # nm_iters = [ get_iters(x) for x in files ]
  return vls, get_name(files)

def parse_files(argfiles):
  inside = False
  fvals = {}
  fnames = []
  cur_files = []
  for argfile in argfiles:
    if inside:
      if argfile == ']':
        inside = False
        vls, fnm  = get_values(cur_files)
        niters = get_iters(cur_files)
        fvals[fnm] = { k : np.array(vls[k]) for k in vls.keys() }
        fvals[fnm]['niters'] = niters
        cur_files = []
      else:
        cur_files.append(argfile)
    else:
      if argfile == '[':
        inside = True
      else:
        vls, fnm = get_values([argfile])
        fvals[fnm] = { k : np.array(vls[k]) for k in vls.keys() }
        # fvals[fnm]['niters'] = []
  for fname, vals in fvals.items():
    if 'niters' not in vals:
      niters_min, niters_max = 10**14, 0
      for fname2 in fvals.keys():
        if 'niters' in fvals[fname2]:
          niters_max = max(niters_max, np.max(fvals[fname2]['niters']))
          niters_min = min(niters_min, np.min(fvals[fname2]['niters']))
      vals['niters'] = [ niters_min, niters_max ]
      for k in vls.keys():
        if k != 'niters':
          vals[k] = np.array([ vals[k][0], vals[k][0] ])
  return fvals

agg_keys = [ 'l1', 'l2', 'time' ] # + [ 'ssame', 'sdiff' ]
def agg(vals):
  aggs = dict(zip(agg_keys, [ None for nm in agg_keys ]))
  for k in [ 'l1', 'l2', 'bce', 'time' ]:
    aggs[k] = (np.mean(vals[k], 1), np.std(vals[k], 1))
  # for k in [ 'ssame', 'sdiff' ]:
  #   aggs[k] = ( np.mean(vals[k + '_m'], 1), stdagg(vals[k + '_s'], 1) )
  return aggs


yaxis_name = { 'l1' : 'Error' , 'l2' : 'Error', 'time' : 'Time (sec)' }
# TODO: Maybe figure out how to make this more general
labels = { 'MatchALS010Iter' : 'MatchALS',
           'PGDDS010Iter' : 'PGDDS',
           'NormedSkip2Geom10-3View' : 'GCN (ours), 12 layers',
           'Spectral' : 'Spectral', }
fonttitle = {'fontsize':20, 'fontname':'Times New Roman'}
fontaxis = {'fontsize':12, 'fontname':'Times New Roman'}
fontlegend = {'fontsize':13, 'fontname':'Times New Roman'}

args = parser.parse_args()
fvals = parse_files(args.files)
fig, ax = plt.subplots(nrows=1, ncols=len(fvals))
ii = 0
for fname, vals in fvals.items():
  ax[ii].hist(vals['l1'][-1])
  ax[ii].set_title(fname)
  ii += 1
plt.show()
colors_ = [ 'r', 'y', 'm', 'b', 'g' ]
fnames = sorted(list(fvals.keys()), key=lambda x: myord(x[0]))
color_map = dict(zip(fnames, colors_[:len(fnames)]))
fig, ax_ = plt.subplots(nrows=1, ncols=len(agg_keys))
ax = dict(zip(agg_keys, ax_))
for fname in fnames:
  vals = fvals[fname]
  aggs = agg(vals)
  for k in agg_keys:
    mm, mstd = aggs[k]
    oo = np.ones_like(mm) # + np.random.randn(*mm.shape)*0.01
    # ax[k].errorbar(vals['niters']*oo, mm, yerr=mstd)
    errorfill(vals['niters']*oo, mm, yerr=mstd,
              color=color_map[fname], label=labels[fname],
              semilogy=(k == 'time'), ax=ax[k])

for k in agg_keys:
  ax[k].set_title(k.title(), fontdict=fonttitle)
  ax[k].set_xlabel('Iterations', **fontaxis)
  ax[k].set_ylabel(yaxis_name[k], **fontaxis)
lgd = ax[agg_keys[0]].legend()
plt.setp(lgd.texts, **fontlegend)
plt.show()



