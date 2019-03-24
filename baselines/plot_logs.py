import os
import sys
import numpy as np
import yaml
import copy
import tqdm

import matplotlib.pyplot as plt

import argparse

# Debug printing
def process(x):
  xp = None
  if type(x) == dict:
    xp = {}
    for k in x:
      xp[k] = process(x[k])
  elif type(x) == list:
    xp = [ process(z) for z in x ]
  elif type(x) == np.ndarray:
    xp = ( x.shape, x.dtype )
  else:
    xp = x
  return xp
import pprint
pp_xfawedfssa = pprint.PrettyPrinter(indent=2)
def myprint(x):
  if type(x) == str:
    print(x)
  else:
    pp_xfawedfssa.pprint(process(x))

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

def disp_plot(y, x=None, color=None, label=None, alpha=0.25, semilogy=None, ax=None):
  """Fancy mean/median/percentile plotter. y is presumed NxS, x length N."""
  if x is None:
    x = np.arange(len(y))
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = ax._get_lines.color_cycle.next()
  ymean = np.mean(y,1)
  ymin, ymax, ymid_b, ymid_u, ymed = np.percentile(y, [ 2, 98, 25, 75, 50 ], axis=1)
  ax.plot(x,ymean,color=color,linestyle='-', label=label)
  # ax.plot(x,ymed,color=color,linestyle='--')
  ax.fill_between(x,ymid_b,ymid_u,facecolor=color,alpha=alpha)
  # ax.fill_between(x,ymin,ymax,facecolor=color,alpha=alpha**2)
  return ax


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

def get_info(fname):
  fname = os.path.basename(fname)
  if 'MatchALS' in fname or 'PGDDS' in fname:
    k = str.find(fname, 'Iter')
    niters = int(fname[k-3:k])
    k = str.find(fname, 'View')
    views = int(fname[k-2:k])
    return { 'iters': [niters], 'views': views }
  elif 'Spectral' in fname or 'Random' in fname:
    k = str.find(fname, 'View')
    views = int(fname[k-2:k])
    return { 'iters': [0], 'views': views }
  elif 'rome16kgeom' in fname:
    k = str.find(fname, 'skiphop')
    k = str.find(fname, 'view')
    views = int(fname[k-2:k])
    if 'skiphop0' in fname:
      niters = 8
    elif 'skiphop1' in fname:
      niters = 12
    elif 'skiphop2' in fname:
      niters = 16
    else:
      niters = 2
    return { 'iters': [niters], 'views': views }
  else:
    return { }

def get_title(k):
  if k == 'p_r':
    return 'AUC Precision-Recall'
  elif k == 'roc':
    return 'AUC ROC'
  else:
    return k.title()

def get_label(fname):
  if 'MatchALS' in fname:
    return 'MatchALS'
  elif 'PGDDS' in fname:
    return 'PGDDS'
  elif 'Spectral' in fname:
    return 'Spectral'
  elif 'Random' in fname:
    return 'Random'
  elif 'rome16kgeom' in fname:
    return 'Ours (6 Passes)'
  else:
    return ''

def get_color(label):
  if label == 'MatchALS':
    return 'r'
  elif label == 'PGDDS':
    return 'g'
  elif label == 'Spectral':
    return 'm'
  elif label == 'Random':
    return 'y'
  elif label == 'Ours (6 Passes)':
    return 'b'
  else:
    return 'y'

def gen_agg_dict(default_value=[]):
  return {
    **{ k: copy.deepcopy(default_value)
        for k in ['l1', 'l2', 'roc', 'p_r', 'time'] },
    **{ k: { 'm': copy.deepcopy(default_value),
             'std': copy.deepcopy(default_value) }
        for k in ['ssame', 'sdiff'] },
  }

# TODO: Error checking
def append_to(vals, v):
  if type(vals) == dict:
    for key in vals:
      append_to(vals[key], v[key])
  elif type(vals) == list:
    vals.append(v)

def npify(vals, transpose=False):
  if type(vals) == dict:
    for key in vals:
      vals[key] = npify(vals[key])
    return vals
  elif type(vals) == list:
    if transpose:
      return np.array(vals).T
    else:
      return np.array(vals)

# def concat_aggs(aggs):
#   aggs_c = { k: { k0: {} for k0 in [ 'mean', 'std' ] } for k in aggs }
#   for k in aggs:
#     for t, v in aggs[k][0].items():
#       aggs_c[k]['mean'][t] = [ v['mean'] ]
#       aggs_c[k]['std'][t] = [ v['std'] ]
#     for a in aggs[k][1:]:
#       for t, v in a.items():
#         aggs_c[k]['mean'][t].append(v['mean'])
#         aggs_c[k]['std'][t].append(v['std'])
#   return aggs_c
# 
# def aggregate(vals):
#   aggs = { k: { k0: [] for k0 in [ 'mean', 'std' ] } for k in vals }
#   for k in [ 'l1', 'l2', 'roc', 'p_r', 'time' ]:
#     aggs[k]['mean'] = np.mean(vals[k])
#     aggs[k]['std'] = np.std(vals[k])
#   for k in [ 'ssame', 'sdiff' ]:
#     aggs[k]['mean'] = np.mean(vals[k]['m'])
#     aggs[k]['std'] = stdagg(vals[k]['std'])
#   return aggs

def parse_files(argfiles):
  iters_view = {}
  aggs_view = {}
  # Get info
  for fname in argfiles:
    info_ = get_info(fname)
    # Extract number of views
    v = info_['views']
    if v not in iters_view:
      iters_view[v] = {}
    if v not in aggs_view:
      aggs_view[v] = {}
    # Extract number of iterations
    if get_label(fname) in iters_view[v]:
      iters_view[v][get_label(fname)].append(info_['iters'][0])
    else:
      iters_view[v][get_label(fname)] = info_['iters']
    # Load the yaml file to get info
    agg_dict = gen_agg_dict()
    with open(fname, 'r') as f:
      yml = yaml.load(f)
    # Parse file line by line
    for num, row in yml.items():
      vals_ = row
      append_to(agg_dict, row)
    # Add to aggregation keys
    if get_label(fname) not in aggs_view[v]:
      aggs_view[v][get_label(fname)] = gen_agg_dict([])
    append_to(aggs_view[v][get_label(fname)], agg_dict)
  for v in aggs_view:
    aggs_view[v] = npify(aggs_view[v], transpose=False)
  return aggs_view, iters_view


# agg_keys = [ 'l1', 'time' ] # + [ 'ssame', 'sdiff' ]
# def agg(vals):
#   aggs = dict(zip(agg_keys, [ None for nm in agg_keys ]))
#   for k in [ 'l1', 'time' ]:
#     aggs[k] = (np.mean(vals[k], 1), np.std(vals[k], 1))
#   # for k in [ 'ssame', 'sdiff' ]:
#   #   aggs[k] = ( np.mean(vals[k + '_m'], 1), stdagg(vals[k + '_s'], 1) )
#   return aggs


plot_keys = [ 'l1', 'roc'  ]
yaxis_name = {
  'l1': 'Error (lower better)',
  'l2': 'Error (lower better)',
  'time': 'Time (sec)',
  'p_r': 'AUC Prec.-Recall (lower better)',
  'roc': 'AUC ROC (higher better)',
}
fonttitle = {'fontsize':14, 'fontname':'Times New Roman'}
fontaxis = {'fontsize':12, 'fontname':'Times New Roman'}
fontlegend = {'fontsize':13, 'fontname':'Times New Roman'}

args = parser.parse_args()
aggs_view, iters_view = parse_files(args.files)
nviews, nplots = len(iters_view), len(plot_keys)
sz, R, C = 4, nviews, nplots
fig, ax_ = plt.subplots(nrows=R, ncols=C, figsize=(3+C*sz, 3+R*sz))
ax = {
  v: dict(zip(plot_keys, ax_[i]))
  for i, v in enumerate(sorted(iters_view.keys()))
}
miniters = { v: 10**9 for v in iters_view }
maxiters = { v: 0 for v in iters_view }
for v in iters_view:
  for iters_ in iters_view[v].values():
    if len(iters_) > 1 or iters_[0] > 0:
      miniters[v] = min(miniters[v], min(iters_))
      maxiters[v] = max(maxiters[v], max(iters_))

for v in sorted(iters_view.keys()):
  iters = iters_view[v]
  for label, agg_vals in aggs_view[v].items():
    for k in plot_keys:
      means = []
      stds = []
      # ax[k].errorbar(vals['niters']*oo, mm, yerr=mstd)
      if len(iters[label]) == 1:
        y = agg_vals[k]
        x = np.array([ miniters[v], maxiters[v] ])
        y = np.concatenate([ y, y ])
      else:
        x = iters[label]
        y = agg_vals[k]
      disp_plot(y, x=x,
                color=get_color(label),
                label=label,
                ax=ax[v][k],
                alpha=0.2)
      title = '{} ({} views)'.format(get_title(k), v)
      ax[v][k].set_title(title, fontdict=fonttitle)
      ax[v][k].set_xlabel('Iterations', **fontaxis)
      ax[v][k].set_ylabel(yaxis_name[k], **fontaxis)
  # if 'time' in plot_keys:
  #   ax[v]['time'].set_yscale('log')
  if 'roc' in plot_keys:
    ax[v]['roc'].set_ylim([0.75, 1])

lgd = ax_.reshape(-1)[0].legend()
plt.setp(lgd.texts, **fontlegend)
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))


plt.show()

