import os
import sys
import subprocess
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

def get_opts():
  """Parse arguments from command line and get all options for training."""
  parser = argparse.ArgumentParser(description='Train motion estimator')
  # Directory and dataset options
  parser.add_argument('--experiments',
                      default=[],
                      nargs='+',
                      help='Experiments to test data to run analysis on')
  parser.add_argument('--exclude',
                      default=[],
                      nargs='+',
                      help='Experiments to exclude')

  opts = parser.parse_args()
  return opts

def in_set(x, S):
  for s in S:
    if x in s:
      return True
  return False

def get_log_names(opts):
  experiments = []
  for name in opts.experiments:
    yaml_name = os.path.splitext(name)[0]
    if len(yaml_name) > 56:
      yaml_name = yaml_name[:56]
    experiments.append(yaml_name) 
  raw = subprocess.check_output(["kubectl", "get", "pods"])
  decoded = [ z.split() for z in raw.decode("utf-8").split("\n") ]
  log_names_all = [ w[0] for w in decoded[:-1] ]
  exclude = [ 'kssh-session', 'NAME' ] + opts.exclude
  log_names = [ w for w in log_names_all if not in_set(w, exclude) ]
  return log_names

def get_logs(log_names, yaml_name):
  sel_names = [ w for w in log_names if yaml_name in w  ]
  logs_bytes = [ subprocess.check_output(["kubectl", "logs", log_name]) for log_name in sel_names ]
  return [ lb.decode("utf-8").split("\n") for lb in logs_bytes ]

def get_loss_values(log):
  loss_vals = {
    'train': [],
    'test': [],
    'gt_l1': [],
    'gt_l2': [],
  }
  for x in log:
    line = x.split()
    if '=' in line: # Marks where the 
      if 'Test' in line:
        loss_vals['test'].append(float(line[line.index('=')+1]))
        if 'L1' in line:
          loss_vals['gt_l1'].append(float(line[line.index('L1') + 3]))
        if 'L2' in line:
          loss_vals['gt_l2'].append(float(line[line.index('L2') + 3]))
      else:
        loss_vals['train'].append(float(line[line.index('=')+1]))
  for k in list(loss_vals.keys()):
    if len(loss_vals[k]) == 0:
      del loss_vals[k]
    else:
      loss_vals[k] = np.array(loss_vals[k])
  return loss_vals

opts = get_opts()
print("Getting data...")
losses = []
log_names = get_log_names(opts)
for n in range(len(opts.experiments)):
  yaml_name = opts.experiments[n]
  name = os.path.splitext(yaml_name)[0]
  logs = get_logs(log_names, name)
  if len(logs) > 0:
    for l in range(len(logs)):
      loss_vals = get_loss_values(logs[l])
      if len(logs) > 1:
        loss_name ="{}-{}".format(name,l)
      else:
        loss_name = name
      losses.append((loss_vals, loss_name, name))
  elif len(logs) == 0:
    print("ERROR - {} has no logs".format(yaml_name))
    sys.exit(1)

print([ (name, ( "{}: {}".format(k,len(v)) for k, v in l.items() )) for l, name, _ in losses ])
path="/home/stephen/cycleconsistencynet/save/save"
for loss_vals, loss_name, path_name in losses:
  print("Loss {}".format(loss_name))
  dirpath = "../logs/save-{}".format(loss_name)
  if not os.path.exists(dirpath):
    subprocess.call([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(path,path_name),
      dirpath
    ])
    np.save(os.path.join(dirpath, 'loss.npy'), loss_vals['train'])
    np.save(os.path.join(dirpath, 'test_loss.npy'), loss_vals['test'])
    for k in (set(loss_vals.keys()) - {'train','test'}):
      np.save(os.path.join(dirpath, '{}_loss.npy'.format(k)), loss_vals[k])

print(losses)
loss_keys = sorted(list(set().union(*( set(l.keys()) for l, _, _ in losses ))), reverse=True)
fig, ax = plt.subplots(nrows=1, ncols=len(loss_keys))
for a, k in enumerate(loss_keys):
  for i in range(len(losses)):
    if k in losses[i][0]:
      ax[a].plot(losses[i][0][k], label=losses[i][1])
  ax[a].set_title('{} Loss'.format(k.title()))
  ax[a].legend()

plt.show()
print("\n".join([
        "{}, {}: {} ({})".format(n, l['train'][-1], l['test'][-1], len(l['train']))
        for l, n, _ in losses
      ]))

