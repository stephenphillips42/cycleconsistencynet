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


def get_loss_values(log):
  loss_vals = {
    'train': [],
    'test': [],
    'gt_l1': [],
    'gt_l2': [],
  }
  for x in log:
    line = x.split()
    if '=' in line: # Marks where the loss value is
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
assert(all([ x[-len('.yaml'):] == '.yaml' for x in opts.experiments ]))
print("Getting data...")
names = [ x[:-len('.yaml')] for x in opts.experiments ]

save_path="/home/stephen/cycleconsistencynet/save/save"
losses = []
for name in names:
  print("Loss {}".format(name))
  logs_path = os.path.join("..","logs","save-{}".format(name))
  if not os.path.exists(logs_path):
    print([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(save_path,name),
      logs_path
    ])
    subprocess.call([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(save_path,name),
      logs_path
    ])
    with open(os.path.join(logs_path, 'logfile.log'), 'r') as log:
      loss_vals = get_loss_values(log)
    losses.append((loss_vals, name))
    np.save(os.path.join(logs_path, 'loss.npy'), loss_vals['train'])
    np.save(os.path.join(logs_path, 'test_loss.npy'), loss_vals['test'])
    for k in (set(loss_vals.keys()) - {'train','test'}):
      np.save(os.path.join(logs_path, '{}_loss.npy'.format(k)), loss_vals[k])

loss_keys = sorted(list(set().union(*( set(l.keys()) for l, _ in losses ))), reverse=True)
fig, ax = plt.subplots(nrows=1, ncols=len(loss_keys))
for a, k in enumerate(loss_keys):
  for i in range(len(losses)):
    if k in losses[i][0]:
      ax[a].plot(losses[i][0][k], label=losses[i][1])
  ax[a].set_title('{} Loss'.format(k.title()))
  ax[a].legend()

plt.show()

