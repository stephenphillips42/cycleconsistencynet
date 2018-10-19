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
  loss = []
  test_loss = []
  gt_loss = []
  for x in log:
    line = x.split()
    if '=' in line:
      if 'Test' in line:
        test_loss_val = line[line.index('=')+1]
        if test_loss_val[-1] not in '0123456789.':
          test_loss_val = test_loss_val[:-1]
        test_loss.append(float(test_loss_val))
        if 'GT' in line:
          gt_loss_val = line[line.index('GT') + 2]
          gt_loss.append(float(gt_loss_val))
      else:
        loss.append(float(line[line.index('=')+1]))
  return loss, test_loss, gt_loss

opts = get_opts()
print("Getting data...")
losses = []
log_names = get_log_names(opts)
for n in range(len(opts.experiments)):
  yaml_name = opts.experiments[n]
  name = os.path.splitext(yaml_name)[0]
  logs = get_logs(log_names, name)
  if len(logs) > 1:
    for l in range(len(logs)):
      train_loss, test_loss, gt_loss = np.array(get_loss_values(logs[l]))
      loss_name ="{}-{}".format(name,l)
      losses.append((train_loss, test_loss, gt_loss, loss_name, name))
  elif len(logs) == 0:
    print("ERROR - {} has no logs".format(yaml_name))
    sys.exit(1)
  else:
    loss, test_loss, gt_loss = np.array(get_loss_values(logs[0]))
    losses.append((loss, test_loss, gt_loss, name, name))

print([ (name, len(l), len(tl), len(gtl)) for l, tl, gtl, name, _ in losses ])
path="/home/stephen/cycleconsistencynet/save/save"
for loss, test_loss, gt_loss, loss_name, path_name in losses:
  print("Loss {}".format(loss_name))
  dirpath = "../logs/save-{}".format(loss_name)
  if not os.path.exists(dirpath):
    subprocess.call([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(path,path_name),
      dirpath
    ])
    np.save(os.path.join(dirpath, 'loss.npy'), loss)
    np.save(os.path.join(dirpath, 'test_loss.npy'), test_loss)
    if gt_loss:
      np.save(os.path.join(dirpath, 'gt_loss.npy'), gt_loss)

nplots = 2
if any([ loss_vals[2] != [] for loss_vals in losses ]):
  nplots = 3
fig, ax = plt.subplots(nrows=1, ncols=nplots)
for i in range(len(losses)):
  ax[0].plot(losses[i][0], label=losses[i][3])
  ax[1].plot(losses[i][1], label=losses[i][3])
  if nplots == 3:
    ax[2].plot(losses[i][2], label=losses[i][3])
ax[0].legend()
ax[1].legend()
if len(ax) > 2:
  ax[2].legend()
plt.show()
print("\n".join([
        "{}, {}: {} ({})".format(n, l[-1], tl[-1], len(l)) for l, tl, _, n, _ in losses
      ]))

