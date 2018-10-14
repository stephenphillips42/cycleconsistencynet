import os
import sys
import subprocess
import csv

import numpy as np
import matplotlib.pyplot as plt

def get_logs(yaml_name):
  # raw = subprocess.check_output(["kubectl", "get", "pods", "--show-all"])
  check_name = os.path.splitext(yaml_name)[0]
  if len(check_name) > 56:
    check_name = check_name[:56]
  raw = subprocess.check_output(["kubectl", "get", "pods"])
  decoded = [ z.split() for z in raw.decode("utf-8").split("\n") ]
  log_names = [ w[0] for w in decoded[:-1] if check_name in w[0] ]
  logs_bytes = [ subprocess.check_output(["kubectl", "logs", log_name]) for log_name in log_names ]
  return [ lb.decode("utf-8").split("\n") for lb in logs_bytes ]

def get_loss_values(log):
  loss = []
  test_loss = []
  for x in log:
    line = x.split()
    if '=' in line:
      if 'Test' in line:
        test_loss.append(float(line[line.index('=')+1]))
      else:
        loss.append(float(line[line.index('=')+1]))
  return loss, test_loss


losses = []
for n in range(1,len(sys.argv)):
  yaml_name = sys.argv[n]
  name = os.path.splitext(yaml_name)[0]
  logs = get_logs(yaml_name)
  if len(logs) > 1:
    for l in range(len(logs)):
      loss, test_loss = np.array(get_loss_values(logs[l]))
      losses.append((loss,test_loss,"{}-{}".format(name,l),name))
  elif len(logs) == 0:
    print("ERROR - {} has no logs".format(yaml_name))
    sys.exit(1)
  else:
    loss, test_loss = np.array(get_loss_values(logs[0]))
    losses.append((loss, test_loss, name, name))

print([ (len(loss), len(test_loss)) for loss, test_loss, name, _ in losses ])
path="/home/stephen/cycleconsistencynet/save/save"
for loss, test_loss, name, pathname in losses:
  dirpath = "../logs/save-{}".format(name)
  if not os.path.exists(dirpath):
    subprocess.call([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(path,pathname),
      dirpath
    ])
    np.save(os.path.join(dirpath, 'loss.npy'), loss)
    np.save(os.path.join(dirpath, 'test_loss.npy'), test_loss)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
for i in range(len(losses)):
  ax0.plot(losses[i][0], label=losses[i][2])
  ax1.plot(losses[i][1], label=losses[i][2])
ax0.legend()
ax1.legend()
plt.show()
print("\n".join([
        "{}, {}: {} ({})".format(n, l[-1], tl[-1], len(l)) for l, tl, n, _ in losses
      ]))

