import os
import sys
import subprocess
import csv

import numpy as np
import matplotlib.pyplot as plt

def get_logs(yaml_name):
  # raw = subprocess.check_output(["kubectl", "get", "pods", "--show-all"])
  raw = subprocess.check_output(["kubectl", "get", "pods"])
  decoded = [ z.split() for z in raw.decode("utf-8").split("\n") ]
  log_names = [ w[0] for w in decoded[:-1] if os.path.splitext(yaml_name)[0] in w[0] ]
  logs_bytes = [ subprocess.check_output(["kubectl", "logs", log_name]) for log_name in log_names ]
  return [ lb.decode("utf-8").split("\n") for lb in logs_bytes ]

def get_loss_values(log):
  loss = []
  for x in log:
    line = x.split()
    if '=' in line:
      loss.append(float(line[line.index('=')+1]))
  return loss


losses = []
for n in range(1,len(sys.argv)):
  yaml_name = sys.argv[n]
  name = os.path.splitext(yaml_name)[0]
  logs = get_logs(yaml_name)
  if len(logs) > 1:
    for l in range(len(logs)):
      loss = np.array(get_loss_values(logs[l]))
      losses.append((loss,"{}-{}".format(name,l),name))
  elif len(logs) == 0:
    print("ERROR - {} has no logs".format(yaml_name))
    sys.exit(1)
  else:
    loss = np.array(get_loss_values(logs[0]))
    losses.append((loss,name,name))

print([ len(loss) for loss, name, _ in losses])
path="/home/stephen/cycleconsistencynet/save/save"
for loss, name, pathname in losses:
  dirpath = "../logs/save-{}".format(name)
  if not os.path.exists(dirpath):
    subprocess.call([
      "scp",
      "-r",
      "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(path,pathname),
      dirpath
    ])
    np.save(os.path.join(dirpath, 'loss.npy'), loss)

for i in range(len(losses)):
  plt.plot(losses[i][0], label=losses[i][1])
plt.legend()
plt.show()
print("\n".join([
        "{}: {} ({})".format(n, l[-1], len(l)) for l, n, _ in losses
      ]))

