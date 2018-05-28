import os
import sys
import subprocess
import csv

import numpy as np
import matplotlib.pyplot as plt

def get_log(yaml_name):
  raw = subprocess.check_output(["kubectl", "get", "pods", "--show-all"])
  decoded = [ z.split() for z in raw.decode("utf-8").split("\n") ]
  log_names = [ w[0] for w in decoded[:-1] if os.path.splitext(yaml_name)[0] in w[0] ]
  logs_bytes = [ subprocess.check_output(["kubectl", "logs", log_name]) for log_name in log_names ]
  l = max(range(len(logs_bytes)),key=lambda i: len(logs_bytes[i]))
  return logs_bytes[l].decode("utf-8").split("\n")

def get_loss_values(log):
  loss = []
  for x in log:
    line = x.split()
    if '=' in line:
      loss.append(float(line[5]))
  return loss


losses = []
for n in range(1,len(sys.argv)):
  yaml_name = sys.argv[n]
  name = os.path.splitext(yaml_name)[0]
  log = get_log(yaml_name)
  loss = np.array(get_loss_values(log))
  losses.append((loss,name))

print([ len(loss) for loss, name in losses])
path="/home/stephen/cycleconsistencynet/save/save"
for _, name in losses:
  if not os.path.exists("../logs/save-{}".format(name)):
    subprocess.call(["scp", "-r", "stephen@kostas-ap.grasp.upenn.edu:{}-{}".format(path,name), "../logs"])

for i in range(len(losses)):
  plt.plot(losses[i][0], label=losses[i][1])
plt.legend()
plt.show()
print([ loss[-1] for loss, name in losses ])

