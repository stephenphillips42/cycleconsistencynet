import os
import sys
import numpy as np
import sklearn.metrics as metrics

def test():
  with open(os.path.join(sys.argv[1], 'y_true.npy'), 'rb') as f:
    y_true = np.load(f)
  with open(os.path.join(sys.argv[1], 'y_pred.npy'), 'rb') as f:
    y_pred = np.load(f)
  roc =  metrics.roc_auc_score(y_true, y_pred)
  p_r =  metrics.average_precision_score(y_true, y_pred)
  # import pdb; pdb.set_trace()
  precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
  fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
  print("{:.06e} {:.06e}".format(roc,p_r))
  print(precision.shape)
  print(recall.shape)
  with open(os.path.join(sys.argv[1], 'precision.npy'), 'wb') as f:
    np.save(f, precision)
  with open(os.path.join(sys.argv[1], 'recall.npy'), 'wb') as f:
    np.save(f, recall)
  with open(os.path.join(sys.argv[1], 'fpr.npy'), 'wb') as f:
    np.save(f, fpr)
  with open(os.path.join(sys.argv[1], 'tpr.npy'), 'wb') as f:
    np.save(f, tpr)

test() 



