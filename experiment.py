import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import myutils

def myload(fdir,name):
  fnames = sorted(glob.glob(os.path.join(fdir,'e*i*_{}.npy'.format(name))))
  return np.load(fnames[-1])

def load_all(fdir):
  true_emb = myload(fdir,'true_emb')
  ids = np.sum(np.arange(10) * true_emb,1)
  order = np.argsort(ids)
  initemb = myload(fdir,'initemb')
  initemb_ord = myutils.dim_normalize(initemb[order,:])
  emb = myload(fdir,'embedding')
  emb_ord = emb[order,:]

  return {
    'true_emb': true_emb,
    'ids': ids,
    'order': order,
    'initemb': initemb_ord,
    'emb': emb_ord,
  }

def main(fdir):
  sample = load_all(fdir)
  initemb = sample['initemb']
  initemb_nrm = np.dot(myutils.dim_norm(initemb),np.ones((1,len(initemb))))
  plt.imshow(-2*np.dot(initemb, initemb.T) + initemb_nrm + initemb_nrm.T)
  plt.colorbar()
  plt.title('Initial distances')
  plt.show()
  plt.imshow(2-2*np.dot(sample['emb'], sample['emb'].T))
  plt.colorbar()
  plt.title('Final distances')
  plt.show()
  plt.imshow(np.dot(sample['emb'], sample['emb'].T))
  plt.colorbar()
  plt.title('Final similarities')
  plt.show()

def soft_matches(fdir):
  sample = load_all(fdir)
  n = int(np.max(np.sum(sample['true_emb'],0)))
  soft = (np.diag(np.ones(n-2),k=-2) + np.diag(np.ones(n-2),k=2))*0.25 + \
         (np.diag(np.ones(n-1),k=-1) + np.diag(np.ones(n-1),k=1))*0.5 + \
         np.eye(n)
  plt.imshow(np.dot(np.dot(sample['true_emb'],soft),sample['true_emb'].T)) 
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  # main(sys.argv[1])
  soft_matches(sys.argv[1])

