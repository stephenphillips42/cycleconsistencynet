import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import myutils

def main(fdir):
  fnames = sorted(glob.glob(os.path.join(fdir,'e*i*_true_emb.npy')))
  true_emb = np.load(fnames[-1])
  ids = np.sum(np.arange(10) * true_emb,1)
  fnames = sorted(glob.glob(os.path.join(fdir,'e*i*_initemb.npy')))
  initemb = np.load(fnames[-1])
  initemb_ord = np.concatenate([ initemb[ids==i,:] for i in range(10) ])
  initemb_nrm = np.dot(myutils.dim_norm(initemb_ord)**2,np.ones((1,len(initemb))))
  plt.imshow(-2*np.dot(initemb_ord, initemb_ord.T) + initemb_nrm + initemb_nrm.T)
  plt.colorbar()
  plt.show()
  fnames = sorted(glob.glob(os.path.join(fdir,'e*i*_embedding.npy')))
  emb = np.load(fnames[-1])
  emb_lst = np.concatenate([ emb[ids==i,:] for i in range(10) ])
  plt.imshow(np.dot(emb_lst, emb_lst.T))
  plt.colorbar()
  plt.show()


if __name__ == "__main__":
  main(sys.argv[1])

