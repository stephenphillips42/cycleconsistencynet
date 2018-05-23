import os
import sys
import glob
import numpy as np

import myutils
import options

def npload(fdir,idx):
  return np.load("{}/np_test-{:04d}.npz".format(fdir,idx))

def main(opts):
  ld = npload(os.path.join(opts.data_dir, 'np_test'), opts.debug_index)
  print(ld)

if __name__ == "__main__":
  opts = options.get_opts()
  main(opts)

