# -*- coding: utf-8 -*-
import os
import sys
import collections
import signal
import time
import numpy as np

import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog                 
                 
import data_util.datasets
import model
import myutils
import tfutils
import options

log_file = None
def log(string):
  tf.logging.info(string)
  log_file.write(string)
  log_file.write('\n')

def train(opts):
  pass
  dataset = data_util.datasets.get_dataset(opts)
  # with tf.device('/cpu:0'):
  sample = dataset.load_batch('train')


if __name__ == "__main__":
  opts = options.get_opts()
  log_file = open(os.path.join(opts.save_dir, 'logfile.log'), 'a')
  train(opts)
  log_file.close()


