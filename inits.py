# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:14:24 2022

@author: tuan
"""
from consts import *

import os
import sys
import random
import numpy as np
from matplotlib import pyplot as plt

# show less log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if is_train:
    import tensorflow as tf
    # run on GPU with TF v2
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    # run on CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.set_printoptions(suppress=True)

sys.path.insert(0, "dataset_configs")

random.seed(rand_seed)
np.random.seed(rand_seed)

log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

log_dir = os.path.join(log_dir, datasets[dataset_id])

module = __import__(datasets[dataset_id], globals(), locals(), ["*"])
for k in dir(module):
    globals()[k] = getattr(module, k)
