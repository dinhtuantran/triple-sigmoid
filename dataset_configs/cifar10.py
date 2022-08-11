# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:24:02 2022

@author: tuan
"""
from consts import *

n_scenarios = 5
n_classes = 10
n_tests = 4
input_shape = (cifar_size, cifar_size, 3)
batch_size = 512
epochs = 600
epochs_drop = 150.
model_id = 0

alpha = 0.
beta = 500.
gamma = 0.
delta = 3.5

w_2 = 0.1
w_3 = 0.001

augmentation = False
