# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:58:54 2022

@author: tuan
"""
from consts import *

n_scenarios = 1
n_classes = 10
input_shape = (mnist_size, mnist_size, 1)
batch_size = 512
epochs = 50
epochs_drop = 25.
model_id = 1

alpha = 0.
beta = 500.
gamma = 0.
delta = 1.5

w_2 = 0.1
w_3 = 0.001

augmentation = False
