# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:54:00 2022

@author: tuan
"""
from consts import *

n_scenarios = 1
n_classes = 200
input_shape = (imagenet_size, imagenet_size, 3)
batch_size = 64
epochs = 600
epochs_drop = 150.
model_id = 2

alpha = 0.
beta = 500.
gamma = 0.
delta = 3.5

w_2 = 0.1
w_3 = 0.001

augmentation = True
rotation_range = 30
width_shift_range = 0.2
height_shift_range = 0.2
brightness_range = (0.5, 1.5)
shear_range = 5
zoom_range = 0.2
horizontal_flip = True
fill_mode = "constant"
cval = 127.5
validation_split = 0.1
