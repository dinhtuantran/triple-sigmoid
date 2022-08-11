# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:42:21 2022

@author: tuan
"""
from utils import *

is_train = True
datasets = {0: "mnist",
            1: "svhn",
            2: "cifar10",
            3: "cifar+10",
            4: "cifar+50",

            5: "imagenet_crop",
            6: "imagenet_resize",
            7: "lsun_crop",
            8: "lsun_resize",

            9: "omniglot",
            10: "mnist_noise",
            11: "noise",

            12: "imagenet_mnist",
            13: "imagenet_svhn",
            14: "imagenet_cifar10"}
dataset_id = check_dataset_id()
w_1 = check_w_1()

rand_seed = 10

cifar100_animal_ids = [1, 2, 3, 4, 6, 7, 11, 14, 15, 18, 19, 21, 24, 26, 27,
                       29, 30, 31, 32, 34, 35, 36, 38, 42, 43, 44, 45, 46, 50,
                       55, 63, 64, 65, 66, 67, 72, 73, 74, 75, 77, 78, 79, 80,
                       88, 91, 93, 95, 97, 98, 99]

mnist_size = 28
cifar_size = 32
imagenet_size = 64
num_test_samples = 10000

weight_file_best = "best.h5"
weight_file_last = "last.h5"
weight_file = "{epoch:04d}.h5"
history_file = "history.npy"

drop = 0.5
val_split = 0.1
learning_rate_adam = 0.001

activation_id = 0
loss_id = 0

# thresh for traditional methods using softmax or sigmoid
thresh_default = 0.5

fig_size = (2.1, 1.4)
font_size = 7
