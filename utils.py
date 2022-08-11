# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:54:37 2022

@author: tuan
"""
import os
import math
import yaml
import numpy as np


def plot_triple_sigmoid(alpha, beta, gamma, delta,
                        w_1, w_2, w_3, x_from, x_to):
    b_0 = delta
    b_2 = gamma
    b_3 = beta

    b_1 = (w_2 * b_2 + (w_1 - w_2) * alpha) / w_1

    temp = math.exp(-b_0) / (1 + math.exp(-b_0))
    y_beta = 1. / (1. + math.exp(-w_2 * (beta - b_2))) - temp

    x, y = [[], [], []], [[], [], []]
    _x = np.arange(x_from, alpha, 0.1)
    for item in _x:
        x[0].append(item)
        y[0].append(1. / (1. + math.exp(-w_1 * (item - b_1) - b_0)))

    _x = np.arange(alpha, beta, 0.1)
    for item in _x:
        x[1].append(item)
        y[1].append(1. / (1. + math.exp(-w_2 * (item - b_2) - b_0)))

    _x = np.arange(beta, x_to, 0.1)
    for item in _x:
        x[2].append(item)
        _y = 1. + math.exp(-w_3 * (item - b_3) - b_0)
        _y = y_beta + math.exp(-w_3 * (item - b_3) - b_0) / _y
        y[2].append(_y)

    return x, y


def check_dataset_id():
    if os.path.exists("consts.yaml") is False:
        with open("consts.yaml", "w") as file:
            doc = yaml.dump({"dataset_id": 0, "w_1": 0.005}, file)

    with open("consts.yaml") as file:
        doc = yaml.full_load(file)
        return doc["dataset_id"]


def check_w_1():
    if os.path.exists("consts.yaml") is False:
        with open("consts.yaml", "w") as file:
            doc = yaml.dump({"dataset_id": 0, "w_1": 0.005}, file)

    with open("consts.yaml") as file:
        doc = yaml.full_load(file)
        return doc["w_1"]
