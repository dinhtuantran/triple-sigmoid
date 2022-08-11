# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:07:54 2022

@author: tuan
"""
from consts import *

import argparse
import yaml

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Triple-Sigmoid Activation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset_id",
                        default=0,
                        help="dataset id " + str(datasets))
    parser.add_argument("-w", "--w_1",
                        default=0.005,
                        help="w_1")
    args = parser.parse_args()
    config = vars(args)
    dataset_id = int(config["dataset_id"])
    w_1 = float(config["w_1"])

    with open("consts.yaml", "w") as file:
        doc = yaml.dump({"dataset_id": dataset_id, "w_1": w_1}, file)
