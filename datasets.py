# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:58:18 2022

@author: tuan
"""
from inits import *
from consts import *

import os
import numpy as np
import pandas as pd
import lmdb
import shutil
import math
import cv2
from urllib.request import urlretrieve, urlopen
from io import BytesIO
from zipfile import ZipFile
from scipy.io import loadmat
from skimage.transform import resize
from tensorflow.keras.datasets import mnist, cifar10, cifar100

datasets_folder = "datasets"

svhn_download_urls = ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                      "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                      "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"]
svhn_folder = "svhn"
svhn_dir = os.path.join(datasets_folder, svhn_folder)

imagenet_download_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
imagenet_folder = "tiny-imagenet-200"
imagenet_dir = os.path.join(datasets_folder, imagenet_folder)
imagenet_train_dir = os.path.join(imagenet_dir, "train")
imagenet_val_dir = os.path.join(imagenet_dir, "val")
imagenet_test_dir = os.path.join(imagenet_dir, "test")

lsun_download_url = "http://dl.yf.io/lsun/scenes/test_lmdb.zip"
lsun_folder = "lsun"
lsun_dir = os.path.join(datasets_folder, lsun_folder)

omniglot_download_url = "https://github.com/brendenlake/omniglot/raw/" + \
    "master/python/images_evaluation.zip"
omniglot_folder = "omniglot"
omniglot_dir = os.path.join(datasets_folder, omniglot_folder)

noise_folder = "noise"
mnist_noise_folder = "mnist-noise"
noise_dir = os.path.join(datasets_folder, noise_folder)
mnist_noise_dir = os.path.join(datasets_folder, mnist_noise_folder)

MODE_CROP = 0
MODE_RESIZE = 1


def load_mnist():
    return mnist.load_data()


def load_svhn():
    if os.path.isdir(svhn_dir) is False:
        print("downloading svhn dataset")
        os.makedirs(svhn_dir, exist_ok=True)
        for svhn_download_url in svhn_download_urls:
            urlretrieve(svhn_download_url,
                        os.path.join(
                            svhn_dir, svhn_download_url.split("/")[-1]))
        print("download finished")

    print("reading svhn test images")
    train_raw = loadmat(
        os.path.join("datasets", "svhn", "train_32x32.mat"))
    test_raw = loadmat(
        os.path.join("datasets", "svhn", "test_32x32.mat"))
    x_train = np.array(train_raw["X"])
    x_test = np.array(test_raw["X"])
    y_train = train_raw["y"]
    y_test = test_raw["y"]
    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    y_train = np.squeeze(y_train, axis=1)
    y_test = np.squeeze(y_test, axis=1)
    print("reading finished")

    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np.squeeze(y_train, axis=1)
    y_test = np.squeeze(y_test, axis=1)

    return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(
        label_mode="fine")
    y_test = np.squeeze(y_test, axis=1)

    return (x_train, y_train), (x_test, y_test)


def crop(img, input_h, input_w, output_size):
    top = np.random.randint(0, input_h - output_size)
    left = np.random.randint(0, input_w - output_size)
    bottom = top + output_size
    right = left + output_size
    img = img[top:bottom, left:right, :]

    return img


def load_imagenet(mode=MODE_CROP):
    if os.path.isdir(imagenet_dir) is False:
        print("downloading imagenet dataset")
        os.makedirs(imagenet_dir, exist_ok=True)
        http_response = urlopen(imagenet_download_url)
        zip_file = ZipFile(BytesIO(http_response.read()))
        print("extracting zip file")
        zip_file.extractall(datasets_folder)
        print("download finished")

    print("reading imagenet test images")
    x, y = [], []
    for root, dirs, files in os.walk(imagenet_test_dir):
        for file in files:
            _x = cv2.imread(os.path.join(root, file))
            _x = cv2.cvtColor(_x, cv2.COLOR_BGR2RGB)
            if mode is MODE_CROP:
                _x = crop(_x, _x.shape[0], _x.shape[1], cifar_size)
            elif mode is MODE_RESIZE:
                _x = cv2.resize(_x, dsize=(cifar_size, cifar_size),
                                interpolation=cv2.INTER_NEAREST)
            x.append(_x)
            y.append(0)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)


def load_lsun(mode=MODE_CROP):
    if os.path.isdir(lsun_dir) is False:
        print("downloading lsun dataset")
        os.makedirs(lsun_dir, exist_ok=True)
        http_response = urlopen(lsun_download_url)
        zip_file = ZipFile(BytesIO(http_response.read()))
        print("extracting zip file")
        zip_file.extractall(lsun_dir)
        print("extracting lmdb file")
        extracted_folder_dir = os.path.join(lsun_dir, "test_lmdb")
        env = lmdb.open(extracted_folder_dir, map_size=1099511627776,
                        max_readers=100, readonly=True)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                img_dir = os.path.join(lsun_dir,
                                       key.decode("ascii") + ".webp")
                with open(img_dir, "wb") as fp:
                    fp.write(val)
            cursor.close()
        env.close()
        shutil.rmtree(extracted_folder_dir)
        print("download finished")

    print("reading lsun images")
    x, y = [], []
    for root, dirs, files in os.walk(lsun_dir):
        for file in files:
            _x = cv2.imread(os.path.join(root, file))
            _x = cv2.cvtColor(_x, cv2.COLOR_BGR2RGB)
            if mode is MODE_CROP:
                _x = crop(_x, _x.shape[0], _x.shape[1], cifar_size)
            elif mode is MODE_RESIZE:
                _x = cv2.resize(_x, dsize=(cifar_size, cifar_size),
                                interpolation=cv2.INTER_NEAREST)
            x.append(_x)
            y.append(0)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)


def load_omniglot():
    if os.path.isdir(omniglot_dir) is False:
        print("downloading omniglot dataset")
        os.makedirs(omniglot_dir, exist_ok=True)
        http_response = urlopen(omniglot_download_url)
        zip_file = ZipFile(BytesIO(http_response.read()))
        print("extracting zip file")
        zip_file.extractall(omniglot_dir)
        print("download finished")

    print("reading omniglot test images")
    x, y = [], []
    for root, dirs, files in os.walk(omniglot_dir):
        for file in files:
            _x = cv2.imread(os.path.join(root, file))
            _x = cv2.cvtColor(_x, cv2.COLOR_BGR2GRAY)
            _x = cv2.resize(_x, dsize=(mnist_size, mnist_size),
                            interpolation=cv2.INTER_NEAREST)
            x.append(_x)
            y.append(0)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    idxs = np.random.choice(len(x), size=num_test_samples, replace=False)
    x = x[idxs]
    y = y[idxs]
    return (x, y)


def generate_mnist_noise_dataset():
    if os.path.isdir(noise_dir) is True:
        return
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(mnist_noise_dir, exist_ok=True)

    print("generating noise and mnist-noise datasets")
    imgs = []
    for i in range(num_test_samples):
        img = 255*np.random.uniform(low=0.0, high=1.0,
                                    size=(mnist_size, mnist_size))
        cv2.imwrite(os.path.join(noise_dir, str(i) + ".jpg"), img)
        imgs.append(img)

    _, (x_test, _) = load_mnist()

    for i in range(num_test_samples):
        img = imgs[i] + x_test[i]
        cv2.imwrite(os.path.join(mnist_noise_dir, str(i) + ".jpg"), img)
    print("generating finished")


def _load_noise(dataset_dir, dataset_name):
    generate_mnist_noise_dataset()

    print("reading " + dataset_name + " test images")
    x, y = [], []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            _x = cv2.imread(os.path.join(root, file))
            _x = cv2.cvtColor(_x, cv2.COLOR_BGR2GRAY)

            x.append(_x)
            y.append(0)

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)


def load_mnist_noise():
    return _load_noise(mnist_noise_dir, "mnist-noise")


def load_noise():
    return _load_noise(noise_dir, "noise")


def load_imagenet_as_close_set():
    if os.path.isdir(imagenet_dir) is False:
        print("downloading imagenet dataset")
        os.makedirs(imagenet_dir, exist_ok=True)
        http_response = urlopen(imagenet_download_url)
        zip_file = ZipFile(BytesIO(http_response.read()))
        print("extracting zip file")
        zip_file.extractall(datasets_folder)
        print("download finished")

    print("reading imagenet train images")
    imagenet_val_img_dir = os.path.join(imagenet_val_dir, "images")

    with open(os.path.join(imagenet_dir, "wnids.txt")) as f:
        lines = f.readlines()
        classes = [line[:-1] for line in lines]

    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(classes)):
        class_dir = os.path.join(imagenet_train_dir, classes[i], "images")
        imgs = os.listdir(class_dir)

        for j in range(len(imgs)):
            _x = cv2.imread(os.path.join(class_dir, imgs[j]))
            _x = cv2.cvtColor(_x, cv2.COLOR_BGR2RGB)
            x_train.append(_x)
            y_train.append(i)

    val_annotations = pd.read_csv(
        os.path.join(imagenet_val_dir, "val_annotations.txt"),
        sep="\t", header=None)

    for val_annotation in val_annotations.iterrows():
        file_name = val_annotation[1][0]
        class_name = val_annotation[1][1]
        class_idx = classes.index(class_name)

        _x = cv2.imread(os.path.join(imagenet_val_img_dir, file_name))
        _x = cv2.cvtColor(_x, cv2.COLOR_BGR2RGB)
        x_test.append(_x)
        y_test.append(class_idx)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print("reading finished")

    return (x_train, y_train), (x_test, y_test)


def load_mnist_as_open_set_for_imagenet():
    print("reading mnist test images")
    _, (x_test, _) = mnist.load_data()
    x, y = [], []
    for i in range(len(x_test)):
        _x = cv2.resize(x_test[i], dsize=(imagenet_size, imagenet_size),
                        interpolation=cv2.INTER_LINEAR)
        _x = np.repeat(_x.reshape(imagenet_size, imagenet_size, 1),
                       repeats=3, axis=-1)
        x.append(_x)
        y.append(0)
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)


def load_svhn_as_open_set_for_imagenet():
    if os.path.isdir(svhn_dir) is False:
        print("downloading svhn dataset")
        os.makedirs(svhn_dir, exist_ok=True)
        for svhn_download_url in svhn_download_urls:
            urlretrieve(svhn_download_url,
                        os.path.join(
                            svhn_dir, svhn_download_url.split("/")[-1]))
        print("download finished")

    print("reading svhn test images")
    test_raw = loadmat(
        os.path.join("datasets", "svhn", "test_32x32.mat"))
    x_test = np.array(test_raw["X"])
    x_test = np.moveaxis(x_test, -1, 0)

    x, y = [], []
    for i in range(len(x_test)):
        _x = cv2.resize(x_test[i], dsize=(imagenet_size, imagenet_size),
                        interpolation=cv2.INTER_LINEAR)
        x.append(_x)
        y.append(0)
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)


def load_cifar10_as_open_set_for_imagenet():
    print("reading cifar10 test images")
    _, (x_test, _) = cifar10.load_data()
    x, y = [], []
    for i in range(len(x_test)):
        _x = cv2.resize(x_test[i], dsize=(imagenet_size, imagenet_size),
                        interpolation=cv2.INTER_LINEAR)
        x.append(_x)
        y.append(0)
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print("reading finished")

    return (x, y)
