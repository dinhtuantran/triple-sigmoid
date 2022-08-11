# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:21:25 2022

@author: tuan
"""
from consts import *

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Activation, LeakyReLU, Lambda, ReLU
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet201


def use_multiple_batch_normalizations():
    return (dataset_id != 0 and dataset_id != 9
            and dataset_id != 10 and dataset_id != 11)


def is_input_img_with_small_size():
    return (dataset_id == 0 or dataset_id == 9
            or dataset_id == 10 or dataset_id == 11)


def build_model_0(visible):
    hidden = Conv2D(32, (3, 3),
                    padding="same", name="Conv2D-0")(visible)
    hidden = LeakyReLU(name="ReLU-0")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-0")(hidden)
    hidden = Conv2D(32, (3, 3),
                    padding="same", name="Conv2D-1")(hidden)
    hidden = LeakyReLU(name="ReLU-1")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-1")(hidden)
    hidden = MaxPooling2D((2, 2), name="MaxPooling2D-0")(hidden)
    hidden = Dropout(0.5, name="Dropout-0")(hidden)

    hidden = Conv2D(64, (3, 3),
                    padding="same", name="Conv2D-2")(hidden)
    hidden = LeakyReLU(name="ReLU-2")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-2")(hidden)
    hidden = Conv2D(64, (3, 3),
                    padding="same", name="Conv2D-3")(hidden)
    hidden = LeakyReLU(name="ReLU-3")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-3")(hidden)
    hidden = MaxPooling2D((2, 2), name="MaxPooling2D-1")(hidden)
    hidden = Dropout(0.5, name="Dropout-1")(hidden)

    hidden = Conv2D(128, (3, 3),
                    padding="same", name="Conv2D-4")(hidden)
    hidden = LeakyReLU(name="ReLU-4")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-4")(hidden)
    hidden = Conv2D(128, (3, 3),
                    padding="same", name="Conv2D-5")(hidden)
    hidden = LeakyReLU(name="ReLU-5")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-5")(hidden)
    hidden = MaxPooling2D((2, 2), name="MaxPooling2D-2")(hidden)
    hidden = Dropout(0.5, name="Dropout-2")(hidden)

    hidden = Flatten(name="Flatten")(hidden)
    hidden = Dense(128, name="Dense-0")(hidden)
    hidden = LeakyReLU(name="ReLU-6")(hidden)
    hidden = BatchNormalization(name="BatchNormalization-6")(hidden)
    hidden = Dropout(0.5, name="Dropout-3")(hidden)

    return hidden


def build_model_1(visible):
    # VGG13
    hidden = Conv2D(64, (3, 3),
                    padding="same",
                    name="Conv2D-0")(visible)
    hidden = LeakyReLU(name="ReLU-0")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-0")(hidden)
    hidden = Dropout(0.3, name="Dropout-0")(hidden)

    hidden = Conv2D(64, (3, 3),
                    padding="same",
                    name="Conv2D-1")(hidden)
    hidden = LeakyReLU(name="ReLU-1")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-1")(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2),
                          name="MaxPooling2D-0")(hidden)

    hidden = Conv2D(128, (3, 3),
                    padding="same",
                    name="Conv2D-2")(hidden)
    hidden = LeakyReLU(name="ReLU-2")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-2")(hidden)
    hidden = Dropout(0.4, name="Dropout-1")(hidden)

    hidden = Conv2D(128, (3, 3),
                    padding="same",
                    name="Conv2D-3")(hidden)
    hidden = LeakyReLU(name="ReLU-3")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-3")(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2),
                          name="MaxPooling2D-1")(hidden)

    hidden = Conv2D(256, (3, 3),
                    padding="same",
                    name="Conv2D-4")(hidden)
    hidden = LeakyReLU(name="ReLU-4")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-4")(hidden)
    hidden = Dropout(0.4, name="Dropout-2")(hidden)

    hidden = Conv2D(256, (3, 3),
                    padding="same",
                    name="Conv2D-5")(hidden)
    hidden = LeakyReLU(name="ReLU-5")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-5")(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2),
                          name="MaxPooling2D-2")(hidden)

    if not is_input_img_with_small_size():
        hidden = Conv2D(512, (3, 3),
                        padding="same",
                        name="Conv2D-6")(hidden)
        hidden = LeakyReLU(name="ReLU-6")(hidden)
        if use_multiple_batch_normalizations():
            hidden = BatchNormalization(name="BatchNormalization-6")(hidden)
        hidden = Dropout(0.4, name="Dropout-3")(hidden)

        hidden = Conv2D(512, (3, 3),
                        padding="same",
                        name="Conv2D-7")(hidden)
        hidden = LeakyReLU(name="ReLU-7")(hidden)
        if use_multiple_batch_normalizations():
            hidden = BatchNormalization(name="BatchNormalization-7")(hidden)
        hidden = MaxPooling2D(pool_size=(2, 2),
                              name="MaxPooling2D-3")(hidden)

    hidden = Conv2D(512, (3, 3),
                    padding="same",
                    name="Conv2D-8")(hidden)
    hidden = LeakyReLU(name="ReLU-8")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-8")(hidden)
    hidden = Dropout(0.4, name="Dropout-4")(hidden)

    hidden = Conv2D(512, (3, 3),
                    padding="same",
                    name="Conv2D-9")(hidden)
    hidden = LeakyReLU(name="ReLU-9")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-9")(hidden)
    hidden = MaxPooling2D(pool_size=(2, 2),
                          name="MaxPooling2D-4")(hidden)
    hidden = Dropout(0.5, name="Dropout-5")(hidden)

    hidden = Flatten(name="Flatten")(hidden)
    hidden = Dense(512, name="Dense-0")(hidden)
    hidden = LeakyReLU(name="ReLU-10")(hidden)
    if use_multiple_batch_normalizations():
        hidden = BatchNormalization(name="BatchNormalization-10")(hidden)

    hidden = Dense(512, name="Dense-1")(hidden)
    hidden = LeakyReLU(name="ReLU-11")(hidden)
    hidden = BatchNormalization(name="BatchNormalization-11")(hidden)
    hidden = Dropout(0.5, name="Dropout-6")(hidden)

    return hidden


def build_model_2(visible):
    hidden = InceptionResNetV2(
        include_top=False, pooling="avg", weights=None)(visible)
    hidden = DenseNet201(
        include_top=False, pooling="avg", weights=None)(visible)
    hidden = Dropout(0.5, name="Dropout")(hidden)

    return hidden


def build_model(visible, model_id):
    if model_id == 0:
        return build_model_0(visible)
    elif model_id == 1:
        return build_model_1(visible)
    elif model_id == 2:
        return build_model_2(visible)
