# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:46:15 2022

@author: tuan
"""
import os
import sys
import datetime
import math
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# show less log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 64
random_seed = 4321
target_size = (224, 224)


def fix_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


fix_random_seed(random_seed)


def get_test_labels_df(test_labels_path):
    # read the test data labels for all files in the test set as a data frame
    test_df = pd.read_csv(test_labels_path, sep="\t",
                          index_col=None, header=None)
    test_df = test_df.iloc[:, [0, 1]].rename(
        {0: "filename", 1: "class"}, axis=1)
    return test_df


def get_train_valid_test_data_generators(batch_size, target_size):
    # define a data-augmenting image data generator
    # and a standard image data generator
    image_gen_aug = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, brightness_range=(0.5, 1.5), shear_range=5,
        zoom_range=0.2, horizontal_flip=True, fill_mode="constant", cval=127.5,
        validation_split=0.1)
    image_gen = ImageDataGenerator()

    # define a training data generator
    partial_flow_func = partial(
        image_gen_aug.flow_from_directory,
        directory=os.path.join("datasets", "tiny-imagenet-200", "train"),
        target_size=target_size, classes=None,
        class_mode="categorical", batch_size=batch_size,
        shuffle=True, seed=random_seed)

    # get the training data subset
    train_gen = partial_flow_func(subset="training")
    # get the validation data subset
    valid_gen = partial_flow_func(subset="validation")

    # define the test data generator
    test_df = get_test_labels_df(os.path.join(
        "datasets", "tiny-imagenet-200",  "val", "val_annotations.txt"))
    test_gen = image_gen.flow_from_dataframe(
        test_df, directory=os.path.join(
            "datasets", "tiny-imagenet-200",  "val", "images"),
        target_size=target_size, classes=None,
        class_mode="categorical", batch_size=batch_size, shuffle=False
    )
    return train_gen, valid_gen, test_gen


def data_gen_augmented_inception_resnet_v2(gen,
                                           random_gamma=False,
                                           random_occlude=False):
    for x, y in gen:
        if random_gamma:
            # gamma correction
            # do this in the image process fn doesn't help improve performance
            rand_gamma = np.random.uniform(0.93, 1.06, (x.shape[0], 1, 1, 1))
            x = x**rand_gamma

        if random_occlude:
            # randomly occluding sections in the image
            occ_size = 10
            occ_h = np.random.randint(0, x.shape[0]-occ_size)
            occ_w = np.random.randint(0, x.shape[0]-occ_size)
            x[::2, occ_h:occ_h+occ_size, occ_w:occ_w+occ_size,
                :] = np.random.choice([0., 128., 255.])

        # https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/applications/imagenet_utils.py#L181
        x /= 127.5
        x -= 1

        yield x, y


# get the train,valid, test data generators
train_gen, valid_gen, test_gen = get_train_valid_test_data_generators(
    batch_size, target_size)
train_gen_aux = data_gen_augmented_inception_resnet_v2(
    train_gen, random_gamma=True, random_occlude=True)
# we do not augment data in the validation/test datasets
valid_gen_aux = data_gen_augmented_inception_resnet_v2(valid_gen)
test_gen_aux = data_gen_augmented_inception_resnet_v2(test_gen)

# we"re going to download the InceptionResNetv2 model,
# remove the top layer and wrap it around an input layer
# and a prediction layer (with 200 classes)
model = Sequential([
    Input(shape=(224, 224, 3)),
    InceptionResNetV2(include_top=False, pooling="avg"),
    Dropout(0.4),
    Dense(200, activation="softmax")
])
# define the loss
loss = tf.keras.losses.CategoricalCrossentropy()
# define the optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
# compile the model
model.compile(loss=loss, optimizer=adam, metrics=["accuracy"])
model.summary()

# define callbacks
es_callback = EarlyStopping(monitor="val_loss", patience=10)
n_epochs = 50

lr_callback = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="auto"
)

checkpoint_last = ModelCheckpoint("last.h5",
                                  monitor="val_loss",
                                  verbose=0,
                                  save_best_only=False,
                                  save_weights_only=True,
                                  mode="auto", save_freq="epoch")

# train the model
history = model.fit(
    train_gen_aux,
    validation_data=valid_gen_aux,
    steps_per_epoch=int(0.9*(500*200)) // batch_size,
    validation_steps=int(0.1*(500*200)) // batch_size,
    epochs=n_epochs,
    callbacks=[es_callback, lr_callback, checkpoint_last]
)

# evaluate the model
test_res = model.evaluate(
    test_gen_aux, steps=500*50 // batch_size)
