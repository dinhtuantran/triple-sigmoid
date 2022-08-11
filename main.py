# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:21:06 2022

@author: tuan
"""
from inits import *
from utils import *
from datasets import *
from models import *

import os
import datetime
import math
import numpy as np
import tensorflow as tf
import random
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TestCallback(Callback):
    def __init__(self, x_test_close, y_test_close, x_test_open, y_test_open,
                 cur_log_dir, alpha, beta, gamma, delta, w_1, w_2, w_3,
                 patience):
        super().__init__()
        self.x_test_close = x_test_close
        self.y_test_close = y_test_close
        self.x_test_open = x_test_open
        self.y_test_open = y_test_open
        self.cur_log_dir = cur_log_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.patience != 0:
            return
        model_1 = Model(inputs=self.model.input,
                        outputs=self.model.get_layer("Dense").output)
        out_0 = self.model.predict(self.x_test_close)
        out_1 = model_1.predict(self.x_test_close)
        points_0_close, points_1_close = [], []
        for i in range(len(self.y_test_close)):
            out_i_0 = out_0[i]
            out_i_1 = out_1[i]
            idx = np.argmax(out_i_0)

            points_0_close.append(out_i_0[idx])
            points_1_close.append(out_i_1[idx])

        out_0 = self.model.predict(self.x_test_open)
        out_1 = model_1.predict(self.x_test_open)
        points_0_open, points_1_open = [], []
        for i in range(len(self.y_test_open)):
            out_i_0 = out_0[i]
            out_i_1 = out_1[i]
            idx = np.argmax(out_i_0)

            points_0_open.append(out_i_0[idx])
            points_1_open.append(out_i_1[idx])

        range_min = int(-self.beta*2)
        range_max = int(self.beta*2)

        plt.clf()
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams.update({"font.size": font_size})
        fig, ax_0 = plt.subplots()
        h_y_0, _, _ = ax_0.hist(points_1_close, 100,
                                range=(range_min, range_max),
                                facecolor="tab:blue", alpha=0.8, zorder=1)
        h_y_1, _, _ = ax_0.hist(points_1_open, 100,
                                range=(range_min, range_max),
                                facecolor="crimson", alpha=0.8, zorder=2)
        h_y_max = max(h_y_0.max(), h_y_1.max())
        h_y_step = 1000.
        h_y_max = h_y_max / h_y_step
        if h_y_max.is_integer() is False:
            h_y_max = math.floor(h_y_max)+1
        h_y_max = h_y_max * h_y_step
        yticks = np.arange(0, h_y_max+1, h_y_step)
        ytick_labels = [str(int(ytick/h_y_step)) for ytick in yticks]
        for i in range(len(ytick_labels)):
            if ytick_labels[i] != "0":
                ytick_labels[i] += "K"
        xticks = np.arange(
            range_min, range_max+1, int(self.beta))
        plt.setp(ax_0, yticks=yticks, yticklabels=ytick_labels, xticks=xticks)
        # ax_0.set_ylabel("Input Distribution")

        x, y = plot_triple_sigmoid(self.alpha, self.beta, self.gamma,
                                   self.delta, self.w_1, self.w_2, self.w_3,
                                   range_min, range_max)

        ax_1 = ax_0.twinx()
        ax_1.plot(x[0], y[0], color="crimson", linewidth=1, zorder=3)
        ax_1.plot(x[1], y[1], color="tab:blue", linewidth=1, zorder=3)
        ax_1.plot(x[2], y[2], color="mediumvioletred", linewidth=1, zorder=3)
        plt.setp(ax_1, yticks=np.arange(0, 1.1, 0.5))
        # ax_1.set_ylabel("Triple-Sigmoid Output")

        fig.savefig(os.path.join(self.cur_log_dir,
                                 "epoch_" + str(epoch) + ".png"), dpi=1000,
                    bbox_inches="tight", transparent="True", pad_inches=0)
        plt.close(fig)


class TripleSigmoid(object):

    def set_hyperparams(self, alpha, beta, gamma, delta, w_1, w_2, w_3):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.b_0 = delta
        self.b_2 = gamma

        self.b_1 = (w_1 - w_2) * alpha
        self.b_1 = w_2 * self.b_2 + self.b_1
        self.b_1 /= w_1
        self.b_3 = beta

        self.y_beta = -w_2 * (beta - self.b_2) - self.b_0
        self.y_beta = 1. + math.exp(self.y_beta)
        temp = math.exp(-self.b_0) / (1 + math.exp(-self.b_0))
        self.y_beta = 1. / self.y_beta - temp

        if activation_id == 0:
            self.thresh = -w_1 * (alpha - self.b_1) - self.b_0
            self.thresh = 1. + math.exp(self.thresh)
            self.thresh = 1. / self.thresh
        else:
            self.thresh = thresh_default

        self.w_1_k = K.cast(w_1, dtype="float64")
        self.w_2_k = K.cast(w_2, dtype="float64")
        self.w_3_k = K.cast(w_3, dtype="float64")
        self.b_0_k = K.cast(self.b_0, dtype="float64")
        self.b_1_k = K.cast(self.b_1, dtype="float64")
        self.b_2_k = K.cast(self.b_2, dtype="float64")
        self.b_3_k = K.cast(self.b_3, dtype="float64")
        self.alpha_k = K.cast(alpha, dtype="float64")
        self.beta_k = K.cast(beta, dtype="float64")
        self.y_beta_k = K.cast(self.y_beta, dtype="float64")

    @tf.function
    def triple_sigmoid_activation(self, x):
        x = K.cast(x, dtype="float64")

        mask = K.cast(K.less(x, self.alpha_k), dtype="float64")
        x_0 = x * mask
        x_0 = 1. / \
            (1. + K.exp(
                K.minimum(-self.w_1_k * (x_0 - self.b_1_k) - self.b_0_k, 88.)))
        x_0 = x_0 * mask

        mask_l = K.cast(K.greater_equal(x, self.alpha_k), dtype="float64")
        mask_h = K.cast(K.less(x, self.beta_k), dtype="float64")
        mask = mask_l * mask_h
        x_1 = x * mask
        x_1 = 1. / \
            (1. + K.exp(
                K.minimum(-self.w_2_k * (x_1 - self.b_2_k) - self.b_0_k, 88.)))
        x_1 = x_1 * mask

        mask = K.cast(K.greater_equal(x, self.beta_k), dtype="float64")
        x_2 = x * mask
        x_2 = self.y_beta_k + \
            K.exp(K.minimum(
                -self.w_3_k * (x_2 - self.b_3_k) - self.b_0_k, 88.)) / \
            (1. + K.exp(K.minimum(
                -self.w_3_k * (x_2 - self.b_3_k) - self.b_0_k, 88.)))
        x_2 = x_2 * mask

        return x_0 + x_1 + x_2

    def step_decay_adam(self, epoch):
        initial_lrate = learning_rate_adam
        lrate = initial_lrate * \
            math.pow(drop, math.floor(epoch / epochs_drop))
        return lrate

    def prepare_data(self, rand_state):
        x_val = None
        if (dataset_id == 0
           or dataset_id == 9 or dataset_id == 10 or dataset_id == 11):
            (x_train, y_train), (x_test, y_test) = load_mnist()
        elif dataset_id == 1:
            (x_train, y_train), (x_test, y_test) = load_svhn()
        elif (dataset_id == 2 or dataset_id == 3 or dataset_id == 4
              or dataset_id == 5 or dataset_id == 6
              or dataset_id == 7 or dataset_id == 8):
            (x_train, y_train), (x_test, y_test) = load_cifar10()
        elif dataset_id == 12 or dataset_id == 13 or dataset_id == 14:
            (x_train, y_train), (x_test, y_test) = load_imagenet_as_close_set()

        if x_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=val_split, random_state=rand_seed, shuffle=True)

        random.setstate(rand_state)
        # svhn dataset has labels from 1 to 10
        if dataset_id == 1:
            classes = [i+1 for i in range(n_classes)]
        else:
            classes = [i for i in range(n_classes)]

        if dataset_id == 0 or dataset_id == 1 or dataset_id == 2:
            # mnist, svhn, cifar10
            if dataset_id == 1:
                train_classes = [i+1 for i in range(n_classes)]
            else:
                train_classes = [i for i in range(n_classes)]
            test_classes = random.sample(classes, n_tests)
            for c in test_classes:
                train_classes.remove(c)
            test_mask_open = np.isin(y_test, test_classes)
            x_test_open, y_test_open = x_test[test_mask_open], \
                y_test[test_mask_open]
            print(test_classes)

        elif dataset_id == 3 or dataset_id == 4:  # cifar100 as unknown
            train_classes = [0, 1, 8, 9]
            if dataset_id == 3:
                test_classes = random.sample(cifar100_animal_ids, 10)
            else:
                test_classes = random.sample(cifar100_animal_ids, 50)

            (x_train_100, y_train_100), (x_test_100, y_test_100) = \
                load_cifar100()
            test_mask_open = np.isin(y_test_100, test_classes)
            x_test_open, y_test_open = x_test_100[test_mask_open], \
                y_test_100[test_mask_open]

        elif (dataset_id == 5 or dataset_id == 6
              or dataset_id == 7 or dataset_id == 8):
            # imagenet or lsun as unknown
            train_classes = [i for i in range(n_classes)]
            if dataset_id == 5:
                (x_test_open, y_test_open) = load_imagenet(mode=MODE_CROP)
            elif dataset_id == 6:
                (x_test_open, y_test_open) = load_imagenet(mode=MODE_RESIZE)
            elif dataset_id == 7:
                (x_test_open, y_test_open) = load_lsun(mode=MODE_CROP)
            elif dataset_id == 8:
                (x_test_open, y_test_open) = load_lsun(mode=MODE_RESIZE)

        elif dataset_id == 9 or dataset_id == 10 or dataset_id == 11:
            # omniglot, mnist-noise, or noise as unknown
            train_classes = [i for i in range(n_classes)]
            if dataset_id == 9:
                (x_test_open, y_test_open) = load_omniglot()
            elif dataset_id == 10:
                (x_test_open, y_test_open) = load_mnist_noise()
            elif dataset_id == 11:
                (x_test_open, y_test_open) = load_noise()

        elif dataset_id == 12 or dataset_id == 13 or dataset_id == 14:
            # omniglot, mnist-noise, or noise as unknown
            train_classes = [i for i in range(n_classes)]
            if dataset_id == 12:
                (x_test_open, y_test_open) = \
                    load_mnist_as_open_set_for_imagenet()
            elif dataset_id == 13:
                (x_test_open, y_test_open) = \
                    load_svhn_as_open_set_for_imagenet()
            elif dataset_id == 14:
                (x_test_open, y_test_open) = \
                    load_cifar10_as_open_set_for_imagenet()

        num_classes = len(train_classes)
        print(train_classes)
        rand_state = random.getstate()

        train_mask = np.isin(y_train, train_classes)
        val_mask = np.isin(y_val, train_classes)
        test_mask_close = np.isin(y_test, train_classes)

        x_train, y_train = x_train[train_mask], y_train[train_mask]
        x_val, y_val = x_val[val_mask], y_val[val_mask]
        x_test_close, y_test_close = x_test[test_mask_close], \
            y_test[test_mask_close]

        # scale images to the 0~1 range
        x_train = x_train.astype("float32") / 255.
        x_val = x_val.astype("float32") / 255.
        x_test_close = x_test_close.astype("float32") / 255.
        x_test_open = x_test_open.astype("float32") / 255.

        if (dataset_id == 0 or dataset_id == 9
           or dataset_id == 10 or dataset_id == 11):
            x_train = np.expand_dims(x_train, -1)
            x_val = np.expand_dims(x_val, -1)
            x_test_close = np.expand_dims(x_test_close, -1)
            x_test_open = np.expand_dims(x_test_open, -1)

        print("x_train shape:", x_train.shape)
        print("x_val shape:", x_val.shape)
        print("x_test_close shape:", x_test_close.shape)
        print("x_test_open shape:", x_test_open.shape)

        # encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y_train)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_train = onehot_encoder.fit_transform(integer_encoded)

        integer_encoded = label_encoder.transform(y_val)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_val = onehot_encoder.transform(integer_encoded)

        integer_encoded = label_encoder.transform(y_test_close)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_test_close = onehot_encoder.transform(integer_encoded)

        return (x_train, y_train), (x_val, y_val), \
            (x_test_close, y_test_close), (x_test_open, y_test_open), \
            num_classes, rand_state

    def build_and_run_model(self,
                            num_classes,
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_test_close,
                            y_test_close,
                            x_test_open,
                            y_test_open,
                            model_id,
                            activation_id,
                            loss_id,
                            cur_log_dir,
                            is_train):
        visible = Input(shape=input_shape, name="Input")
        hidden = build_model(visible, model_id)
        hidden = Dense(num_classes, name="Dense")(hidden)

        if activation_id == 0:
            output = Activation(self.triple_sigmoid_activation,
                                name="Activation")(hidden)
        elif activation_id == 1:
            output = Activation("sigmoid", name="Activation")(hidden)
        else:
            output = Activation("softmax", name="Activation")(hidden)

        model = Model(inputs=visible, outputs=output)
        model.summary()

        optimizer = Adam(learning_rate=learning_rate_adam)

        if loss_id == 0:
            metric = "binary_accuracy"
            loss = "binary_crossentropy"
        elif loss_id == 1:
            metric = "categorical_accuracy"
            loss = "categorical_crossentropy"

        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

        weight_dir = os.path.join(cur_log_dir, "weight")
        os.makedirs(weight_dir, exist_ok=True)
        weight_path_best = os.path.join(weight_dir, weight_file_best)
        weight_path_last = os.path.join(weight_dir, weight_file_last)
        weight_path = os.path.join(weight_dir, weight_file)

        if is_train:
            tensorboard_callback = TensorBoard(cur_log_dir, histogram_freq=1)

            checkpoint_best = ModelCheckpoint(weight_path_best,
                                              monitor="val_" + metric,
                                              verbose=0,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode="auto", save_freq="epoch")

            checkpoint_last = ModelCheckpoint(weight_path_last,
                                              monitor="val_" + metric,
                                              verbose=0,
                                              save_best_only=False,
                                              save_weights_only=True,
                                              mode="auto", save_freq="epoch")

            lrate = LearningRateScheduler(self.step_decay_adam)
            # lrate = ReduceLROnPlateau(monitor="val_loss", factor=0.8,
            #                           patience=10, verbose=1, mode="auto")
            estop = EarlyStopping(monitor="val_loss", patience=1000)

            test_callback = TestCallback(x_test_close,
                                         y_test_close,
                                         x_test_open,
                                         y_test_open,
                                         cur_log_dir,
                                         self.alpha, self.beta, self.gamma,
                                         self.delta,
                                         self.w_1, self.w_2, self.w_3,
                                         patience=1)

            if augmentation:
                gen_train = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    shear_range=shear_range,
                    zoom_range=zoom_range,
                    horizontal_flip=horizontal_flip,
                    vertical_flip=False)
                gen_val = ImageDataGenerator()
                flow_train = gen_train.flow(x_train, y_train,
                                            batch_size=batch_size)
                flow_val = gen_val.flow(x_val, y_val,
                                        batch_size=batch_size)

                history = model.fit(
                    flow_train,
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=flow_val,
                    validation_steps=x_val.shape[0]//batch_size,
                    callbacks=[checkpoint_best, checkpoint_last,
                               tensorboard_callback, lrate, estop,
                               test_callback])
            else:
                history = model.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint_best, checkpoint_last,
                               tensorboard_callback, lrate, estop,
                               test_callback])

            np.save(os.path.join(cur_log_dir, history_file), history.history)

            plt.clf()
            plt.rcParams["figure.figsize"] = fig_size
            plt.rcParams.update({"font.size": font_size})
            fig = plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"], loc="upper right")
            # fig.savefig(os.path.join(cur_log_dir, "loss.png"), dpi=1000,
            #             bbox_inches="tight", transparent="True",
            #             pad_inches=0)
            plt.close(fig)

        model.load_weights(weight_path_best)

        return model

    def post_process(self,
                     model_0,
                     num_classes,
                     x_test_close,
                     y_test_close,
                     x_test_open,
                     y_test_open,
                     cur_log_dir):
        out = model_0.predict(x_test_close)
        n_correct = 0
        for i in range(len(y_test_close)):
            idx = np.argmax(out[i])
            if idx == np.argmax(y_test_close[i]):
                n_correct += 1
        acc = n_correct*100/len(y_test_close)
        print("accuracy of close set classification: ", acc)

        y_true, y_pred = [], []
        for i in range(len(y_test_close)):
            _y_true = np.argmax(y_test_close[i])
            out_i = out[i]
            idx = np.argmax(out_i)
            count = 0
            for j in range(len(out_i)):
                if out_i[j] >= self.thresh:
                    count += 1

            y_true.append(_y_true)
            if count >= 1:
                y_pred.append(idx)
            else:
                y_pred.append(num_classes)

        out = model_0.predict(x_test_open)
        for i in range(len(y_test_open)):
            out_i = out[i]
            idx = np.argmax(out_i)
            count = 0
            for j in range(len(out_i)):
                if out_i[j] >= self.thresh:
                    count += 1

            y_true.append(num_classes)
            if count == 0:
                y_pred.append(num_classes)
            else:
                y_pred.append(idx)

        report_dict = classification_report(y_true, y_pred, digits=3,
                                            output_dict=True)
        report_dict["accuracy"] = acc
        report_str = classification_report(y_true, y_pred, digits=3)
        print(report_str)

        with open(os.path.join(cur_log_dir, "report.txt"), "w") as f:
            f.write("accuracy of close set classification: %f\n" % acc)
            f.write("macro averaged f1-score: %f\n" %
                    report_dict["macro avg"]["f1-score"])
            f.write("\ndetail report:\n")
            f.write(report_str)

        model_1 = Model(inputs=model_0.input,
                        outputs=model_0.get_layer("Dense").output)
        out_0 = model_0.predict(x_test_close)
        out_1 = model_1.predict(x_test_close)
        points_0_close, points_1_close = [], []
        for i in range(len(y_test_close)):
            out_i_0 = out_0[i]
            out_i_1 = out_1[i]
            idx = np.argmax(out_i_0)

            points_0_close.append(out_i_0[idx])
            points_1_close.append(out_i_1[idx])

        out_0 = model_0.predict(x_test_open)
        out_1 = model_1.predict(x_test_open)
        points_0_open, points_1_open = [], []
        for i in range(len(y_test_open)):
            out_i_0 = out_0[i]
            out_i_1 = out_1[i]
            idx = np.argmax(out_i_0)

            points_0_open.append(out_i_0[idx])
            points_1_open.append(out_i_1[idx])

        range_min = int(-self.beta*2)
        range_max = int(self.beta*2)

        if activation_id == 2:
            with open(os.path.join(cur_log_dir, "points_close"), "wb") as fp:
                pickle.dump(points_0_close, fp)
            with open(os.path.join(cur_log_dir, "points_open"), "wb") as fp:
                pickle.dump(points_0_open, fp)
        else:
            with open(os.path.join(cur_log_dir, "points_close"), "wb") as fp:
                pickle.dump(points_1_close, fp)
            with open(os.path.join(cur_log_dir, "points_open"), "wb") as fp:
                pickle.dump(points_1_open, fp)

        plt.clf()
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams.update({"font.size": font_size})
        fig, ax_0 = plt.subplots()
        h_y_0, _, _ = ax_0.hist(points_1_close, 100,
                                range=(range_min, range_max),
                                facecolor="tab:blue", alpha=0.8, zorder=1)
        h_y_1, _, _ = ax_0.hist(points_1_open, 100,
                                range=(range_min, range_max),
                                facecolor="crimson", alpha=0.8, zorder=2)
        h_y_max = max(h_y_0.max(), h_y_1.max())
        h_y_step = 1000.
        h_y_max = h_y_max / h_y_step
        if h_y_max.is_integer() is False:
            h_y_max = math.floor(h_y_max)+1
        h_y_max = h_y_max * h_y_step
        yticks = np.arange(0, h_y_max+1, h_y_step)
        ytick_labels = [str(int(ytick/h_y_step)) for ytick in yticks]
        for i in range(len(ytick_labels)):
            if ytick_labels[i] != "0":
                ytick_labels[i] += "K"
        xticks = np.arange(
            range_min, range_max+1, int(self.beta))
        plt.setp(ax_0, yticks=yticks, yticklabels=ytick_labels, xticks=xticks)
        # ax_0.set_ylabel("Input Distribution")

        x, y = plot_triple_sigmoid(self.alpha, self.beta, self.gamma,
                                   self.delta, self.w_1, self.w_2, self.w_3,
                                   range_min, range_max)

        ax_1 = ax_0.twinx()
        ax_1.plot(x[0], y[0], color="crimson", linewidth=1, zorder=3)
        ax_1.plot(x[1], y[1], color="tab:blue", linewidth=1, zorder=3)
        ax_1.plot(x[2], y[2], color="mediumvioletred", linewidth=1, zorder=3)
        plt.setp(ax_1, yticks=np.arange(0, 1.1, 0.5))
        # ax_1.set_ylabel("Triple-Sigmoid Output")

        # fig.savefig(os.path.join(cur_log_dir, "1.png"), dpi=1000,
        #             bbox_inches="tight", transparent="True", pad_inches=0)
        plt.close(fig)

        return report_dict

    def run_a_scenario(self,
                       model_id,
                       cur_log_dir,
                       rand_state,
                       is_train):

        (x_train, y_train), (x_val, y_val), \
            (x_test_close, y_test_close), (x_test_open, y_test_open), \
            num_classes, rand_state = self.prepare_data(rand_state)

        model_0 = self.build_and_run_model(num_classes,
                                           x_train,
                                           y_train,
                                           x_val,
                                           y_val,
                                           x_test_close,
                                           y_test_close,
                                           x_test_open,
                                           y_test_open,
                                           model_id,
                                           activation_id,
                                           loss_id,
                                           cur_log_dir,
                                           is_train)

        report_dict = self.post_process(model_0,
                                        num_classes,
                                        x_test_close,
                                        y_test_close,
                                        x_test_open,
                                        y_test_open,
                                        cur_log_dir)

        return report_dict, num_classes, rand_state

    def run(self, alpha, beta, gamma, delta, w_1, w_2, w_3):
        rand_state = random.getstate()

        self.set_hyperparams(alpha, beta, gamma, delta, w_1, w_2, w_3)

        accs = []
        f1s = []
        f1s_i = []
        for scenario in range(n_scenarios):
            items = ["{:.4f}".format(w_1),
                     "{:.4f}".format(w_2),
                     "{:.4f}".format(w_3),
                     alpha,
                     beta,
                     gamma,
                     delta,
                     model_id,
                     activation_id,
                     loss_id,
                     scenario]
            items = [str(i) for i in items]
            folder_name = "_".join(items)
            if is_train:
                folder_name += "_" + \
                    datetime.datetime.now().strftime("%m%d%H%M%S")
                cur_log_dir = os.path.join(log_dir, folder_name)
                os.makedirs(cur_log_dir, exist_ok=True)
            else:
                folder_list = os.listdir(log_dir)
                for f in folder_list:
                    if folder_name in f:
                        cur_log_dir = os.path.join(log_dir, f)

            report, num_classes, rand_state = self.run_a_scenario(
                model_id,
                cur_log_dir,
                rand_state,
                is_train=is_train)

            accs.append(report["accuracy"])
            f1s.append(report["macro avg"]["f1-score"])
            f1s_i.append(report[str(num_classes)]["f1-score"])

        acc = sum(accs) / n_scenarios
        acc_std = np.std(accs)
        f1 = sum(f1s) / n_scenarios
        f1_std = np.std(f1s)
        f1_i = sum(f1s_i) / n_scenarios
        f1_i_std = np.std(f1s_i)
        print(f1s_i)

        with open(os.path.join(cur_log_dir, "report_final.txt"), "w") as f:
            f.write("accuracy of close set classification: %f\n" % acc)
            f.write("standard deviation of accuracy: %f\n" % acc_std)
            f.write("macro averaged f1-score: %f\n" % f1)
            f.write("standard deviation of f1-score: %f\n" % f1_std)
            f.write("averaged f1-score of unknown classes: %f\n" % f1_i)
            f.write(
                "standard deviation of f1-score of unknown classes: %f\n"
                % f1_i_std)


if __name__ == "__main__":
    triple_sigmoid = TripleSigmoid()
    triple_sigmoid.run(alpha, beta, gamma, delta, w_1, w_2, w_3)
