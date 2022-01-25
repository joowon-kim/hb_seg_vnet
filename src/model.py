#!/usr/bin/env python3

# VNet keras
#
# Joo-won Kim

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, concatenate, UpSampling3D, Dropout
from keras.layers.advanced_activations import PReLU
import json

class config_train(object):
    def __init__(
            self,
            batch_size=5,
            img_size=(64, 64, 64),
            fn_model="saved_model/hb_seg_model.json",
            fn_weight="saved_model/hb_seg_weight",
            fn_check="saved_model/weights.{epoch:02d}-{val_loss:.4f}",
            dn_testdata=".",
            logdir="saved_model/log"
            ):
        self.batch_size = batch_size
        self.img_size = tuple(img_size)
        self.fn_model = fn_model
        self.fn_weight = fn_weight
        self.dn_testdata = dn_testdata
        self.logdir = logdir

    def load_config_json(self, fn):
        c = json.load(open(fn))
        self.batch_size = c["train"]["batch_size"]
        self.img_size = tuple(c["train"]["img_size"])
        self.fn_model = c["train"]["fn_model"]
        self.fn_weight = c["train"]["fn_weight"]
        self.fn_check = c["train"]["fn_check"]
        self.logdir = c["train"]["logdir"]
        if "dn_testdata" in c["train"]:
            self.dn_testdata = c["train"]["dn_testdata"]
        if "fn_testcsv" in c["train"]:
            self.fn_test_csv = c["train"]["fn_testcsv"]
        else:
            self.fn_test_csv = self.fn_csv

class config_predict(object):
    def __init__(
            self,
            batch_size=5,
            img_size=(64, 64, 64),
            fn_model="saved_model/hb_seg_model.json",
            fn_weight="saved_model/hb_seg_weight",
            dn_data=".",
            dn_testdata=".",
            fn_t1w="T1w_brain.nii",
            fn_seg="hb_seg_{lr}.nii.gz",
            fn_seg_pred="hb_predict.nii.gz",
            fn_csv="hb_seg_predict.csv",
            fn_testcsv="hb_seg_test.csv"
            ):
        self.batch_size = batch_size
        self.img_size = tuple(img_size)
        self.fn_model = fn_model
        self.fn_weight = fn_weight
        self.dn_data = dn_data
        self.dn_testdata = dn_testdata
        self.fn_t1w = fn_t1w
        self.fn_seg = fn_seg
        self.fn_seg_pred = fn_seg_pred
        self.fn_csv = fn_csv
        self.fn_testcsv = fn_testcsv

    def load_config_json(self, fn):
        c = json.load(open(fn))
        self.batch_size = c["predict"]["batch_size"]
        self.img_size = tuple(c["train"]["img_size"])
        self.fn_model = c["train"]["fn_model"]
        self.fn_weight = c["predict"]["fn_weight"]
        self.fn_csv = c["predict"]["fn_csv"]
        self.fn_t1w = c["predict"]["fn_t1w"]
        self.fn_seg = c["predict"]["fn_seg"]
        self.fn_seg_pred = c["predict"]["fn_seg_pred"]
        if "dn_data" in c["predict"]:
            self.dn_data = c["predict"]["dn_data"]
        if "dn_testdata" in c["train"]:
            self.dn_testdata = c["train"]["dn_testdata"]
        if "fn_testcsv" in c["train"]:
            self.fn_test_csv = c["train"]["fn_testcsv"]
        else:
            self.fn_test_csv = self.fn_csv


from tensorflow.math import reduce_sum, reduce_mean, reduce_min
from tensorflow.math import sqrt as tf_sqrt
def jaccard_distance_loss(y_true, y_pred, smooth=0.01, axis=[1, 2, 3]): #, smooth=1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    y_pred_mod = y_pred
    y_true_mod = tf.cast(y_true, tf.float32)

    intersection = reduce_sum(y_true_mod * y_pred_mod, axis=axis)
    sum_ = reduce_sum(y_true_mod + y_pred_mod, axis=axis)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - reduce_mean(jac)

def dice_loss(y_true, y_pred, smooth=0.01, axis=[1, 2, 3]):
    """
    Dice = 2 * (|X and Y|) / (|X| + |Y|)
    Only works for 3D image. [batch, x, y, z, 1]
    """
    y_pred_mod = y_pred
    y_true_mod = tf.cast(y_true, tf.float32)

    dice = (reduce_sum(y_true_mod * y_pred_mod, axis=axis) * 2.0 + smooth) / (reduce_sum(y_true_mod, axis=axis) + reduce_sum(y_pred_mod, axis=axis) + smooth)
    return 1.0 - reduce_mean(dice)


def get_model(
        img_size,
        num_classes=2,
        init_filters=32,
        kernel_size=3,
        lst_num_convolutions=(1, 2, 3),
        bottom_convolutions=3,
        activation_fn=PReLU,
        dropout_rate=0.01
        ):
    inputs = keras.Input(shape=img_size + (1,))
    filters = init_filters
    num_levels = len(lst_num_convolutions)

    x = inputs

    lst_levels = []
    for level in range(num_levels):
        for i in range(lst_num_convolutions[level]):
            x = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = activation_fn()(x)
            x = Dropout(dropout_rate)(x)
        lst_levels.append(x)
        x = Conv3D(1, kernel_size=1, strides=2, padding='same')(x)
        filters *= 2

    for i in range(bottom_convolutions):
        x = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = activation_fn()(x)
        x = Dropout(dropout_rate)(x)

    for level in reversed(range(num_levels)):
        filters /= 2
        prev_level = lst_levels[level]
        output_padding = -2*x.shape[1] + prev_level.shape[1]
        x = Conv3DTranspose(prev_level.shape[-1], kernel_size=2, strides=2, padding='valid', output_padding=output_padding)(x)
        x = concatenate((x, prev_level), axis=-1)
        for i in range(lst_num_convolutions[level]):
            x = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = activation_fn()(x)
            x = Dropout(dropout_rate)(x)

    #outputs = Conv3D(num_classes, kernel_size=1, activation="softmax")(x)
    outputs = Conv3D(1, kernel_size=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    keras.backend.clear_session()

    img_size = (64, 64, 64)
    num_classes = 2
    # Build model
    model = get_model(img_size, num_classes, dropout_rate=0.5)
    model.summary()

