#!/usr/bin/env python3

# VNet keras
#
# Joo-won Kim

import random
from tensorflow import keras
import tensorflow as tf
import model
import data
import glob
import os
import sys
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from model import jaccard_distance_loss, dice_loss


def get_data(config):
    dn_data = os.environ["DATA"]

    # training
    base_dn = os.path.join(dn_data, "train")
    fn_img = "T1w_brain_reduced.nii"
    fn_lbl = "seg_prob_reduced_0.3.nii"

    lst_dn = glob.glob(os.path.join(base_dn, "*"))
    for i, dn in enumerate(lst_dn):
        if not os.path.isfile(os.path.join(dn, fn_img)) or not os.path.isfile(os.path.join(dn, fn_lbl)):
            _ = lst_dn.pop(i)

    input_img_paths = [os.path.join(dn, fn_img) for dn in lst_dn]
    input_lbl_paths = [os.path.join(dn, fn_lbl) for dn in lst_dn]

    lst_transformations = [
            data.Add_noise(noise_level=0.05),
            data.normalization,
            data.Crop_pad(config.img_size, padding_value=(0,0), is_training=True),
            data.random_flip_x
            ]

    train_gen = data.LoadData(config.batch_size, config.img_size, input_img_paths, input_lbl_paths, lst_transformations)

    # validating
    base_dn = os.path.join(dn_data, "validation")
    fn_img = "T1w_brain_reduced.nii"
    fn_lbl = "seg_prob_reduced_0.3.nii"

    lst_dn = glob.glob(os.path.join(base_dn, "*"))
    for i, dn in enumerate(lst_dn):
        if not os.path.isfile(os.path.join(dn, fn_img)) or not os.path.isfile(os.path.join(dn, fn_lbl)):
            _ = lst_dn.pop(i)

    input_img_paths = [os.path.join(dn, fn_img) for dn in lst_dn]
    input_lbl_paths = [os.path.join(dn, fn_lbl) for dn in lst_dn]

    lst_transformations = [
            data.normalization,
            data.Crop_pad(config.img_size, padding_value=(0,0), is_training=False),
            data.random_flip_x
            ]

    val_gen = data.LoadData(config.batch_size, config.img_size, input_img_paths, input_lbl_paths, lst_transformations)

    return train_gen, val_gen


if __name__ == "__main__":
    keras.backend.clear_session()

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("cannot set memory growth")

    fn_config = "config.json"
    config = model.config_train()
    config.load_config_json(fn_config)

    train_gen, val_gen = get_data(config)

    model = model.get_model(config.img_size, num_classes=2, dropout_rate=0.5)
    model.summary()

    # save model
    if not os.path.isdir(os.path.dirname(config.fn_model)):
        os.makedirs(os.path.dirname(config.fn_model))
    with open(config.fn_model, 'w') as fout:
        fout.write(model.to_json())

    #model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy")
    #model.compile(optimizer=Adam(learning_rate=0.01), loss=jaccard_distance_loss)
    #model.compile(optimizer="Adam", loss=dice_loss)
    #model.compile(optimizer=Adam(learning_rate=0.01, epsilon=0.0001), loss=dice_loss)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=dice_loss)
    #model.compile(optimizer=Adam(learning_rate=0.001), loss=jaccard_distance_loss)
    #model.compile(optimizer="Adam", loss="binary_crossentropy")
    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                config.fn_check,
                save_weights_only=True,
                save_best_only=False,
                verbose=1
                ),
            tf.keras.callbacks.TensorBoard(log_dir=config.logdir)
            ]

    epochs = 50
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    model.save_weights(config.fn_weight)


