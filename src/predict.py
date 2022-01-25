#!/usr/bin/env python3

# VNet keras
#
# Joo-won Kim

from tensorflow import keras
import glob
import os
import sys
import data
import nibabel as nib
import numpy as np
import post_processing
from tensorflow.keras.models import model_from_json
import model


def get_data(base_dn, config, fn_img="T1w_brain.nii"):
    # predict
    lst_dn = glob.glob(os.path.join(base_dn, "*"))
    for i, dn in enumerate(lst_dn):
        if not os.path.isfile(os.path.join(dn, fn_img)):
            _ = lst_dn.pop(i)

    input_img_paths = [os.path.join(dn, fn_img) for dn in lst_dn]

    lst_transformations = [
            data.Crop_singel_image(img_size=config.img_size),
            data.normalization_single   # crop should be earlier than normalization
            ]

    predict_gen = data.LoadPredictData(config.batch_size, config.img_size, input_img_paths, lst_transformations)

    return predict_gen, input_img_paths

if __name__ == '__main__':
    keras.backend.clear_session()
    fn_config = "config.json" if len(sys.argv) < 2 else sys.argv[1]
    config = model.config_predict()
    config.load_config_json(fn_config)

    base_dn = config.dn_data
    fn_img = config.fn_t1w
    fn_seg_out = config.fn_seg
    fn_prob_out = config.fn_seg_pred
    thr_seg = 0.5

    with open(config.fn_model) as fin:
        model_json = fin.read()
    model = model_from_json(model_json)
    model.load_weights(config.fn_weight)

    predict_gen, lst_img = get_data(base_dn=base_dn, config=config, fn_img=fn_img)
    predict_preds = model.predict(predict_gen)

    output_text = "subject,volume_left,volume_right\n"
    for i in range(len(lst_img)):
        img = nib.load(lst_img[i])
        #dat_prob = data.pad_single_image(predict_preds[i,:,:,:,1]) # if softmax
        dat_prob = data.pad_single_image(predict_preds[i,:,:,:,0]) # if sigmoid

        img_prob = nib.Nifti1Image(dat_prob, img.affine, img.header)
        nib.save(img_prob, os.path.join(os.path.dirname(lst_img[i]), fn_prob_out))

        # binary segmentation
        dat_seg = (dat_prob>=thr_seg).astype("uint8")
        dat_l, dat_r = post_processing.separate_lr(dat_seg, img.affine)

        for dat_out, lr in [(dat_l, 'left'), (dat_r, 'right'), (dat_l + dat_r, 'bi')]:
            img_seg = nib.Nifti1Image(dat_out, img.affine, img.header)
            nib.save(img_seg, os.path.join(os.path.dirname(lst_img[i]), fn_seg_out.format(lr=lr)))

        dx, dy, dz = img.header.get_zooms()
        unit = dx*dy*dz

        output_text += "{subject},{volume_left},{volume_right}\n".format(
                subject=os.path.basename(os.path.dirname(lst_img[i])),
                volume_left=dat_l.sum()*unit,
                volume_right=dat_r.sum()*unit)

    print("")
    print(output_text)
    with open(config.fn_csv, 'w') as fout:
        fout.write(output_text)

