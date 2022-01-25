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
from voxel_similarity import dice, hausdorff_distance, mean_distance

def get_data(base_dn, config):
    # test
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
            data.Crop_pad(config.img_size, padding_value=(0,0), is_training=False)
            ]

    test_gen = data.LoadData(config.batch_size, config.img_size, input_img_paths, input_lbl_paths, lst_transformations)

    return test_gen, input_img_paths, input_lbl_paths

def dice(dat1, dat2):
    return 2.0*np.logical_and(dat1, dat2).sum() / (dat1.sum() + dat2.sum())

if __name__ == '__main__':
    keras.backend.clear_session()
    fn_config = "config.json" if len(sys.argv) < 2 else sys.argv[1]
    config = model.config_predict()
    config.load_config_json(fn_config)

    base_dn = config.dn_testdata
    fn_seg_out = "hb_seg_bi.nii.gz"
    fn_prob_out = "hb_predict.nii.gz"
    thr_seg = 0.5

    with open(config.fn_model) as fin:
        model_json = fin.read()
    model = model_from_json(model_json)
    model.load_weights(config.fn_weight)

    test_gen, lst_img, lst_lbl = get_data(base_dn=base_dn, config=config)
    test_preds = model.predict(test_gen)

    print('test_preds shape:', test_preds.shape)

    output_text = "subject,volume_pred,volume_label,dice,mean_distance,hausdorff_distance\n"
    for i in range(len(lst_img)):
        print(lst_img[i])
        img = nib.load(lst_img[i])
        hdr_out = img.header.copy()
        padder = data.Crop_pad(img.shape, is_training=False)

        #dat_prob = test_preds[i,:,:,:,1]   # if softmax
        dat_prob = test_preds[i,:,:,:,0]    # if sigmoid
        dat_prob_pad = padder(dat_prob)

        hdr_out.set_data_dtype(dat_prob_pad.dtype)
        img_prob = nib.Nifti1Image(dat_prob_pad, img.affine, hdr_out)
        nib.save(img_prob, os.path.join(os.path.dirname(lst_img[i]), fn_prob_out))

        dat_seg_pad = (dat_prob_pad>=thr_seg).astype("uint8")
        dat_l, dat_r = post_processing.separate_lr(dat_seg_pad, img.affine)
        dat_seg_bi = dat_l + dat_r
        
        hdr_out.set_data_dtype(dat_l.dtype)
        img_seg = nib.Nifti1Image(dat_seg_bi, img.affine, hdr_out)
        nib.save(img_seg, os.path.join(os.path.dirname(lst_img[i]), fn_seg_out))

        dx, dy, dz = img.header.get_zooms()
        unit = dx*dy*dz

        dat_lbl = nib.load(lst_lbl[i]).get_fdata().astype('uint8')

        zooms = img.header.get_zooms()
        _, md = mean_distance(dat_seg_bi, dat_lbl, zooms)
        _, hd = hausdorff_distance(dat_seg_bi, dat_lbl, zooms)

        output_text += "{subject},{volume_pred},{volume_label},{dice},{mean_distance},{hausdorff_distance}\n".format(
                subject=os.path.basename(os.path.dirname(lst_img[i])),
                volume_pred=dat_seg_bi.sum()*unit,
                volume_label=dat_lbl.sum()*unit,
                dice=dice(dat_seg_bi, dat_lbl),
                mean_distance=md,
                hausdorff_distance=hd
                )
    print("")
    print(output_text)
    with open(config.fn_test_csv, 'w') as fout:
        fout.write(output_text)


