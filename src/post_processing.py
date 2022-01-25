#!/usr/bin/env python3

# VNet keras
#
# Joo-won Kim

from skimage import measure
import scipy.ndimage
import os
import sys
import nibabel as nib
import numpy as np

def basename_nifti(fn):
    """return basename (without extenstion) of a nifti image"""
    fn_out = fn[:-3] if fn[-3:] == '.gz' else fn
    fn_out = fn_out[:-4] if fn_out[-4:] == '.nii' else fn_out
    return fn_out

def separate_lr(dat_bin, aff, verbose=True):
    """Separate left and right habenula from segmented image and exclude mis-segmented region outside habenula.
    separate_lr(
        dat_bin: binary segmentation
        aff: affine matrix
        verbose=True
    )
    return: left_habenula, right_habenula
    """
    labels = measure.label(dat_bin, connectivity=3)

    if labels.max() == 1:
        sys.stderr.write("only one connected region found\n")
        return dat_bin, dat_bin
    elif labels.max() > 2:
        if verbose:
            txt = "# There are {num_labels} labels\n".format(num_labels=labels.max())

        # find two largest volume labels
        vol_labels = [(labels==value).sum() for value in range(1, labels.max() + 1)]
        sorted_arg = np.argsort(vol_labels)
        labels_mod = np.zeros(labels.shape, dtype=labels.dtype)
        labels_mod[labels==(sorted_arg[-1]+1)] = 1
        labels_mod[labels==(sorted_arg[-2]+1)] = 2

        # if other label is connected to the left or right habenula, include the label
        for value in sorted_arg[:-2]:
            label_dil = scipy.ndimage.binary_dilation(labels==(value + 1))
            for voxel in zip(*label_dil.nonzero()):
                if labels_mod[voxel] == 1:
                    labels_mod[labels==(value + 1)] = 1
                    if verbose:
                        txt += " a label has been assigned to the habenula\n"
                    break
                if labels_mod[voxel] == 2:
                    labels_mod[labels==(value + 1)] = 2
                    if verbose:
                        txt += " a label has been assigned to the habenula\n"
                    break
            else:
                if verbose:
                    txt += " a label has been excluded\n"
        labels = labels_mod
        if verbose:
            print(txt)

    # define which label is left/right habenula
    if aff[0][0] < 0 and scipy.ndimage.center_of_mass(labels==1) < scipy.ndimage.center_of_mass(labels==2):
        left, right = 2, 1
    else:
        left, right = 1, 2

    dat_left = dat_bin.copy()
    dat_left[labels!=left] = 0
    dat_right = dat_bin.copy()
    dat_right[labels!=right] = 0

    return dat_left, dat_right


def post_processing_hb(fn, thr=0.5, bn_out=None, verbose=False):
    """Post processing on hb-vnet prediction probability
    1. set threshold value to make binary segmentation
    2. label connected segmented voxels
    3. define left / right habenula and mis-segmentation if there is any
    4. save segmented habenula

    Usage: post_processing_hb(fn, thr=0.5, bn_out=None)
        fn: hb-vnet probability output
        thr: threshold value to segment the habenula. (Probability >= thr)
        bn_out: basename of output. [{bn_out}.nii.gz, {bn_out}_left.nii.gz, {bn_out}_right.nii.gz].
            if bn_out is None, {basename of fn}_seg will be the {bn_out}.
"""
    
    if bn_out is None:
        bn_out = basename_nifti(fn) + '_seg'
    img = nib.load(fn)
    dat = (img.get_fdata() >= thr).astype("uint8")
    hdr_out = img.header.copy()
    hdr_out.set_data_dtype(dat.dtype)

    dat_left, dat_right = separate_lr(dat, img.affine, verbose=verbose)

    for dat_out, lr in [(dat_left, 'left'), (dat_right, 'right')]:
        img_out = nib.Nifti1Image(dat_out, img.affine, hdr_out)
        nib.save(img_out, bn_out + '_' + lr + '.nii.gz')
    img_out = nib.Nifti1Image(dat_left + dat_right, img.affine, hdr_out)
    nib.save(img_out, bn_out + '.nii.gz')


if __name__ == '__main__':
    verbose = True

    USAGE="{name} fn_probability_seg [threshold_value=0.5 [bn_output]]".format(name=sys.argv[0])
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        sys.stderr.write(USAGE)
        sys.exit(1)
    thr = 0.5 if len(sys.argv) <= 2 else float(sys.argv[2])
    bn_out = None if len(sys.argv) <= 3 else float(sys.argv[3])

    post_processing_hb(fn, verbose=verbose)

