# Automated Human Habenula Segmentation from T1-weighted Magnetic Resonance Images using V-Net ([Kim and Xu. 2022](#Kim2022))

---

## Introduction

* It is a Habenula segmentation model based on VNet.
* train/validation/test: using HCP S1200 and labels from myelin content-based segmentation ([Kim et al. 2016](#Kim2016) and [Kim et al. 2017](#Kim2018))

* input: T1-weighted MRI aligned to AC-PC
* output: left and right habenula segmentation

## Requirements

* tensorflow 2.2.0
* keras 2.4.3

* (optional) docker

## Install requirements

either of ...

1. Build docker container
    ```
    docker build -t hb_seg_vnet docker/
    ```

2. Install directly

    * (optional) Install CUDA
    * Install python3
    * (optional) Create virtualenv
    * Install tensorflow 2.2.0 and keras 2.4.3
    ```
    pip install tensorflow-gpu==2.2.0 # or tensorflow==2.2.0
    pip install keras==2.4.3
    ```
    * Install other packages
    ```
    pip install nibabel scikit-image numpy
    ```

## Preprocessing: AC-PC alignment and resample to 0.7 mm isotropic resolution

choose one among

1. You can run HCP PreFreesurfer pipeline (https://github.com/Washington-University/Pipelines).

2. You can register your T1w image to the MNI152 template with 6 degrees of freedom (rigid body transformation).

    * If you use FSL,
    ```
    bet T1w T1w_bet -m
    flirt \
        -in T1w \
        -inweight T1w_bet_mask \
        -ref ${FSLDIR}/data/standard/MNI152_T1_1mm \
        -refweight ${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask \
        -dof 6 \
        -interp spline \
        -omat T1w_to_MNI_rigid.mat \
        -applyisoxfm 0.7 \
        -out T1w_to_MNI_rigid
    fsleyes ${FSLDIR}/data/standard/MNI152_T1_1mm T1w_to_MNI_rigid
    ```

3. You can manually rotate your image using, e.g., 3D Slicer.

## Store T1w images and set `config.json` file

* Under a directory, create each subjec's directory.

* Put T1w images to their subject directories. The T1w image filenames should be the same.

* Set the `config.json` file, especially `dn_data` and `fn_t1w`.

## Run the prediction
1. If you installed it in a docker container or virtualenv, enter it
2. Go where `config.json` file
3. run 
    ```
    python /path/to/code/predict.py
    ```

## Example directory/file structure and config.json

* Example directory/file structure
    ```
    + working/
    |   + saved_model/
    |   |   + hb_seg_model.json
    |   |   + hb_seg_weight.data-00000-of-00002
    |   |   + hb_seg_weight.data-00001-of-00002
    |   |   + hb_seg_weight.index
    |   |
    |   + config.json
    |
    + data/
    |   + subj1/
    |   |   + T1w_brain.nii
    |   |
    |   + subj2/
    |       + T1w_brain.nii
    |
    + code/
        + data.py
        + model.py
        + post_processing.py
        + predict.py
        + test.py
        + train.py
        + voxel_similarity.py
    ```

* The `data/config.json` should contain `predict:dn_data: "../data"` and `predict:fn_weight: "saved_model/hb_seg_weight"`

* Call `predict.py` in working/ directory.

    ```
    python ../code/predict.py
    ```

## Note that trained models are not in the github repository.

## Reference
<an name=Kim2022>[1] Kim and Xu, Automated Human Habenula Segmentation from T1-weighted Magnetic Resonance Images using V-Net. bioRxiv, 2022.01.25.477768 https://www.biorxiv.org/content/10.1101/2022.01.25.477768v1

<a name=Kim2016>[2] Kim et al, Human habenula segmentation using myelin content. Neuroimage, 2016, 130 : 145-156 http://www.ncbi.nlm.nih.gov/pubmed/26826517

<a name=Kim2018>[3] Kim et al, Reproducibility of myelin content‐based human habenula segmentation at 3 Tesla. Human Brain Mapping, 2018, 39 : 3058-3071 https://doi.org/10.1002/hbm.24060

