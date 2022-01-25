# hb_seg_vnet
Habenula Segmentation using VNet

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

## Preprocessing: AC-PC alignment

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

## Note that trained models are not in the github repository.

## Reference
<a name=Kim2016>[1] Kim et al, Human habenula segmentation using myelin content. Neuroimage, 2016, 130 : 145-156 http://www.ncbi.nlm.nih.gov/pubmed/26826517

<a name=Kim2018>[2] Kim et al, Reproducibility of myelin content‐based human habenula segmentation at 3 Tesla. Human Brain Mapping, 2018, 39 : 3058-3071 https://doi.org/10.1002/hbm.24060

<an name=Kim2022>[3] Kim and Xu, Automated Human Habenula Segmentation from T1-weighted Magnetic Resonance Images using V-Net.
