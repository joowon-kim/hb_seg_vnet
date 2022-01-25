#!/bin/sh

image="hb_seg_vnet"
cmd="bash"

host_data=$HOME/data            # mount point
env_data=/data/hb_seg           # directory containing predict data
host_code=$HOME/hb_seg_vnet
uid=`id -u`
gid=`id -g`
gpu_option="--gpus all"

docker run $gpu_option -ti --rm \
    -v $host_data:/data \
    -e DATA=$env_data \
    -v $host_code:/code \
    --user $uid:$gid \
    $image $cmd

