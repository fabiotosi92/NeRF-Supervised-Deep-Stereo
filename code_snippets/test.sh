#!/bin/bash

# Set the visible CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Path to the data directory
data_path="/media/data/"

# Dataset name
dataset="middlebury" 

# Maximum disparity value
max_disparity=256 # for PSMNet only

# Arguments
model=$1 # Specify the model to use: psmnet or raft-stereo
version=$2 # Specify the dataset version: midd-T, midd-A, midd-21, eth3d, kitti15, kitti12
resolution=$3 # Specify the resolution level: F (full), H (half), Q (quarter)

# Set the checkpoint path based on the selected model
case "$model" in
    psmnet) checkpoint="./weights/psmnet-NS.tar" ;;
    raft-stereo) checkpoint="./weights/raftstereo-NS.tar" ;;
esac

# Set the version path based on the selected dataset and version
case "$version" in
    midd-T) version="MiddEval3/training$resolution" ;;
    midd-A) version="MiddEvalA/training$resolution" ;;
    midd-21) version="MiddEval4" ;;
    eth3d) version="ETH3D/Stereo/training" ;;
    kitti15) version="KITTI/2015/training/"; dataset="kitti" ;;
    kitti12) version="KITTI/2012/training/"; dataset="kitti" ;;
esac

# Evaluation on all regions or only on non-occluded regions
occ=true

# Set the output directory path
output_dir="./output/$dataset/$model"
mkdir -p "$output_dir"

# build the base command
command="python test.py \
    --maxdisp \"$max_disparity\" \
    --dataset \"$dataset\" \
    --version \"$version\" \
    --datapath \"$data_path\" \
    --model \"$model\" \
    --outdir \"$output_dir\" \
    --loadmodel \"$checkpoint\""

# Add the --occ option only if occ is set to true
if $occ; then
    command="$command --occ"
fi

# Execute the command
eval $command
