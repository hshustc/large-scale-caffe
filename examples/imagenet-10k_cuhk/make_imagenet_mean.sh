#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=examples/imagenet-10k_cuhk/data
DESDIR=data/imagenet-10k_cuhk/data/gv7_data
EXAMPLE=data/imagenet-10k_cuhk/data
DATA=data/imagenet-10k_cuhk/data
TOOLS=build/tools

$TOOLS/compute_image_mean $DESDIR/imagenet-10k_cuhk_train_lmdb \
  $DESDIR/imagenet-10k_cuhk_mean.binaryproto

echo "Done."
