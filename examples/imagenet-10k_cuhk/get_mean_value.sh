#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=examples/imagenet-10k_cuhk/data
EXAMPLE=data/imagenet-10k_cuhk/data
DATA=data/imagenet-10k_cuhk/data
TOOLS=build/tools

$TOOLS/compute_image_mean -logtostderr $EXAMPLE/imagenet-10k_cuhk_train_lmdb 2>&1 | tee $DATA/imagenet-10k_cuhk_mean_value.log

echo "Done."
