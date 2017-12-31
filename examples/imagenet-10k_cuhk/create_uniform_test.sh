#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

#EXAMPLE=examples/imagenet-10k_cuhk/data
DESDIR=data/imagenet-10k_cuhk/data/gv7_data
EXAMPLE=data/imagenet-10k_cuhk/data
DATA=data/imagenet-10k_cuhk/data
TOOLS=build/tools

TRAIN_DATA_ROOT=data/imagenet-10k_cuhk/data/imagenet-10k_cuhk_images/
VAL_DATA_ROOT=data/imagenet-10k_cuhk/data/imagenet-10k_cuhk_images/
TEST_DATA_ROOT=data/imagenet-10k_cuhk/data/imagenet-10k_cuhk_images/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

#echo "Creating train lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
    #--resize_height=$RESIZE_HEIGHT \
    #--resize_width=$RESIZE_WIDTH \
    #--shuffle \
    #--check_size \
    #$TRAIN_DATA_ROOT \
    #$DATA/train.txt \
    #$DESDIR/imagenet-10k_cuhk_train_lmdb

#echo "Creating val lmdb..."

## GLOG_logtostderr=1 $TOOLS/convert_imageset \
##     --resize_height=$RESIZE_HEIGHT \
##     --resize_width=$RESIZE_WIDTH \
##     --shuffle \
##     --check_size \
##     $VAL_DATA_ROOT \
##     $DATA/val.txt \
##     $EXAMPLE/imagenet-10k_cuhk_val_lmdb

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
    #--resize_height=$RESIZE_HEIGHT \
    #--resize_width=$RESIZE_WIDTH \
    #--shuffle \
    #--check_size \
    #$TEST_DATA_ROOT \
    #$DATA/test.txt \
    #$DESDIR/imagenet-10k_cuhk_test_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --check_size \
    $TEST_DATA_ROOT \
    $DATA/uniform_test.txt \
    $DESDIR/uniform_imagenet-10k_cuhk_test_lmdb
echo "Done."
