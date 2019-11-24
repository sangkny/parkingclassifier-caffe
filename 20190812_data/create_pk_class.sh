#!/usr/bin/env sh 
# Create the char_rec lmdb inputs 
# N.B. set the path to the char_rec train + test data dirs 

FOLDER=20190812_data 
DATA=/workspace/$FOLDER 
TOOLS=build/tools 

TRAIN_DATA_ROOT=$DATA/
TEST_DATA_ROOT=$DATA/

TRAIN_NAME=${FOLDER}_train 
TEST_NAME=${FOLDER}_test 

BACKEND="lmdb" 
# Set RESIZE=true to resize the images to 28x28. Leave as false if images have 
# already been resized using another tool. 

#RESIZE=false 
RESIZE=true 
if $RESIZE; then 
 RESIZE_HEIGHT=32 
 RESIZE_WIDTH=40 
else 
 RESIZE_HEIGHT=0 
 RESIZE_WIDTH=0 
fi 

echo "Removing the existing ${BACKEND}..." 
rm -rf $DATA/${TRAIN_NAME}_${BACKEND} 
rm -rf $DATA/${TEST_NAME}_${BACKEND}
echo "Creating train ${BACKEND}..." 
GLOG_logtostderr=1 $TOOLS/convert_imageset -resize_height=$RESIZE_HEIGHT -resize_width=$RESIZE_WIDTH -gray=true -shuffle -backend=${BACKEND} $TRAIN_DATA_ROOT $DATA/${TRAIN_NAME}.txt $DATA/${TRAIN_NAME}_${BACKEND}

echo "Creating val ${BACKEND}"
GLOG_logtostderr=1 $TOOLS/convert_imageset -resize_height=$RESIZE_HEIGHT -resize_width=$RESIZE_WIDTH -gray=true -shuffle -backend=${BACKEND} $TEST_DATA_ROOT $DATA/${TEST_NAME}.txt  $DATA/${TEST_NAME}_${BACKEND}
echo "Done."
