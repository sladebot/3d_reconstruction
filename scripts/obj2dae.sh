#!/bin/sh

DATA_PATH=$1
mkdir -p $DATA_PATH/dae

for d in $DATA_PATH/*_refine.obj; do
    # echo $d
    filename=$(basename $d)
    echo $filename
    meshlabserver -i $d -o $$DATA_PATH/dae/$filename.dae -m vc vn
done
