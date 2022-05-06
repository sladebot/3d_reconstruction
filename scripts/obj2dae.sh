#!/bin/sh

for d in ./data/FigureSkater/refine/*_refine.obj; do
    # echo $d
    filename=$(basename $d)
    echo $filename
    meshlabserver -i $d -o data/FigureSkater/refine/dae/$filename.dae -m vc vn
done
