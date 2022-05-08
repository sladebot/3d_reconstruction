#!/bin/bash

set -ex

mkdir -p checkpoints
cd checkpoints 

# Download PIFUHD
curl "https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt" -o pifuhd.pt
cd ..

