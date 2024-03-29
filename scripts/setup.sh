#!/bin/bash
set -x


# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
conda config --env --set always_yes true
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
conda update -n base -c defaults conda -y

# conda environment setup
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate recon
conda install pytorch=1.9.0  torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

## Downloading  checkpoints
mkdir -p checkpoints && cd checkpoints

# PifuHD
curl "https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt" -o pifuhd.pt

# Light Human Pose estimation
# Check - https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
curl https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -o checkpoint_iter_370000.pth
cd ..
