#!/bin/bash
set -x

dir=models



# conda installation
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
conda config --env --set always_yes true
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh
conda update -n base -c defaults conda -y

# conda environment setup
cd /content/ICON
conda env create -f environment.yaml
conda init bash
source ~/.bashrc
source activate icon
pip install -r requirements.txt --use-deprecated=legacy-resolver

# Download ICON
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi

# get the code of ICON
git clone https://github.com/YuliangXiu/ICON.git models/

# download checkpoints
mkdir -p data/smpl_related/models

function download_checkpoints () {
    # username and password input
    echo -e "\nYou need to register at https://icon.is.tue.mpg.de/, according to Installation Instruction."
    read -p "Username (ICON):" username
    read -p "Password (ICON):" password
    username=$(urle $username)
    password=$(urle $password)

    # SMPL (Male, Female)
    echo -e "\nDownloading SMPL..."
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip&resume=1' -O './data/smpl_related/models/SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
    unzip data/smpl_related/models/SMPL_python_v.1.0.0.zip -d data/smpl_related/models
    mv data/smpl_related/models/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_FEMALE.pkl
    mv data/smpl_related/models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_MALE.pkl
    cd data/smpl_related/models
    rm -rf *.zip __MACOSX smpl/models smpl/smpl_webuser
    cd ../../..

    # SMPL (Neutral, from SMPLIFY)
    echo -e "\nDownloading SMPLify..."
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O './data/smpl_related/models/mpips_smplify_public_v2.zip' --no-check-certificate --continue
    unzip data/smpl_related/models/mpips_smplify_public_v2.zip -d data/smpl_related/models
    mv data/smpl_related/models/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl
    cd data/smpl_related/models
    rm -rf *.zip smplify_public
    cd ../../..

    # ICON
    echo -e "\nDownloading ICON..."
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=icon_data.zip&resume=1' -O './data/icon_data.zip' --no-check-certificate --continue
    cd data && unzip icon_data.zip
    mv smpl_data smpl_related/
    rm -f icon_data.zip
    cd ..
}
