# !/bin/bash

# This script sets up the Intel MLPerf on arm.
# Applies patches to remove ipex from Intel's MLPerf.
# Builds ACL, OpenBlas, PyTorch if needed.


set -ex

data_type=${1:-"int8"}

cd $HOME
dlrm_setup_dir=$HOME/dlrm_setup/
dlrm_test_path="$HOME/inference_results_v4.0/closed/Intel/code/dlrm-v2-99.9/pytorch-cpu-int8"

sudo apt update

# install packages
sudo apt install -y software-properties-common \
                    lsb-release scons \
                    build-essential \
                    libtool \
                    autoconf \
                    unzip \
                    git vim wget \
                    numactl \
                    cmake gcc-12 g++-12 \
                    python3-pip python-is-python3

# update gcc & g++ to gcc-12, g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 --slave /usr/bin/g++ g++ /usr/bin/g++-12

sudo chown -R ubuntu:ubuntu *


# Install required libraries for Intel MLPerf setup

#!/bin/bash

# Check if Miniconda is already installed
if [ -d "/opt/conda" ]; then
    echo "Miniconda is already installed at /opt/conda. Skipping installation."
else
    echo "Miniconda not found. Proceeding with installation..."
    
    wget -O "$HOME/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    chmod +x "$HOME/miniconda.sh"
    # Install Miniconda silently (-b flag)
    sudo bash "$HOME/miniconda.sh" -b -p /opt/conda
    # Change ownership to the user
    sudo chown -R ubuntu:ubuntu /opt/conda
    # Remove the installer
    rm "$HOME/miniconda.sh"
    echo "Miniconda installation completed."
fi

/opt/conda/bin/conda clean -ya

export PATH="/opt/conda/bin:$PATH"
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc

conda env update --file $dlrm_setup_dir/environment.yml --name base


sudo chown -R ubuntu:ubuntu /usr/local/*
sudo chown -R ubuntu:ubuntu /usr/lib/python3/*

# Build Loadgen
cd $HOME
rm -rf inference
git clone --recurse-submodules https://github.com/mlcommons/inference.git inference
cd inference
git submodule update --init --recursive && cd loadgen
CFLAGS="-std=c++14" python setup.py bdist_wheel
pip install dist/*.whl

# clone Intel MLPerf Repo
cd $HOME
rm -rf inference_results_v4.0
git clone https://github.com/mlcommons/inference_results_v4.0.git
cd inference_results_v4.0
git checkout ceef1ea

num_proc=$(nproc)

# Apply patches for int8/fp32
if [ "$data_type" = "fp32" ]; then
    echo "Applying fp32 patch"
    target_qps=$((8 * num_proc))
    git apply $dlrm_setup_dir/mlperf_patches/arm_fp32.patch
else
    echo "Applying int8 patch"
    target_qps=$((30 * num_proc))
    git apply $dlrm_setup_dir/mlperf_patches/arm_int8.patch
fi

# Download & Patch the Embedding code
wget -P $dlrm_test_path/python/model \
https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/release/2.1/intel_extension_for_pytorch/nn/modules/merged_embeddingbag.py

patch $dlrm_test_path/python/model/merged_embeddingbag.py < $dlrm_setup_dir/mlperf_patches/merged_embeddingbag.patch 

# Set target QPS as per test configuration
if [ -n "$target_qps" ]; then
	    cd $dlrm_test_path/
	        sed -i.bak "s/dlrm\.Offline\.target_qps = [0-9.]*/dlrm.Offline.target_qps = $target_qps/" user_default.conf
fi

# set number of cores in the config
cd $dlrm_test_path/
sed -i "s/^number_cores = .*/number_cores = $(nproc)/" user_default.conf

echo "Target QPS is changed"
cat user_default.conf

## link int8 model
cd $dlrm_test_path/
ln -s $HOME/model/aarch64_dlrm_int8.pt dlrm_int8.pt