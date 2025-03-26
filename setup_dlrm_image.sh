#!/bin/bash

set -ex
tool_solutions_branch=${1:-"pytorch-aarch64--r24.12"}
# print what branch is used
echo "Branch $tool_solutions_branch is used to build PyTorch"
tool_solutions_pytorch=$HOME/Tool-Solutions/ML-Frameworks/pytorch-aarch64/


install_docker() {
    echo "Installing prerequisites..."
    sudo apt-get install ca-certificates curl -y

    echo "Setting up Docker repository..."
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    echo "Adding Docker repository..."
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin make

    echo "Docker installation complete!"
}

# Install docker
install_docker

# Configure permissions
sudo usermod -aG docker $USER
sudo chmod 666 /var/run/docker.sock

echo "Remove existing Tool-solutions"
sudo rm -rf $HOME/Tool-Solutions


cd $HOME
# Clone and checkout specific branch
git clone https://github.com/ARM-software/Tool-Solutions.git
cd $HOME/Tool-Solutions/
git checkout $tool_solutions_branch


# build pytorch image
cd $tool_solutions_pytorch
./build.sh




