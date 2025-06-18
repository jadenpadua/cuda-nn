#!/bin/bash
# CUDA installation script for Debian (installs CUDA toolkit + drivers from scratch)
# Run this script on any debian host with an NVIDIA GPU

echo "========================================="
echo "CUDA Installation Script for Debian"
echo "========================================="

echo "Updating system packages..."
sudo apt update

# Enable non-free repository (required for NVIDIA drivers)
echo "Enabling non-free repository..."
sudo sed -i 's/main$/main contrib non-free/' /etc/apt/sources.list
sudo apt update

echo "Installing prerequisites..."
sudo apt install -y wget gnupg2 software-properties-common linux-headers-$(uname -r)

echo "Removing any existing NVIDIA/CUDA installations..."
sudo apt remove --purge -y nvidia-* cuda-* libnvidia-*
sudo apt autoremove -y

echo "Adding NVIDIA package repositories..."
# Detect Debian version and use appropriate repository
DEBIAN_VERSION=$(grep 'VERSION_ID' /etc/os-release | cut -d'"' -f2)
if [ "$DEBIAN_VERSION" = "11" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
    KEYRING_FILE="cuda-keyring_1.0-1_all.deb"
else
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    KEYRING_FILE="cuda-keyring_1.1-1_all.deb"
fi
sudo dpkg -i $KEYRING_FILE
sudo apt update

echo "Installing CUDA toolkit..."
sudo apt install -y cuda-toolkit-12-4

echo "Installing NVIDIA drivers..."
sudo apt install -y nvidia-driver

# Check if DKMS module needs to be built
DKMS_STATUS=$(sudo dkms status nvidia-current 2>/dev/null || echo "not_found")
if [[ $DKMS_STATUS == *"added"* ]] || [[ $DKMS_STATUS == "not_found" ]]; then
    echo "Building NVIDIA kernel module with DKMS..."
    sudo dkms build nvidia-current/$(dpkg -l | grep nvidia-kernel-dkms | awk '{print $3}' | cut -d'-' -f1) 2>/dev/null || true
    sudo dkms install nvidia-current/$(dpkg -l | grep nvidia-kernel-dkms | awk '{print $3}' | cut -d'-' -f1) 2>/dev/null || true
    echo "DKMS build complete."
fi

echo "Setting up environment variables..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Create symlink for cuda if it doesn't exist
if [ ! -L /usr/local/cuda ]; then
    sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda
fi
# Clean up downloaded files
rm -rf $KEYRING_FILE

echo "========================================="
echo "Installation complete! Rebooting system to load in new kernel modules..."
echo "After reboot, please run nvidia-smi to verify install."
echo "========================================="

sudo reboot