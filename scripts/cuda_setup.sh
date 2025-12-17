#!/bin/bash

# Удаляем старый тулкит и остатки драйверов
sudo apt-get purge -y cuda* nvidia*
sudo apt-get autoremove -y

# Устанавливаем официальный репозиторий NVIDIA (WSL-Ubuntu подходит для Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Устанавливаем ТОЛЬКО тулкит (без драйверов!)
sudo apt-get install -y cuda-toolkit-12-9

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc