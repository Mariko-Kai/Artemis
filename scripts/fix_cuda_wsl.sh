#!/bin/bash
set -e

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Removing conflicting nvidia-cuda-toolkit package..."
sudo apt-get remove -y nvidia-cuda-toolkit || true
sudo apt-get autoremove -y

echo "Ensuring CUDA paths are correct..."
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Verifying nvcc version..."
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found in PATH. Please ensure CUDA 12.x is installed via cuda_setup.sh"
    exit 1
fi
nvcc --version

echo "Reinstalling llama-cpp-python..."
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found in $PROJECT_ROOT"
    exit 1
fi
pip uninstall -y llama-cpp-python
# Explicitly set CUDA architecture to 7.5 (Turing/GTX 1650)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

echo "Fix complete. Please try running the backend again."
