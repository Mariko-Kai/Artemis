#!/bin/bash
set -e

# Determine project root (one level up from scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root determined as: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Enable non-free repositories if on Debian (required for nvidia-cuda-toolkit)
if grep -q "Debian" /etc/issue || [ -f /etc/debian_version ]; then
    sudo sed -i 's/main$/main contrib non-free non-free-firmware/g' /etc/apt/sources.list
fi

# Update usage of repository
sudo apt-get update

# Install build dependencies
sudo apt-get install -y build-essential cmake ffmpeg

# Create virtual environment if it doesn't exist in root
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in $PROJECT_ROOT/venv..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip

# Install llama-cpp-python with CUDA support
# Explicitly set CUDA architecture to 7.5 (Turing/GTX 1650) to avoid "no GPU detected" error during build
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install llama-cpp-python

# Install other dependencies
pip install fastapi uvicorn pydantic-settings huggingface_hub faster-whisper langchain langchain-community playwright duckduckgo-search sentence-transformers faiss-cpu rank_bm25 sqlalchemy aiosqlite alembic redis tenacity psutil tiktoken

# Install Playwright browsers
playwright install chromium

echo "Setup complete. To activate the environment, run 'source venv/bin/activate'"
