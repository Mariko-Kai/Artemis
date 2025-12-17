import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
MODELS_DIR = "models"

def download_model():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    print(f"Downloading {MODEL_FILENAME} from {MODEL_REPO}...")
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model()
