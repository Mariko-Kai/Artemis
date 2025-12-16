# Project Context: Local AI Agent (MVP)

## Hardware Constraints (CRITICAL)
- **GPU:** NVIDIA GeForce 1650 (4 GB VRAM).
- **RAM:** 16 GB System RAM.
- **CPU:** Intel Core i5-12450H (4 cores / 8 threads).
- **OS:** Windows 11 Pro (WSL2 Ubuntu 22.04 Backend).

## Architectural Requirements
1. **Offline First:** No paid APIs. All models must run locally.
2. **Backend:** Python (FastAPI).
3. **Frontend:** React (Vite).
4. **LLM Engine:** `llama-cpp-python` with GGUF models.
   - *Target Model:* Microsoft Phi-3-mini-4k-instruct (3.8B parameters) or Qwen-2.5-3B.
   - *Quantization:* q4_k_m (must fit in VRAM).
5. **ASR Engine:** `faster-whisper` (optimized CTranslate2 backend) for performance on limited VRAM.
6. **Agent:** LangChain or custom orchestrator using Playwright (headless).

## Path Structure
- `/backend`: FastAPI app.
- `/frontend`: React app.
- `/infra`: Setup scripts.
- `/models`: Local GGUF and Whisper models.