# Project Context: Artemis - Local AI Agent

## Hardware Constraints (CRITICAL)
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM)
- **RAM:** 16 GB System RAM
- **CPU:** Intel Core i5-12450H (8 cores / 12 threads)
- **OS:** Windows 11 Pro with WSL2 (Debian)

## Technology Stack

### Backend (WSL2/Debian)
- **Framework:** FastAPI (Python 3.11+)
- **LLM Engine:** `llama-cpp-python` with CUDA support
  - Model: Phi-3-mini-4k-instruct-q4.gguf (3.8B params, Q4 quantization)
  - Device: CUDA (32 GPU layers)
  - Context: 4096 tokens
- **ASR Engine:** `faster-whisper` (small model)
  - Device: CPU (cuDNN not available in WSL setup)
  - Language: Russian (`ru`)
- **Agent Framework:** LangChain with custom tools (DuckDuckGo, Playwright)
- **Database:** SQLite (sessions.db)
- **Concurrency:** AsyncIO with custom GPU lock (single model instance)

### Frontend (Vite + React)
- **Framework:** React 18 with Vite
- **Routing:** None (single-page app)
- **Styling:** TailwindCSS
- **State Management:** React useState/useEffect
- **Audio:** Web Audio API with silence detection

## Project Structure

```
Artemis/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI entry point, endpoints
│       ├── core/
│       │   ├── config.py        # Settings (model paths, GPU layers)
│       │   ├── llm_engine.py    # Llama model singleton
│       │   ├── asr_engine.py    # Whisper model singleton
│       │   └── global_lock.py   # Async lock for GPU sharing
│       ├── db/
│       │   ├── database.py      # SQLAlchemy setup
│       │   └── models.py        # ChatSession, ChatMessage models
│       ├── routers/
│       │   └── sessions.py      # CRUD endpoints for sessions
│       └── agent/
│           ├── executor.py      # ReAct agent with LangChain
│           └── tools.py         # Search & Browser tools
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app, session & message logic
│   │   ├── main.jsx             # React entry point
│   │   ├── index.css            # Global styles
│   │   └── components/
│   │       ├── Sidebar.jsx      # Session list
│   │       ├── ChatArea.jsx     # Message display
│   │       ├── InputBar.jsx     # Input with agent toggle
│   │       └── AudioRecorder.jsx # Mic with silence detection
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── models/
│   └── Phi-3-mini-4k-instruct-q4.gguf  # LLM weights
│
├── scripts/                     # Setup & utility scripts
│   ├── setup_env.sh             # Install deps, build llama-cpp-python
│   ├── cuda_setup.sh            # Install CUDA toolkit 12.9
│   ├── fix_cuda_wsl.sh          # Remove conflicting nvidia packages
│   └── download_model.py        # Fetch models from Hugging Face
│
├── tests/                       # Integration tests
│   ├── test_setup.py
│   └── verify_persistence.py
│
├── venv/                        # Python virtual environment (root)
├── sessions.db                  # SQLite database
└── PROJECT_CONTEXT.md           # This file
```

## Key Implementation Details

### 1. CUDA Configuration (WSL2)
**Problem:** Conflicting CUDA toolkits caused initialization failures.

**Solution:**
- Use **official NVIDIA CUDA Toolkit 12.9** installed via `cuda_setup.sh`
- Remove Debian's `nvidia-cuda-toolkit` package (older version conflicts)
- Environment variables in `~/.bashrc`:
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```
- Build `llama-cpp-python` with:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" pip install llama-cpp-python
  ```
  (75 = Turing architecture for GTX 1650)

### 2. Audio Transcription
**Problem:** `faster-whisper` requires cuDNN for CUDA, which conflicts with our setup.

**Solution:**
- Run Whisper on **CPU** with `compute_type="int8"`
- Explicitly set `language="ru"` for better Russian accuracy
- Frontend: Auto-stop recording after **2 seconds of silence** using AnalyserNode
- Frontend: Auto-send transcribed text immediately after recognition

**Code Highlights:**
- `backend/app/core/asr_engine.py`: CPU device, Russian language
- `frontend/src/components/AudioRecorder.jsx`: Volume-based silence detection with `requestAnimationFrame`

### 3. GPU Concurrency Control
**Problem:** Single GPU, multiple async requests.

**Solution:**
- Global `asyncio.Lock` in `backend/app/core/global_lock.py`
- All LLM and ASR calls wrapped with `async with gpu_lock:`
- Prevents parallel model execution, serializes requests

### 4. Session Persistence
**Architecture:**
- SQLite database (`sessions.db`) with two tables:
  - `ChatSession`: id, title, timestamp
  - `ChatMessage`: id, session_id, role, content, timestamp
- Backend saves all messages during `/v1/chat/completions`
- Auto-title generation on first message (background task)
- Frontend loads history on session selection

**Endpoints:**
- `POST /v1/sessions` - Create new session
- `GET /v1/sessions` - List all sessions
- `GET /v1/sessions/{id}/messages` - Get session history
- `DELETE /v1/sessions/{id}` - Delete session

### 5. Agent Mode
**Toggle:** "Sparkles" button in InputBar enables ReAct agent.

**Workflow:**
- Uses LangChain's `create_react_agent` with custom tools:
  - **Search Tool:** DuckDuckGo search (5 results)
  - **Browser Tool:** Playwright headless browser for web scraping
- Agent has access to conversation history
- Tool execution is file-access-restricted for security

**Endpoints:**
- `POST /v1/agent/run` - Execute agent task

### 6. Frontend-Backend Communication
**API Base:** Vite proxy forwards `/v1/*` to `http://localhost:8000`

**Key Flows:**
1. **Text Chat:** 
   - `InputBar` → `App.handleSendMessage()` → `/v1/chat/completions` → Update messages
2. **Audio Input:**
   - `AudioRecorder` (silence detected) → `App.handleAudioRecorded()` → `/v1/audio/transcriptions` → Auto-call `handleSendMessage(text)`
3. **Agent Mode:**
   - Same as text chat but calls `/v1/agent/run` instead

## Setup Instructions

### First-Time Setup (WSL2)
```bash
# 1. Install CUDA Toolkit
./scripts/cuda_setup.sh

# 2. Setup Python environment (creates venv in root, installs deps)
./scripts/setup_env.sh

# 3. Download model (if not present)
source venv/bin/activate
python scripts/download_model.py

# 4. Start backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup (Windows Terminal)
```bash
cd frontend
npm install
npm run dev  # Runs on http://localhost:5173
```

### Known Issues & Fixes

#### Issue: "ggml_cuda_init: failed to initialize CUDA"
**Cause:** Conflicting `nvidia-cuda-toolkit` Debian package.

**Fix:** Run `./scripts/fix_cuda_wsl.sh` to remove and reinstall.

#### Issue: Audio transcription crashes backend
**Cause:** Missing cuDNN libraries for GPU Whisper.

**Fix:** Already resolved - Whisper runs on CPU.

#### Issue: Audio recorded but no message sent
**Cause:** Original implementation only set input text.

**Fix:** Already resolved - Auto-sends after transcription.

## Development Guidelines

### Adding a New Endpoint
1. Define route in `backend/app/main.py` or create new router in `routers/`
2. Use `SessionLocal()` or `Depends(get_db)` for database access
3. Wrap LLM calls with `async with gpu_lock:`
4. Update frontend `App.jsx` to call the new endpoint

### Modifying Model Configuration
**File:** `backend/app/core/config.py`

Key settings:
- `MODEL_PATH`: Path to GGUF file
- `N_GPU_LAYERS`: Number of layers on GPU (32 for Phi-3 mini)
- `N_CTX`: Context window size (4096 for Phi-3)

### Adding New Agent Tools
**File:** `backend/app/agent/tools.py`

1. Define tool using `@tool` decorator
2. Add to tool list in `executor.py`
3. Security: Ensure file system access is restricted

### Frontend Component Changes
- Use TailwindCSS for styling (no inline styles)
- State management via props (no global store)
- API calls use native `fetch` (no Axios)
- Audio UI updates must check `mediaRecorder.state` for safe cleanup

## Performance Characteristics
- **LLM Inference:** ~3-5 tokens/sec on GPU (Phi-3 Q4)
- **Audio Transcription:** ~2-3x realtime on CPU (Whisper small)
- **Cold Start:** ~5-10 seconds for first LLM request (model loading)
- **Memory Usage:**
  - LLM: ~2.5 GB VRAM
  - Whisper: ~1 GB System RAM
  - Total: ~3.5/4 GB VRAM used (safe margin)

## Security Considerations
- No authentication (local-only app)
- Agent tools: File access restricted to prevent directory traversal
- CORS: Not configured (same-origin via Vite proxy)
- SQL Injection: Protected by SQLAlchemy ORM

## Future Improvements
- [ ] Streaming responses for LLM
- [ ] Multi-modal input (images via LLaVA)
- [ ] Context window management (auto-summarization)
- [ ] Export/import sessions
- [ ] Dark mode
- [ ] Voice output (TTS)