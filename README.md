# Project Elixer 🤖

A local AI assistant with RAG (Retrieval-Augmented Generation), PDF document search, web search, and voice I/O — powered by a local LLaMA-compatible model (Multi model Support) and optional Google Gemini.

---

## Features

- 💬 **Streaming chat** with a local LLM (any GGUF-format model via `llama-cpp-python`)
- 📚 **PDF RAG** — upload documents and query them semantically
- 🌐 **Web search** via DuckDuckGo (firewall-controlled per session)
- ✨ **Gemini integration** — route queries to Google Gemini with `use gemini: <query>`
- 🔊 **Voice output** via Piper TTS (offline, GB English voices)
- 🎤 **Voice input** via browser Web Speech API
- 👤 **User profiles** — persisted preferences, interests, and interaction stats
- 🧠 **Adaptive personality** — warmth/verbosity adjust based on conversation patterns

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/project-elixer.git
cd project-elixer
```

### 2. Set up Python environment

```bash
python -m venv venv
venv\Scripts\Activate.ps1     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU acceleration**: To use CUDA, install `llama-cpp-python` with CUDA support:
>
> ```bash
> CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
> ```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your values (model path, Gemini API key, etc.)
```

### 4. Download a model

Place any GGUF model in a folder of your choice and set `MODEL_PATH` in `.env`.

Recommended: [Llama-3.1-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

### 5. Download the embedding model (first run only)

```bash
# Temporarily allow HuggingFace downloads
$env:HF_HUB_OFFLINE=0; python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./hf_cache')"
```

### 6. Run the server

```bash
python flask_server.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
project-elixer/
├── flask_server.py          # Flask API & streaming endpoints
├── yan.py                   # Core AI logic, RAG, personality, web search
├── config.py                # Centralised config (reads from .env)
├── index.html               # Single-page frontend
├── requirements.txt
├── .env.example             # Template — copy to .env
├── .gitignore
│
├── integration_adapter.py   # RAG integration adapter
├── knowledge_rag_db.py      # SQLite + FAISS knowledge store
├── rag_enhanced.py          # Advanced RAG engine
├── hybrid_rag_manager.py    # Hybrid RAG manager
├── gemini_integration.py    # Gemini REST API wrapper
├── ai_personality_v6_final_production.py
│
├── static/
│   └── libs/               # highlight.js, marked.js (not committed — see below)
│
└── chat_cache/             # Runtime data — not committed
    ├── user_data/
    ├── rag/
    └── temp_uploads/
```

---

## Voice Output (Piper TTS)

Piper is an optional offline TTS engine. To enable it:

1. Download [Piper](https://github.com/rhasspy/piper/releases) for your platform
2. Download one or more [GB English voice models](https://github.com/rhasspy/piper/blob/master/VOICES.md)
3. Set `PIPER_EXECUTABLE` and `PIPER_VOICES_DIR` in your `.env`

Without Piper, voice output is disabled (the rest of the app works normally).

---

## Migrating from the hardcoded-path version

If you cloned this from an older version with paths like `C:\Users\User-Name\...`, the fix is:

1. Copy `.env.example` → `.env`
2. Set `MODEL_PATH`, `PIPER_EXECUTABLE`, and `PIPER_VOICES_DIR` in `.env`
3. Replace the hardcoded path blocks at the top of `flask_server.py` and `yan.py` with:
   ```python
   from config import MODEL_PATH, PIPER_EXECUTABLE, PIPER_VOICES_DIR, PIPER_MODELS
   ```

---

## Chat Commands

| Command                | Effect                        |
| ---------------------- | ----------------------------- |
| `search: <query>`      | Lightweight DuckDuckGo search |
| `deep search: <query>` | Full-page deep search         |
| `use gemini: <query>`  | Route to Gemini API           |
| `/gemini on`           | Enable Gemini globally        |
| `/gemini off`          | Disable Gemini                |
| `/gemini status`       | Show current model            |

---

## Environment Variables Reference

See [`.env.example`](.env.example) for the full list with descriptions.

## Example Correct .env Pattern

GEMINI_API_KEY=your_gemini_api_key_here

MODEL_PATH=C:/Users/user/Downloads/Llama-3.1-8B-Instruct-Q4_K_M.gguf

PIPER_VOICES_DIR=C:/Users/user/Downloads/voices
PIPER_EXECUTABLE=C:/Users/user/Downloads/piper/piper.exe

TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe

FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

HF_HOME=./hf_cache
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1

| Variable           | Required | Description                              |
| ------------------ | -------- | ---------------------------------------- |
| `GEMINI_API_KEY`   | No       | Google AI Studio key for Gemini          |
| `MODEL_PATH`       | **Yes**  | Absolute path to your `.gguf` model      |
| `PIPER_EXECUTABLE` | No       | Path to Piper binary                     |
| `PIPER_VOICES_DIR` | No       | Directory containing voice `.onnx` files |
| `FLASK_PORT`       | No       | Server port (default: 5000)              |

---

## License

MIT
