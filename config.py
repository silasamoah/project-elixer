"""
config.py — Centralised configuration for Project Elixer.

All paths and secrets are read from environment variables (loaded from .env).
Import this module at the top of flask_server.py and yan.py instead of
using hard-coded Windows paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Secrets
# ============================================================
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ============================================================
# Local LLM
# ============================================================
MODEL_PATH: str = os.getenv(
    "MODEL_PATH",
    # Sensible default — override in .env
    str(Path.home() / "models" / "Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
)

# ============================================================
# Piper TTS
# ============================================================
_default_piper_dir = Path(__file__).parent / "piper"
_default_voices_dir = Path(__file__).parent / "voices"

PIPER_EXECUTABLE: Path = Path(
    os.getenv("PIPER_EXECUTABLE", str(_default_piper_dir / "piper"))
)
PIPER_VOICES_DIR: Path = Path(
    os.getenv("PIPER_VOICES_DIR", str(_default_voices_dir))
)

# ============================================================
# Flask server
# ============================================================
FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ============================================================
# HuggingFace / embedding model cache
# ============================================================
HF_HOME: str = os.getenv("HF_HOME", "./hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))

# ============================================================
# Tesseract OCR binary (platform-aware default)
# ============================================================
import platform

if platform.system() == "Windows":
    _default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif platform.system() == "Darwin":
    _default_tesseract = "/opt/homebrew/bin/tesseract"
else:
    _default_tesseract = "/usr/bin/tesseract"

TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", _default_tesseract)

# ============================================================
# Piper voice model definitions
# These mirror the PIPER_MODELS dict in flask_server.py but
# use PIPER_VOICES_DIR from the environment instead of a
# hard-coded Windows path.
# ============================================================
def build_piper_models() -> dict:
    vd = PIPER_VOICES_DIR
    return {
        "southern_english_female": {
            "model": vd / "en_GB_southern_english_female" / "en_GB-southern_english_female-low.onnx",
            "config": vd / "en_GB_southern_english_female" / "en_GB-southern_english_female-low.onnx.json",
            "name": "🇬🇧 Southern English Female (Fast)",
            "language": "en-GB",
            "accent": "Southern English",
            "quality": "Low/Fast",
        },
        "alba_scottish": {
            "model": vd / "en_GB_alba_medium" / "en_GB-alba-medium.onnx",
            "config": vd / "en_GB_alba_medium" / "en_GB-alba-medium.onnx.json",
            "name": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Alba (Scottish Female)",
            "language": "en-GB",
            "accent": "Scottish",
            "quality": "Medium",
        },
        "jenny_dioco": {
            "model": vd / "en_GB_jenny_dioco_medium" / "en_GB-jenny_dioco-medium.onnx",
            "config": vd / "en_GB_jenny_dioco_medium" / "en_GB-jenny_dioco-medium.onnx.json",
            "name": "🇬🇧 Jenny (British Female)",
            "language": "en-GB",
            "accent": "Standard British",
            "quality": "Medium",
        },
        "cori_high_quality": {
            "model": vd / "en_GB_cori_high" / "en_GB-cori-high.onnx",
            "config": vd / "en_GB_cori_high" / "en_GB-cori-high.onnx.json",
            "name": "🇬🇧 Cori (High Quality British)",
            "language": "en-GB",
            "accent": "Standard British",
            "quality": "High ⭐",
        },
    }

PIPER_MODELS: dict = build_piper_models()
