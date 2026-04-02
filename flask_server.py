from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import threading
import os
import numpy as np
import json
import time
import re
import sys
import os
import queue  # For thread-safe thought streaming

from config import MODEL_PATH, PIPER_EXECUTABLE, PIPER_VOICES_DIR, PIPER_MODELS

# Add stream tracking for abort handling
active_streams = {}
stream_id_lock = threading.Lock()


def generate_stream_id():
    """Generate unique stream ID for tracking streams"""
    import time
    return str(time.time())

from flask import Flask, jsonify, request, render_template

import logging

DEBUG_MODE = True


def debug_log(message: str):
    logging.debug(message)


import click

click.echo = lambda *args, **kwargs: None


# Then your imports
from flask import Flask, request, jsonify
from yan import WEB_FIREWALL, pdf_tutor

import yan
# --------------------
# Import humanizer from yan.py
# --------------------
try:
    from yan import (
        pdf_tutor,
        persistent_data,
        emb_model,
        llm,
        clear_search_cache,
        export_chat_pdf,
        clear_history,
        format_llm_response,
        run_web_search,
        get_pdf_status,
        RAG_ADAPTER,
        process_pdfs,
        RAG_ENGINE,
        humanize_response,  
        humanize_response_sentient,  
        user_manager,
        extract_user_info_from_message,
        delete_specific_pdfs,
        get_pdf_list,
        get_pdf_count,
        refresh_rag_metadata, 
        rag_metadata, 
        USE_GEMINI,  # ✨ NEW: For Gemini detection
        gemini_llm,  # ✨ NEW: For Gemini info
        #handle_personality_commands,  # ✨ Personality commands

    )

    print("✅ Successfully imported from yan.py")
except ImportError as e:
    print(f"❌ Failed to import from yan.py: {e}")
    print("⚠️ Make sure yan.py is in the same directory")


    # ✅ Fallback: define rag_metadata locally if import fails
    rag_metadata = {
        "total_chunks": 0,
        "total_documents": 0,
        "last_updated": None,
    }

    def refresh_rag_metadata():
        """Fallback function if yan.py import fails"""
        global rag_metadata
        debug_log("⚠️ Using fallback refresh_rag_metadata")
        rag_metadata["last_updated"] = datetime.now().isoformat()


# --------------------
# Safe imports
# --------------------
try:
    from werkzeug.utils import secure_filename
except ImportError:

    def secure_filename(name):
        return name


try:
    import PyPDF2
except ImportError:
    print("⚠️ PyPDF2 not found, trying pypdf")
    try:
        from pypdf import PdfReader as PyPDF2Reader
    except ImportError:
        PyPDF2 = None
        print("❌ No PDF library found. Install: pip install PyPDF2")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None
    print("⚠️ LangChain not available - using basic splitting")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    FAISS = None
    HuggingFaceEmbeddings = None
    print("⚠️ LangChain vector stores not available")
    print("✅ Successfully imported from yan.py")
except ImportError as e:
    print(f"❌ Failed to import from yan.py: {e}")
    print("⚠️ Make sure yan.py is in the same directory")
    exit(1)


# Initialize default user if none exists
try:
    if user_manager.current_user is None or len(user_manager.all_users) == 0:
        print("Initializing default user...")
        default_user_id = user_manager.create_user(name="Student")
        user_manager.set_current_user(default_user_id)
        print(f"Default user created: {default_user_id}")
except Exception as e:
    print(f"Could not initialize default user: {e}")

'''
def get_pdf_list_helper():
    """Helper to get PDF list from yan.py data"""
    try:
        from yan import RAG_ADAPTER, persistent_data
        from datetime import datetime
        
        pdfs = []
        
        if RAG_ADAPTER:
            cursor = RAG_ADAPTER.rag_db.conn.cursor()
            cursor.execute("""
                SELECT DISTINCT source_name, 
                       COUNT(*) as chunk_count,
                       MAX(timestamp) as last_updated
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
                GROUP BY source_name
                ORDER BY source_name
            """)
            
            for row in cursor.fetchall():
                pdfs.append({
                    "filename": row[0],
                    "chunk_count": row[1],
                    "last_updated": datetime.fromtimestamp(row[2]).strftime("%Y-%m-%d %H:%M") if row[2] else "Unknown"
                })
        
        # Add from legacy
        sources = persistent_data.get("sources", [])
        if sources:
            legacy_pdfs = set(sources)
            existing_names = {p["filename"] for p in pdfs}
            
            for pdf_name in legacy_pdfs:
                if pdf_name not in existing_names:
                    chunk_count = sources.count(pdf_name)
                    pdfs.append({
                        "filename": pdf_name,
                        "chunk_count": chunk_count,
                        "last_updated": "Legacy"
                    })
        
        return pdfs
    
    except Exception as e:
        debug_log(f"Error in get_pdf_list_helper: {e}")
        import traceback
        traceback.print_exc()
        return []
'''


# ============================================================
# PIPER TTS INTEGRATION
# ============================================================
import subprocess
import json
import wave
from pathlib import Path
import tempfile
import os

from config import PIPER_VOICES_DIR, PIPER_EXECUTABLE, PIPER_MODELS, TESSERACT_CMD

# ADD THIS:
try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("⚠️ pytesseract not installed — OCR features disabled")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

"""
# Voice models
# Voice models - UPDATED with all GB Female voices
PIPER_MODELS = {
    "southern_english_female": {
        "model": PIPER_VOICES_DIR
        / "en_GB_southern_english_female"
        / "en_GB-southern_english_female-low.onnx",
        "config": PIPER_VOICES_DIR
        / "en_GB_southern_english_female"
        / "en_GB-southern_english_female-low.onnx.json",
        "name": "🇬🇧 Southern English Female (Fast)",
        "language": "en-GB",
        "accent": "Southern English",
        "quality": "Low/Fast",
    },
    "alba_scottish": {
        "model": PIPER_VOICES_DIR / "en_GB_alba_medium" / "en_GB-alba-medium.onnx",
        "config": PIPER_VOICES_DIR
        / "en_GB_alba_medium"
        / "en_GB-alba-medium.onnx.json",
        "name": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Alba (Scottish Female)",
        "language": "en-GB",
        "accent": "Scottish",
        "quality": "Medium",
    },
    "jenny_dioco": {
        "model": PIPER_VOICES_DIR
        / "en_GB_jenny_dioco_medium"
        / "en_GB-jenny_dioco-medium.onnx",
        "config": PIPER_VOICES_DIR
        / "en_GB_jenny_dioco_medium"
        / "en_GB-jenny_dioco-medium.onnx.json",
        "name": "🇬🇧 Jenny (British Female)",
        "language": "en-GB",
        "accent": "Standard British",
        "quality": "Medium",
    },
    "cori_high_quality": {
        "model": PIPER_VOICES_DIR / "en_GB_cori_high" / "en_GB-cori-high.onnx",
        "config": PIPER_VOICES_DIR / "en_GB_cori_high" / "en_GB-cori-high.onnx.json",
        "name": "🇬🇧 Cori (High Quality British)",
        "language": "en-GB",
        "accent": "Standard British",
        "quality": "High ⭐",
    },
}
"""
def split_into_sentences(text, max_chunk_size=500):
    """
    Split text into sentences for streaming TTS.
    Groups sentences into chunks to avoid too many tiny audio files.
    """
    import re
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def synthesize_with_piper(text, voice="southern_english_female", speed=1.0):
    """
    Synthesize speech using Piper TTS
    For long text, splits into chunks to avoid delays
    Returns WAV file (works in all browsers, no conversion needed!)

    Args:
        text (str): Text to synthesize
        voice (str): Voice model key
        speed (float): Speech rate (0.5 to 2.0)

    Returns:
        tuple: (audio_file_path, error_message)
    """
    try:
        # Validate voice
        if voice not in PIPER_MODELS:
            return (
                None,
                f"Voice '{voice}' not found. Available: {list(PIPER_MODELS.keys())}",
            )

        voice_config = PIPER_MODELS[voice]
        model_path = voice_config["model"]
        config_path = voice_config["config"]

        # Verify files exist
        if not model_path.exists():
            return None, f"Model file not found: {model_path}"
        if not config_path.exists():
            return None, f"Config file not found: {config_path}"

        # Verify Piper executable
        if not PIPER_EXECUTABLE.exists():
            return None, f"Piper executable not found: {PIPER_EXECUTABLE}"

        # ✅ For shorter text, generate all at once (faster)
        if len(text) <= 300:
            return _synthesize_piper_chunk(text, model_path, config_path, speed)
        
        # ✅ For longer text, split into chunks and combine
        debug_log(f"📝 Long text detected ({len(text)} chars), splitting into chunks...")
        chunks = split_into_sentences(text, max_chunk_size=300)
        debug_log(f"📊 Split into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        temp_files = []
        for i, chunk in enumerate(chunks):
            chunk_file, error = _synthesize_piper_chunk(chunk, model_path, config_path, speed)
            if error:
                # Clean up any created files
                for f in temp_files:
                    try:
                        os.remove(f)
                    except:
                        pass
                return None, f"Chunk {i+1} failed: {error}"
            temp_files.append(chunk_file)
        
        # Combine WAV files
        combined_file = _combine_wav_files(temp_files)
        
        # Clean up chunk files
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        
        debug_log(f"✅ Combined {len(chunks)} audio chunks successfully")
        return combined_file, None

    except Exception as e:
        debug_log(f"❌ Piper synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def _synthesize_piper_chunk(text, model_path, config_path, speed=1.0):
    """
    Internal function to synthesize a single chunk of text with Piper.
    Returns: (audio_file_path, error_message)
    """
    try:
        # Create temp WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        # Prepare Piper command
        cmd = [
            str(PIPER_EXECUTABLE),
            "--model",
            str(model_path),
            "--config",
            str(config_path),
            "--output_file",
            temp_wav_path,
        ]

        # Add length scale (inverse of speed)
        length_scale = 1.0 / speed if speed > 0 else 1.0
        length_scale = max(0.5, min(2.0, length_scale))
        cmd.extend(["--length_scale", str(length_scale)])

        # Run Piper with text as stdin
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Shorter timeout for chunks
        timeout = max(10, len(text) * 0.05 + 5)
        stdout, stderr = process.communicate(input=text, timeout=timeout)

        if process.returncode != 0:
            error_msg = stderr or stdout or "Unknown Piper error"
            debug_log(f"❌ Piper chunk error: {error_msg}")
            return None, f"Piper failed: {error_msg}"

        # Verify WAV file was created
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            return None, "Piper did not generate audio file"

        return temp_wav_path, None

    except subprocess.TimeoutExpired:
        return None, f"Piper TTS timed out (text: {len(text)} chars)"
    except Exception as e:
        return None, str(e)


def _combine_wav_files(wav_files):
    """
    Combine multiple WAV files into a single WAV file.
    Returns: combined_file_path
    """
    import wave
    
    # Create output file
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")
    output_path = output.name
    output.close()
    
    # Read first file to get parameters
    with wave.open(wav_files[0], 'rb') as first_wav:
        params = first_wav.getparams()
        
        # Create output WAV with same parameters
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setparams(params)
            
            # Write all audio data
            for wav_file in wav_files:
                with wave.open(wav_file, 'rb') as w:
                    output_wav.writeframes(w.readframes(w.getnframes()))
    
    return output_path


def get_piper_voices():
    """
    Get list of available Piper voices

    Returns:
        list: Available voice configurations
    """
    voices = []

    for key, config in PIPER_MODELS.items():
        if config["model"].exists():
            voices.append(
                {
                    "id": key,
                    "name": config["name"],
                    "language": config["language"],
                    "available": True,
                }
            )
        else:
            voices.append(
                {
                    "id": key,
                    "name": config["name"],
                    "language": config["language"],
                    "available": False,
                    "error": "Model file not found",
                }
            )

    return voices


def get_pdf_list():
    """Get list of all loaded PDFs with metadata"""
    try:
        pdfs = []

        if RAG_ADAPTER:
            cursor = RAG_ADAPTER.rag_db.conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT source_name,
                       COUNT(*) as chunk_count,
                       MAX(timestamp) as last_updated
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
                GROUP BY source_name
                ORDER BY source_name
            """
            )

            from datetime import datetime

            for row in cursor.fetchall():
                pdfs.append(
                    {
                        "filename": row[0],
                        "chunk_count": row[1],
                        "last_updated": (
                            datetime.fromtimestamp(row[2]).strftime("%Y-%m-%d %H:%M")
                            if row[2]
                            else "Unknown"
                        ),
                    }
                )

        sources = persistent_data.get("sources", [])
        if sources:
            existing = {p["filename"] for p in pdfs}
            for name in set(sources):
                if name not in existing:
                    pdfs.append(
                        {
                            "filename": name,
                            "chunk_count": sources.count(name),
                            "last_updated": "Legacy",
                        }
                    )

        return pdfs

    except Exception as e:
        debug_log(f"Error getting PDF list: {e}")
        return []


# --------------------
# APP
# --------------------
app = Flask(__name__, template_folder=".", static_folder="static")

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)


# Update the root route to serve index.html
@app.route("/")
def index():
    return render_template("index.html")


# Profile route
@app.route("/profile")
def profile():
    return render_template("profile.html")


# Add this route for profile.html if accessed directly
@app.route("/profile.html")
def profile_html():
    return render_template("profile.html")


# --------------------
# PATHS
# --------------------
CACHE_DIR = Path("./chat_cache")
PDF_STORAGE_DIR = CACHE_DIR / "temp_uploads"
UPLOAD_FOLDER = PDF_STORAGE_DIR.resolve()

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# STATE
# --------------------
uploaded_files = []
llm_lock = threading.Lock()
profile_lock = threading.Lock()

# ✅ Initialize RAG metadata tracking
rag_metadata = {
    "total_chunks": 0,
    "total_documents": 0,
    "last_updated": None,
}


# --------------------
# HELPERS
# --------------------
def safe_float(v, default=0.0):
    """Safely convert value to float"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_pdf_tutor_output(result):
    """
    Normalize pdf_tutor output to consistent format.
    Returns: (reply, score, sources_list, search_used)
    """
    reply = ""
    score = 0.0
    sources = []
    search_used = False

    try:
        if isinstance(result, tuple):
            if len(result) >= 4:
                reply, score, sources, search_used = result
            elif len(result) >= 3:
                reply, score, sources = result
                search_used = False
            elif len(result) >= 2:
                reply, score = result
                sources = []
                search_used = False
            elif len(result) >= 1:
                reply = result[0]

        elif isinstance(result, dict):
            reply = result.get("response", "") or result.get("answer", "")
            score = result.get("confidence", 0.0)
            sources = result.get("sources", [])
            search_used = result.get("searchUsed", False)
        else:
            reply = str(result)

    except Exception as e:
        print(f"⚠️ Error normalizing output: {e}")
        import traceback

        traceback.print_exc()

    # Return 4 values
    return reply, score, sources, search_used


def extract_pdf_text_simple(filepath):
    """Extract text from PDF using available library"""
    text = ""
    try:
        if PyPDF2:
            with open(filepath, "rb") as f:
                if hasattr(PyPDF2, "PdfReader"):
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                else:
                    # Old PyPDF2 API
                    reader = PyPDF2.PdfFileReader(f)
                    for i in range(reader.numPages):
                        text += reader.getPage(i).extractText() or ""
        return text
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return ""


def coerce_to_text(result):
    if result is None:
        return ""

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, str) and item.strip():
                return item
        return ""

    if isinstance(result, str):
        return result

    return str(result)


# flask_server.py - Enhanced streaming with error boundaries

import traceback
import sys


import re

def should_use_gemini(prompt: str) -> bool:
    """
    Decide whether Gemini should be used based on explicit user commands
    and global configuration.
    """
    try:
        prompt_lc = prompt.lower().strip()

        # 1️⃣ Explicit OFF (highest priority)
        if re.search(r'\b/gemini\s+off\b', prompt_lc):
            return False

        # 2️⃣ Explicit ON
        if re.search(r'\b/gemini\s+on\b', prompt_lc):
            return True

        # 3️⃣ Explicit prefix usage
        if prompt_lc.startswith(("use gemini:", "gemini:")):
            return True

        # 4️⃣ Fallback to global default
        return yan.USE_GEMINI

    except Exception:
        # Fail safe → never force Gemini on
        return False

def stream_llm_response(
    prompt,
    chunks,
    sources,
    chunk_emb,
    emb_model,
    llm,
    allow_web,
    max_tokens,
    search_mode,
    abort_event=None,  # NEW: Add abort event parameter
):
    """
    Enhanced streaming with abort support and client disconnect detection
    """
    import sys

    # Thread-safe queue for thoughts
    thought_queue = queue.Queue()
    pdf_tutor_done = threading.Event()
    result_container = [None]
    error_container = [None]

    def stream_thought(thought_step):
        """Callback that puts thoughts in queue immediately"""
        if abort_event and abort_event.is_set():
            return  # Don't queue if aborted
        try:
            thought_queue.put(thought_step)
            debug_log(f"📤 Queued thought: {thought_step[:50]}...")
        except Exception as e:
            debug_log(f"⚠️ Thought queueing error: {e}")

    def run_pdf_tutor():
        """Run pdf_tutor in background thread"""
        try:
            if abort_event and abort_event.is_set():
                debug_log("🛑 Aborted before pdf_tutor start")
                return
                
            debug_log("🔄 Starting pdf_tutor in background thread...")
            result = pdf_tutor(
                prompt,
                chunks,
                sources,
                chunk_emb,
                emb_model,
                llm,
                allow_web,
                max(max_tokens, 800),
                search_mode,
                thought_stream_callback=stream_thought,
            )
            result_container[0] = result
            debug_log("✅ pdf_tutor completed")
        except Exception as e:
            if not abort_event or not abort_event.is_set():
                debug_log(f"❌ pdf_tutor error: {e}")
                import traceback
                traceback.print_exc()
                error_container[0] = e
        finally:
            pdf_tutor_done.set()

    try:
        # Check abort before starting
        if abort_event and abort_event.is_set():
            return

        # Initialization
        yield json.dumps({"type": "metadata", "data": {"started": True}}) + "\n"
        sys.stdout.flush()

        # Start background thread
        worker_thread = threading.Thread(target=run_pdf_tutor, daemon=True)
        worker_thread.start()
        debug_log("🚀 Background thread started")

        # Process thoughts with abort checking
        thoughts_sent = 0
        last_thought_time = time.time()
        MIN_THOUGHT_INTERVAL = 2.0

        # Phase 1: Process thoughts while pdf_tutor is running
        while not pdf_tutor_done.is_set() or not thought_queue.empty():
            # Check abort
            if abort_event and abort_event.is_set():
                debug_log("🛑 Abort detected, stopping thought processing")
                break

            try:
                thought_step = thought_queue.get(timeout=0.1)

                elapsed = time.time() - last_thought_time
                if elapsed < MIN_THOUGHT_INTERVAL and thoughts_sent > 0:
                    # Check abort during sleep
                    sleep_time = MIN_THOUGHT_INTERVAL - elapsed
                    for _ in range(int(sleep_time * 10)):
                        if abort_event and abort_event.is_set():
                            break
                        time.sleep(0.1)

                if abort_event and abort_event.is_set():
                    break

                thought_json = (
                    json.dumps({"type": "thought", "data": {"step": thought_step}})
                    + "\n"
                )

                print(
                    f"🚀 YIELDING THOUGHT #{thoughts_sent + 1}: {thought_step[:50]}...",
                    flush=True,
                )

                yield thought_json
                sys.stdout.flush()

                thoughts_sent += 1
                last_thought_time = time.time()

                debug_log(
                    f"✨ Streamed thought {thoughts_sent}: {thought_step[:50]}..."
                )

            except queue.Empty:
                if pdf_tutor_done.is_set():
                    break
                time.sleep(0.05)
                continue

        # Final sweep for remaining thoughts
        if not (abort_event and abort_event.is_set()):
            debug_log("🔄 Final sweep for remaining thoughts...")

            for sweep_attempt in range(3):
                time.sleep(0.2)

                if not thought_queue.empty():
                    debug_log(f"📥 Found thoughts in sweep #{sweep_attempt + 1}")

                    while not thought_queue.empty():
                        if abort_event and abort_event.is_set():
                            break

                        try:
                            thought_step = thought_queue.get_nowait()

                            elapsed = time.time() - last_thought_time
                            if elapsed < MIN_THOUGHT_INTERVAL and thoughts_sent > 0:
                                time.sleep(MIN_THOUGHT_INTERVAL - elapsed)

                            thought_json = (
                                json.dumps(
                                    {"type": "thought", "data": {"step": thought_step}}
                                )
                                + "\n"
                            )

                            print(
                                f"🚀 YIELDING FINAL THOUGHT #{thoughts_sent + 1}: {thought_step[:50]}...",
                                flush=True,
                            )

                            yield thought_json
                            sys.stdout.flush()

                            thoughts_sent += 1
                            last_thought_time = time.time()

                        except queue.Empty:
                            break
                else:
                    debug_log(f"⚪ Sweep #{sweep_attempt + 1}: Queue empty")

        # Check abort before continuing
        if abort_event and abort_event.is_set():
            debug_log("🛑 Aborted, skipping response streaming")
            return

        # Check for errors
        if error_container[0]:
            yield json.dumps(
                {
                    "type": "error",
                    "data": {
                        "message": "Error processing your question",
                        "details": str(error_container[0]) if DEBUG_MODE else None,
                    },
                }
            ) + "\n"
            sys.stdout.flush()
            return

        result = result_container[0]
        if result is None:
            yield json.dumps(
                {"type": "error", "data": {"message": "No result returned"}}
            ) + "\n"
            sys.stdout.flush()
            return

        # Normalize output
        try:
            reply, score, result_sources, search_used = normalize_pdf_tutor_output(
                result
            )
            reply = str(reply) if reply else ""
            reply = reply.strip()

            if not reply:
                reply = "⚠️ I couldn't generate a response. Please try rephrasing your question."

        except Exception as e:
            debug_log(f"❌ Normalization error: {e}")
            import traceback

            traceback.print_exc()

            if isinstance(result, (str, tuple)):
                reply = str(result[0] if isinstance(result, tuple) else result)
            else:
                reply = "⚠️ I encountered an error processing the response."

            score = 0.0
            result_sources = []
            search_used = False

        # Format response
        try:
            reply = humanize_response(reply) if reply else ""
            debug_log(f"\n{'='*80}\n✨ FORMATTED OUTPUT:\n{reply}\n{'='*80}\n")
        except Exception as e:
            debug_log(f"❌ Formatting error: {e}")

        # Check abort before sending metadata
        if abort_event and abort_event.is_set():
            return

        # Send metadata
        try:
            # ✨ BUILD METADATA WITH GEMINI DETECTION
            metadata = {
                "confidence": float(score) if score else 0.0,
                "sources": (
                    result_sources if isinstance(result_sources, list) else []
                ),
                "searchUsed": bool(search_used),
                "mode": search_mode,
            }
            
            # ✨ DETECT AND ADD GEMINI INFO
            try:
                if should_use_gemini(prompt):
                    if gemini_llm is not None:
                        metadata['isGemini'] = True
                        metadata['model'] = getattr(gemini_llm, 'model', 'gemini-2.5-flash')
                        metadata['provider'] = 'gemini'
                        debug_log("✨ Gemini metadata added to response")
            except Exception as gemini_err:
                debug_log(f"⚠️ Gemini metadata error: {gemini_err}")
            
            yield json.dumps(
                {
                    "type": "metadata",
                    "data": metadata,
                }
            ) + "\n"
            sys.stdout.flush()
        except Exception as e:
            debug_log(f"⚠️ Metadata emission error: {e}")

        # Stream the actual response
        try:
            lines = reply.split("\n")

            for line in lines:
                # Check abort between lines
                if abort_event and abort_event.is_set():
                    debug_log("🛑 Aborted during response streaming")
                    break

                try:
                    if not line.strip():
                        yield json.dumps(
                            {"type": "token", "data": {"content": "\n"}}
                        ) + "\n"
                        sys.stdout.flush()
                        time.sleep(0.01)
                        continue

                    yield json.dumps(
                        {"type": "token", "data": {"content": line + "\n"}}
                    ) + "\n"
                    sys.stdout.flush()

                    time.sleep(0.01)

                except Exception as line_error:
                    debug_log(f"⚠️ Error streaming line: {line_error}")
                    continue

        except Exception as e:
            if not abort_event or not abort_event.is_set():
                debug_log(f"❌ Streaming error: {e}")
            yield json.dumps(
                {"type": "error", "data": {"message": "Streaming interrupted"}}
            ) + "\n"
            sys.stdout.flush()

        # Send done signal if not aborted
        if not (abort_event and abort_event.is_set()):
            yield json.dumps({"type": "done"}) + "\n"
            sys.stdout.flush()

    except GeneratorExit:
        # Client disconnected
        debug_log("🔌 Generator exit - client disconnected")
        if abort_event:
            abort_event.set()
        raise
    except Exception as e:
        debug_log(f"💥 Fatal streaming error: {e}")
        import traceback
        traceback.print_exc()


# --------------------
# HEALTH CHECK
# --------------------
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    """Enhanced health check with RAG status"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER

        rag_status = "not_initialized"
        rag_chunks = 0

        if RAG_ADAPTER:
            try:
                stats = RAG_ADAPTER.get_stats()
                rag_status = "active"
                rag_chunks = stats.get("total_chunks", 0)
            except:
                rag_status = "error"

        return jsonify(
            {
                "status": "ok",
                "llm_loaded": llm is not None,
                "embedding_model_loaded": emb_model is not None,
                "rag_system": rag_status,
                "rag_chunks": rag_chunks,
                "threaded_mode": True,
                "streaming_enabled": True,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================
# NEW RAG ENDPOINTS
# ============================================================


@app.route("/api/rag/stats", methods=["GET", "OPTIONS"])
def rag_stats():
    if request.method == "OPTIONS":
        return "", 200

    try:
        if RAG_ADAPTER is None:
            return jsonify(
                {
                    "stats": {
                        "rag_enabled": False,
                        "total_chunks": 0,
                        "total_documents": 0,
                        "error": "RAG system not initialized",
                    }
                }
            )

        # Force refresh
        refresh_rag_metadata()

        # Get stats from RAG adapter
        adapter_stats = RAG_ADAPTER.get_stats()

        return jsonify(
            {
                "stats": {
                    "rag_enabled": True,
                    "total_chunks": adapter_stats.get("total_chunks", 0),
                    "total_documents": adapter_stats.get("total_sources", 0),
                    "total_sources": adapter_stats.get("total_sources", 0),
                    "index_size": adapter_stats.get("faiss_vectors", 0),
                    "last_updated": rag_metadata.get("last_updated", "Never"),
                }
            }
        )

    except Exception as e:
        debug_log(f"❌ RAG stats error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"stats": {"rag_enabled": False, "error": str(e)}}), 500


@app.route("/api/rag/search", methods=["POST", "OPTIONS"])
def rag_search():
    """Direct search of RAG knowledge base"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import search_rag_knowledge

        data = request.json or {}
        query = data.get("query", "")
        top_k = data.get("top_k", 5)

        if not query:
            return jsonify({"error": "No query provided"}), 400

        results = search_rag_knowledge(query, top_k)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/cache/clear", methods=["POST", "OPTIONS"])
def clear_rag_cache_endpoint():
    """Clear RAG query cache"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import clear_rag_cache

        result = clear_rag_cache()

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/rag/index/rebuild", methods=["POST", "OPTIONS"])
def rebuild_rag_index_endpoint():
    """Rebuild FAISS index"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import rebuild_rag_index

        result = rebuild_rag_index()

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/rag/history", methods=["GET", "OPTIONS"])
def get_rag_history():
    """Get chat history from RAG database"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER

        if RAG_ADAPTER is None:
            return jsonify({"error": "RAG system not initialized"}), 503

        limit = request.args.get("limit", 50, type=int)
        history = RAG_ADAPTER.rag_db.get_chat_history(limit=limit)

        return jsonify({"history": history, "count": len(history)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/sources", methods=["GET", "OPTIONS"])
def get_rag_sources():
    """Get list of all sources in RAG database"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER

        if RAG_ADAPTER is None:
            return jsonify({"error": "RAG system not initialized"}), 503

        cursor = RAG_ADAPTER.rag_db.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT source_name, source_type, COUNT(*) as chunk_count
            FROM knowledge_chunks
            GROUP BY source_name, source_type
            ORDER BY chunk_count DESC
        """
        )

        sources = []
        for row in cursor.fetchall():
            sources.append({"name": row[0], "type": row[1], "chunk_count": row[2]})

        return jsonify({"sources": sources, "count": len(sources)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/knowledge/add", methods=["POST", "OPTIONS"])
def add_knowledge_manually():
    """Manually add knowledge to RAG database"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER
        from knowledge_rag_db import KnowledgeChunk, SourceType, ValidationStatus
        import time
        import hashlib

        if RAG_ADAPTER is None:
            return jsonify({"error": "RAG system not initialized"}), 503

        data = request.json or {}
        content = data.get("content", "")
        source_name = data.get("source_name", "Manual Entry")
        source_type = data.get("source_type", "USER_MESSAGE")

        if not content:
            return jsonify({"error": "No content provided"}), 400

        # Generate chunk ID
        chunk_id = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]

        # Generate embedding
        embedding = None
        if RAG_ADAPTER.rag_db.embedding_model:
            embedding = RAG_ADAPTER.rag_db.embedding_model.encode(content)

        # Create chunk
        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            content=content,
            source_type=SourceType[source_type],
            source_name=source_name,
            source_url=None,
            confidence=0.9,
            timestamp=time.time(),
            validation_status=ValidationStatus.VALID,
            embedding=embedding,
            metadata=data.get("metadata", {}),
        )

        success = RAG_ADAPTER.rag_db.add_chunk(chunk)

        if success:
            return jsonify(
                {
                    "success": True,
                    "chunk_id": chunk_id,
                    "message": "Knowledge added successfully",
                }
            )
        else:
            return (
                jsonify({"success": False, "message": "Failed to add knowledge"}),
                500,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/knowledge/delete", methods=["DELETE", "OPTIONS"])
def delete_knowledge():
    """Delete knowledge chunk by ID"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER

        if RAG_ADAPTER is None:
            return jsonify({"error": "RAG system not initialized"}), 503

        data = request.json or {}
        chunk_id = data.get("chunk_id", "")

        if not chunk_id:
            return jsonify({"error": "No chunk_id provided"}), 400

        cursor = RAG_ADAPTER.rag_db.conn.cursor()
        cursor.execute("DELETE FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))
        RAG_ADAPTER.rag_db.conn.commit()

        if cursor.rowcount > 0:
            return jsonify({"success": True, "message": f"Deleted chunk {chunk_id}"})
        else:
            return jsonify({"success": False, "message": "Chunk not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/export", methods=["GET", "OPTIONS"])
def export_rag_data():
    """Export RAG database as JSON"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        from yan import RAG_ADAPTER
        import json
        from datetime import datetime

        if RAG_ADAPTER is None:
            return jsonify({"error": "RAG system not initialized"}), 503

        cursor = RAG_ADAPTER.rag_db.conn.cursor()

        # Get all chunks (without embeddings for size)
        cursor.execute(
            """
            SELECT chunk_id, content, source_type, source_name, source_url,
                   confidence, timestamp, validation_status, metadata
            FROM knowledge_chunks
            ORDER BY timestamp DESC
        """
        )

        chunks = []
        for row in cursor.fetchall():
            chunks.append(
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "source_type": row[2],
                    "source_name": row[3],
                    "source_url": row[4],
                    "confidence": row[5],
                    "timestamp": row[6],
                    "validation_status": row[7],
                    "metadata": json.loads(row[8]),
                }
            )

        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "chunks": chunks,
            "stats": RAG_ADAPTER.get_stats(),
        }

        return jsonify(export_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/profile", methods=["GET", "OPTIONS"])
def get_user_profile():
    """Thread-safe profile retrieval"""
    if request.method == "OPTIONS":
        return "", 204

    with profile_lock:
        try:
            # Force reload from disk to get latest stats
            user_manager.reload_from_disk()

            user = user_manager.get_current_user()

            if not user:
                debug_log("📝 No user found - creating default user")
                user_id = user_manager.create_user(name="Student")
                user_manager.set_current_user(user_id)
                user = user_manager.get_current_user()
                debug_log(f"✅ Created default user: {user_id}")

            # Log current stats for debugging
            stats = user.get("interaction_stats", {})
            debug_log(
                f"📊 Profile stats: {stats.get('total_questions', 0)} questions, "
                f"{stats.get('session_count', 0)} sessions"
            )

            return jsonify(
                {
                    "hasProfile": True,
                    "profile": user,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            debug_log(f"❌ Error in get_user_profile: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"error": str(e), "hasProfile": False}), 500


@app.route("/api/user/profile", methods=["POST", "OPTIONS"])
def update_user_profile():
    """Thread-safe profile update"""
    if request.method == "OPTIONS":
        return "", 204

    with profile_lock:
        try:
            data = request.json or {}
            updates = []

            # Update name
            if "name" in data:
                if user_manager.set_user_data("name", data["name"]):
                    updates.append("name")

            # Update email
            if "email" in data:
                if user_manager.set_user_data("email", data["email"]):
                    updates.append("email")

            # Update preferences
            if "preferences" in data:
                for key, value in data["preferences"].items():
                    if user_manager.set_user_data(f"preferences.{key}", value):
                        updates.append(f"preferences.{key}")

            # Update interests
            if "interests" in data:
                if user_manager.set_user_data("interests", data["interests"]):
                    updates.append("interests")

            # Update learning goals
            if "learning_goals" in data:
                if user_manager.set_user_data("learning_goals", data["learning_goals"]):
                    updates.append("learning_goals")

            # Reload to get fresh data
            user_manager.reload_from_disk()

            return jsonify(
                {
                    "success": True,
                    "message": f"Updated: {', '.join(updates)}",
                    "profile": user_manager.get_current_user(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            debug_log(f"❌ Error updating profile: {e}")
            return jsonify({"error": str(e)}), 500


@app.route("/api/user/stats", methods=["GET", "OPTIONS"])
def get_user_stats():
    """Thread-safe stats retrieval"""
    if request.method == "OPTIONS":
        return "", 204

    with profile_lock:
        try:
            # Reload to ensure fresh stats
            user_manager.reload_from_disk()

            stats = user_manager.get_user_data("interaction_stats", {})

            return jsonify(
                {
                    "stats": stats,
                    "summary": user_manager.get_user_summary(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            debug_log(f"❌ Error getting stats: {e}")
            return jsonify({"error": str(e)}), 500


# --------------------
# UPDATE PREFERENCES
# --------------------
@app.route("/api/user/preferences", methods=["POST", "OPTIONS"])
def update_preferences():
    """Update user preferences"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}

        valid_preferences = [
            "response_style",
            "code_style",
            "expertise_level",
            "preferred_language",
            "use_emojis",
            "stream_responses",
        ]

        updated = []
        for key in valid_preferences:
            if key in data:
                user_manager.set_user_data(f"preferences.{key}", data[key])
                updated.append(key)

        return jsonify(
            {
                "success": True,
                "updated": updated,
                "preferences": user_manager.get_user_data("preferences", {}),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# ADD INTEREST
# --------------------
@app.route("/api/user/interests", methods=["POST", "OPTIONS"])
def add_interest():
    """Add a user interest"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        interest = data.get("interest")

        if not interest:
            return jsonify({"error": "No interest provided"}), 400

        user_manager.add_interest(interest)

        return jsonify(
            {"success": True, "interests": user_manager.get_user_data("interests", [])}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# REMOVE INTEREST
# --------------------
@app.route("/api/user/interests/remove", methods=["POST", "OPTIONS"])
def remove_interest():
    """Remove a user interest"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        value = data.get("value")

        if not value:
            return jsonify({"error": "No interest value provided"}), 400

        # Get current interests and remove the value
        interests = user_manager.get_user_data("interests", [])

        if value in interests:
            interests.remove(value)
            # Use set_user_data to properly update
            user_manager.set_user_data("interests", interests)

        return jsonify(
            {"success": True, "interests": user_manager.get_user_data("interests", [])}
        )

    except Exception as e:
        print(f"Error removing interest: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
# ADD LEARNING GOAL
# --------------------
@app.route("/api/user/goals", methods=["POST", "OPTIONS"])
def add_learning_goal():
    """Add a learning goal"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        goal = data.get("goal")

        if not goal:
            return jsonify({"error": "No goal provided"}), 400

        user_manager.add_learning_goal(goal)

        return jsonify(
            {"success": True, "goals": user_manager.get_user_data("learning_goals", [])}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# REMOVE LEARNING GOAL
# --------------------
@app.route("/api/user/goals/remove", methods=["POST", "OPTIONS"])
def remove_learning_goal():
    """Remove a learning goal"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        value = data.get("value")

        if not value:
            return jsonify({"error": "No goal value provided"}), 400

        # Get current goals and remove the value
        goals = user_manager.get_user_data("learning_goals", [])

        if value in goals:
            goals.remove(value)
            # Use set_user_data to properly update
            user_manager.set_user_data("learning_goals", goals)

        return jsonify(
            {"success": True, "goals": user_manager.get_user_data("learning_goals", [])}
        )

    except Exception as e:
        print(f"Error removing goal: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
# ADD BOOKMARK
# --------------------
@app.route("/api/user/bookmarks", methods=["POST", "OPTIONS"])
def add_bookmark():
    """Add a bookmark"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        content = data.get("content")
        title = data.get("title")

        if not content:
            return jsonify({"error": "No content provided"}), 400

        user_manager.add_bookmark(content, title)

        return jsonify(
            {"success": True, "bookmarks": user_manager.get_user_data("bookmarks", [])}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# GET BOOKMARKS
# --------------------
@app.route("/api/user/bookmarks", methods=["GET", "OPTIONS"])
def get_bookmarks():
    """Get all bookmarks"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        bookmarks = user_manager.get_user_data("bookmarks", [])

        return jsonify({"bookmarks": bookmarks, "count": len(bookmarks)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# GET CHAT HISTORY
# --------------------
@app.route("/api/user/history", methods=["GET", "OPTIONS"])
def get_user_history():
    """Get user's chat history"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        history = user_manager.get_user_data("chat_history", [])

        # Optional: limit results
        limit = request.args.get("limit", type=int)
        if limit:
            history = history[-limit:]

        return jsonify({"history": history, "count": len(history)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# CLEAR USER HISTORY
# --------------------
@app.route("/api/user/history/clear", methods=["POST", "OPTIONS"])
def clear_user_history():
    """Clear user's chat history"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        user_manager.set_user_data("chat_history", [])

        return jsonify({"success": True, "message": "Chat history cleared"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# SWITCH USER
# --------------------
@app.route("/api/user/switch", methods=["POST", "OPTIONS"])
def switch_user():
    """Switch to a different user or create new user"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        user_id = data.get("user_id")
        name = data.get("name")
        create_new = data.get("create_new", False)

        if create_new:
            # Create new user
            user_id = user_manager.create_user(name=name)
            user_manager.set_current_user(user_id)

            return jsonify(
                {
                    "success": True,
                    "message": f"Created new user: {name}",
                    "user_id": user_id,
                    "profile": user_manager.get_current_user(),
                }
            )
        elif user_id:
            # Switch to existing user
            user_manager.set_current_user(user_id)

            return jsonify(
                {
                    "success": True,
                    "message": "Switched user",
                    "user_id": user_id,
                    "profile": user_manager.get_current_user(),
                }
            )
        else:
            return jsonify({"error": "No user_id or name provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# LIST ALL USERS
# --------------------
@app.route("/api/users/list", methods=["GET", "OPTIONS"])
def list_users():
    """List all users"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        users = []
        for user_id, profile in user_manager.all_users.items():
            users.append(
                {
                    "user_id": user_id,
                    "name": profile.get("name", "Anonymous"),
                    "last_active": profile.get("interaction_stats", {}).get(
                        "last_active"
                    ),
                    "total_questions": profile.get("interaction_stats", {}).get(
                        "total_questions", 0
                    ),
                }
            )

        return jsonify(
            {
                "users": users,
                "current_user_id": user_manager.current_user,
                "count": len(users),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------
# FIREWALL CONTROL
# --------------------
@app.route("/api/firewall/status", methods=["GET", "OPTIONS"])
def firewall_status():
    """Get current firewall status"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        stats = WEB_FIREWALL.get_stats()
        return jsonify({"enabled": stats["enabled"], "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/firewall/enable", methods=["POST", "OPTIONS"])
def enable_firewall():
    """Enable web search (disable firewall)"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        WEB_FIREWALL.enable()
        return jsonify(
            {"success": True, "message": "Web search enabled", "enabled": True}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/firewall/disable", methods=["POST", "OPTIONS"])
def disable_firewall():
    """Disable web search (enable firewall)"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        WEB_FIREWALL.disable()
        return jsonify(
            {
                "success": True,
                "message": "Web search disabled - firewall active",
                "enabled": False,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# CHAT ENDPOINT (NON-STREAMING - LEGACY)
# ---------------------------------------
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        msg = data.get("message", "").strip()
        allow_web = data.get("allowWeb", True)
        use_rag = data.get("useRAG", True)
        ocr_context = data.get("ocr_context", "").strip()

        if not msg:
            return jsonify(
                {
                    "response": "Please enter a message",
                    "confidence": 0.0,
                    "sources": [],
                    "searchUsed": False,
                    "ragUsed": False,
                }
            )

        if ocr_context:
            msg = f"[Attached files]\n{ocr_context}\n\n---\n\n{msg}"

        # 🔥 SYNC FIREWALL WITH CLIENT SETTING
        if allow_web:
            WEB_FIREWALL.enable()
        else:
            WEB_FIREWALL.disable()

        chunks = persistent_data.get("chunks")
        sources = persistent_data.get("sources")
        chunk_emb = persistent_data.get("chunk_emb")

        if llm is None:
            return jsonify(
                {
                    "response": "❌ AI model not loaded. Please check server logs.",
                    "confidence": 0.0,
                    "sources": [],
                    "searchUsed": False,
                    "ragUsed": False,
                }
            )

        with llm_lock:
            result = pdf_tutor(
                msg,
                chunks,
                sources,
                chunk_emb,
                emb_model,
                llm,
                allow_web,
                data.get("maxCodeTokens", 6000),
                data.get("searchMode", "auto"),
            )

        # Normalize output
        reply, score, result_sources, search_used, thought_steps = (
            normalize_pdf_tutor_output(result)
        )

        if reply:
            reply = humanize_response(reply)

        # Extract user info
        try:
            extract_user_info_from_message(msg)
        except Exception as e:
            print(f"⚠️ User info extraction failed: {e}")

        # Save to history
        try:
            user_manager.add_to_chat_history(msg, reply)
        except Exception as e:
            print(f"⚠️ Chat history update failed: {e}")

        rag_used = use_rag and RAG_ENGINE is not None and len(result_sources) > 0

        return jsonify(
            {
                "response": reply or "No response generated",
                "confidence": score,
                "sources": result_sources,
                "searchUsed": search_used,
                "ragUsed": rag_used,
            }
        )

    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback

        traceback.print_exc()

        return (
            jsonify(
                {
                    "response": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "sources": [],
                    "searchUsed": False,
                    "ragUsed": False,
                }
            ),
            500,
        )


# --------------------
# STREAMING CHAT ENDPOINT (NEW)
# --------------------
@app.route("/api/chat/stream", methods=["POST", "OPTIONS"])
def chat_stream():
    """Stream endpoint with proper abort handling"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        msg = data.get("message", "").strip()
        ocr_context = data.get("ocr_context", "").strip()

        # Validate input on the user message only, before OCR context is added
        if not msg:
            return jsonify({"error": "No message provided"}), 400

        if len(msg) > 5000:
            return jsonify({"error": "Message too long (max 5000 chars)"}), 400

        # Attach OCR context after the length check so large files don't trip the limit
        if ocr_context:
            msg = f"[Attached files]\n{ocr_context}\n\n---\n\n{msg}"


        if llm is None:
            return jsonify({"error": "AI model not loaded"}), 503

        # Sync firewall
        if data.get("allowWeb", True):
            WEB_FIREWALL.enable()
        else:
            WEB_FIREWALL.disable()

        # Get PDF context
        chunks = persistent_data.get("chunks")
        sources = persistent_data.get("sources")
        chunk_emb = persistent_data.get("chunk_emb")

        # Generate stream ID and create abort event
        stream_id = generate_stream_id()
        abort_event = threading.Event()
        
        with stream_id_lock:
            active_streams[stream_id] = abort_event

        # Create streaming response with abort detection
        def generate():
            try:
                with llm_lock:
                    # Check if aborted before starting
                    if abort_event.is_set():
                        debug_log(f"🛑 Stream {stream_id} aborted before generation")
                        return
                    
                    for chunk in stream_llm_response(
                        msg,
                        chunks,
                        sources,
                        chunk_emb,
                        emb_model,
                        llm,
                        data.get("allowWeb", True),
                        data.get("maxCodeTokens", 6000),
                        data.get("searchMode", "auto"),
                        abort_event,  # Pass abort event
                    ):
                        # Check for abort between chunks
                        if abort_event.is_set():
                            debug_log(f"🛑 Stream {stream_id} aborted during generation")
                            break
                        yield chunk
            except GeneratorExit:
                # Client disconnected
                debug_log(f"🔌 Client disconnected for stream {stream_id}")
                abort_event.set()
            except Exception as e:
                debug_log(f"💥 Generator exception for stream {stream_id}: {e}")
                import traceback
                traceback.print_exc()
                abort_event.set()

                # Try to send error
                try:
                    yield json.dumps(
                        {"type": "error", "data": {"message": "Stream failed"}}
                    ) + "\n"
                except:
                    pass
            finally:
                # Clean up
                with stream_id_lock:
                    active_streams.pop(stream_id, None)
                debug_log(f"🧹 Cleaned up stream {stream_id}")

        # Wrap generator to detect client disconnect
        def flush_generator():
            try:
                for chunk in generate():
                    yield chunk
                    import sys
                    sys.stdout.flush()
            except GeneratorExit:
                # Client disconnected
                abort_event.set()
                debug_log(f"🔌 Client disconnected (flush generator) for stream {stream_id}")
                raise

        return Response(
            stream_with_context(flush_generator()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Stream-ID": stream_id,
            },
        )

    except Exception as e:
        debug_log(f"❌ Stream endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------
# PDF UPLOAD
# --------------------
@app.route("/api/upload-pdfs", methods=["POST", "OPTIONS"])
def upload_pdfs():
    if request.method == "OPTIONS":
        return "", 204

    try:
        files = request.files.getlist("files")

        if not files:
            return jsonify({"error": "No files provided"}), 400

        saved_paths = []
        for f in files:
            if not f.filename.endswith(".pdf"):
                continue

            name = secure_filename(f.filename)
            path = UPLOAD_FOLDER / name
            f.save(path)
            # Use absolute path and normalize separators
            abs_path = path.resolve()
            saved_paths.append(str(abs_path).replace('\\', '/'))
            debug_log(f"📄 Saved PDF to: {abs_path}")

            if name not in uploaded_files:
                uploaded_files.append(name)

        if not saved_paths:
            return jsonify({"error": "No valid PDF files"}), 400

        # ✅ FIX 1: Call process_pdfs and handle response properly
        result = process_pdfs(saved_paths)

        # ✅ FIX 2: Handle both dict and string responses
        if isinstance(result, dict):
            if "error" in result:
                return jsonify({"error": result["error"]}), 500

            result_msg = result.get("message", "PDFs processed")
            stats = result.get("stats", {})

        else:
            # Legacy string response
            result_msg = str(result)
            stats = {}

        # ✅ FIX 3: Refresh metadata AFTER processing
        try:
            refresh_rag_metadata()
        except Exception as e:
            debug_log(f"⚠️ Failed to refresh metadata: {e}")

        # ✅ FIX 4: Get updated stats safely
        total_chunks = 0
        total_docs = 0

        if RAG_ADAPTER:
            try:
                fresh_stats = RAG_ADAPTER.get_stats()
                total_chunks = fresh_stats.get("total_chunks", 0)
                total_docs = fresh_stats.get("total_sources", 0)
            except Exception as e:
                debug_log(f"⚠️ Failed to get RAG stats: {e}")

        # ✅ FIX 5: Fallback to persistent data if RAG fails
        if total_chunks == 0:
            chunks = persistent_data.get("chunks", [])
            sources = persistent_data.get("sources", [])
            total_chunks = len(chunks) if chunks else 0
            total_docs = len(set(sources)) if sources else 0

        # ✅ FIX 6: Safe metadata access
        try:
            last_updated = rag_metadata.get("last_updated")
        except (NameError, AttributeError):
            last_updated = datetime.now().isoformat()

        # ✅ FIX 7: Return consistent response format
        return jsonify(
            {
                "message": result_msg,
                "files": uploaded_files,
                "totalChunks": total_chunks,
                "totalDocuments": total_docs,
                "ragEnabled": RAG_ADAPTER is not None,
                "rag_stats": {
                    "total_chunks": total_chunks,
                    "total_documents": total_docs,
                    "last_updated": last_updated,
                },
                "detailed_stats": stats,
            }
        )

    except Exception as e:
        debug_log(f"❌ Upload error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
# PDF STATUS
# --------------------


@app.route("/api/pdf-status", methods=["GET", "OPTIONS"])
def pdf_status():
    if request.method == "OPTIONS":
        return "", 200

    try:
        # Get formatted status message
        status_msg = get_pdf_status()

        # ✅ SAFE: Refresh metadata with error handling
        try:
            refresh_rag_metadata()
        except Exception as e:
            debug_log(f"⚠️ Failed to refresh metadata: {e}")

        # Get PDF list for additional data
        pdf_list = get_pdf_list()

        # ✅ SAFE: Access metadata with fallback
        try:
            metadata = {
                "total_chunks": rag_metadata.get("total_chunks", 0),
                "total_documents": len(pdf_list),
                "last_updated": rag_metadata.get("last_updated"),
            }
        except (NameError, AttributeError):
            metadata = {
                "total_chunks": 0,
                "total_documents": len(pdf_list),
                "last_updated": datetime.now().isoformat(),
            }

        return jsonify(
            {
                "status": status_msg,
                "pdf_list": pdf_list,
                "metadata": metadata,
                "success": True,
            }
        )

    except Exception as e:
        debug_log(f"❌ PDF status error: {e}")
        import traceback

        traceback.print_exc()
        return (
            jsonify({"status": f"⚠️ Error: {str(e)}", "error": True, "success": False}),
            500,
        )


def refresh_rag_metadata():
    """
    Refresh RAG metadata from current state - FIXED
    Call this after any operation that changes RAG data
    """
    global rag_metadata

    try:
        if RAG_ADAPTER:
            # Get fresh stats from database
            cursor = RAG_ADAPTER.rag_db.conn.cursor()

            # Count total chunks
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM knowledge_chunks
            """
            )
            total_chunks = cursor.fetchone()["count"]

            # Count unique sources
            cursor.execute(
                """
                SELECT COUNT(DISTINCT source_name) as count 
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
            """
            )
            total_docs = cursor.fetchone()["count"]

            rag_metadata["total_chunks"] = total_chunks
            rag_metadata["total_documents"] = total_docs
        else:
            chunks = persistent_data.get("chunks") or []
            sources = persistent_data.get("sources") or []
            rag_metadata["total_chunks"] = len(chunks)
            rag_metadata["total_documents"] = len(set(sources)) if sources else 0

        rag_metadata["last_updated"] = datetime.now().isoformat()

        debug_log(
            f"📊 Metadata refreshed: {rag_metadata['total_chunks']} chunks, {rag_metadata['total_documents']} docs"
        )

    except Exception as e:
        debug_log(f"❌ Failed to refresh metadata: {e}")
        import traceback

        traceback.print_exc()


# --------------------
# RAG STATUS
# --------------------
@app.route("/api/rag-status", methods=["GET", "OPTIONS"])
def rag_status():
    if request.method == "OPTIONS":
        return "", 200

    try:
        # ✅ SAFE: Refresh metadata with error handling
        try:
            refresh_rag_metadata()
        except Exception as e:
            debug_log(f"⚠️ Failed to refresh metadata: {e}")

        # Get fresh stats from RAG adapter
        if RAG_ADAPTER:
            stats = RAG_ADAPTER.get_stats()
            total_chunks = stats.get("total_chunks", 0)
            total_docs = stats.get("total_sources", 0)
            detailed_stats = stats
        else:
            # Fallback to persistent data
            chunks = persistent_data.get("chunks") or []
            sources = persistent_data.get("sources") or []
            total_chunks = len(chunks)
            total_docs = len(set(sources)) if sources else 0
            detailed_stats = {}

        # ✅ SAFE: Get last_updated with fallback
        try:
            last_updated = rag_metadata.get("last_updated")
        except (NameError, AttributeError):
            last_updated = datetime.now().isoformat()

        return jsonify(
            {
                "enabled": RAG_ADAPTER is not None,
                "total_chunks": total_chunks,
                "total_documents": total_docs,
                "last_updated": last_updated,
                "uploaded_files": uploaded_files,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "detailed_stats": detailed_stats,
            }
        )

    except Exception as e:
        debug_log(f"❌ RAG status error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"enabled": False, "error": str(e)}), 500


# --------------------
# RAG STATS
# --------------------
"""
@app.route("/api/rag/stats", methods=["GET", "OPTIONS"])
def rag_stats():
    if request.method == "OPTIONS":
        return "", 200

    chunks = persistent_data.get("chunks") or []
    sources = persistent_data.get("sources") or []

    return jsonify(
        {
            "stats": {
                "total_chunks": len(chunks),
                "total_documents": len(set(sources)) if sources else 0,
                "rag_enabled": RAG_ENGINE is not None,
                "uploaded_files": len(uploaded_files),
                "last_updated": rag_metadata.get("last_updated", "Never"),
            }
        }
    )

"""


# --------------------
# SEMANTIC SEARCH
# --------------------
@app.route("/api/semantic-search", methods=["POST", "OPTIONS"])
def semantic_search():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json or {}
        query = data.get("query", "")
        k = data.get("k", 5)

        if not query:
            return jsonify({"error": "No query provided"}), 400

        if RAG_ENGINE is None:
            return jsonify({"error": "RAG not available"}), 503

        results = RAG_ENGINE.retrieve(query, top_k=k)

        formatted_results = [
            {
                "content": r.document.content,
                "source": r.document.source,
                "score": float(r.score),
                "page": r.document.metadata.get("page", 0),
            }
            for r in results
        ]

        return jsonify(
            {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            }
        )

    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({"error": str(e)}), 500


# --------------------
# CLEAR HISTORY
# --------------------
@app.route("/api/clear-history", methods=["POST", "OPTIONS"])
def clear_chat_history():
    if request.method == "OPTIONS":
        return "", 204

    try:
        result, _ = clear_history()
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"message": str(e)})


# --------------------
# CLEAR CACHE
# --------------------
@app.route("/api/clear-cache", methods=["POST", "OPTIONS"])
def clear_cache():
    if request.method == "OPTIONS":
        return "", 204

    try:
        result = clear_search_cache()
        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"message": str(e)})


# --------------------
# EXPORT CHAT
# --------------------
@app.route("/api/export-chat", methods=["POST", "OPTIONS"])
def export_chat():
    if request.method == "OPTIONS":
        return "", 204

    try:
        result = export_chat_pdf()

        pdf_files = list(Path(".").glob("chat_with_alex_*.pdf"))
        if pdf_files:
            latest = max(pdf_files, key=lambda p: p.stat().st_mtime)
            return send_file(latest, as_attachment=True, download_name=latest.name)

        return jsonify({"message": "No chat to export"}), 404

    except Exception as e:
        return jsonify({"message": str(e)}), 500


# --------------------
# PDF LIST
# --------------------


@app.route("/api/pdf/list", methods=["GET", "OPTIONS"])
def get_pdf_list_endpoint():
    """Get list of all loaded PDFs"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        # Get fresh list
        pdf_list = get_pdf_list()

        # Refresh metadata
        refresh_rag_metadata()

        return jsonify(
            {
                "pdfs": pdf_list,
                "count": len(pdf_list),
                "stats": {
                    "total_chunks": rag_metadata.get("total_chunks", 0),
                    "total_documents": rag_metadata.get("total_documents", 0),
                    "last_updated": rag_metadata.get("last_updated"),
                },
            }
        )

    except Exception as e:
        debug_log(f"❌ Get PDF list error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
# DELETE PDF
# --------------------
@app.route("/api/pdf/delete", methods=["POST", "OPTIONS"])
def delete_pdf_endpoint():
    """Delete specific PDF(s)"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        pdf_names = data.get("pdf_names", [])

        if not pdf_names:
            return jsonify({"error": "No PDF names provided"}), 400

        # Ensure it's a list
        if isinstance(pdf_names, str):
            pdf_names = [pdf_names]

        # Delete PDFs
        result_msg = delete_specific_pdfs(pdf_names)

        # ✅ Also remove files from temp_uploads (UPLOAD_FOLDER), which
        # delete_specific_pdfs does not know about (it only cleans CACHE_DIR/pdfs).
        for pdf_name in pdf_names:
            temp_path = UPLOAD_FOLDER / pdf_name
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    debug_log(f"🗑️ Deleted from temp_uploads: {pdf_name}")
                except Exception as e:
                    debug_log(f"⚠️ Could not delete from temp_uploads {pdf_name}: {e}")

        # ✅ FORCE REFRESH after deletion

        # Get updated list
        pdf_list = get_pdf_list()

        return jsonify(
            {
                "success": True,
                "message": result_msg,
                "remaining_pdfs": len(pdf_list),
                "pdf_list": pdf_list,
                "updated_stats": {
                    "total_chunks": rag_metadata.get("total_chunks", 0),
                    "total_documents": rag_metadata.get("total_documents", 0),
                },
            }
        )

    except Exception as e:
        debug_log(f"❌ Delete PDF error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --------------------
# TEMPORARY OCR UPLOAD (in-memory only, lost on restart)
# --------------------

import tempfile, json, uuid as _uuid

_OCR_TEMP_FILE = Path(tempfile.gettempdir()) / "ocr_sessions.json"
_ocr_store_lock = threading.Lock()

# Point pytesseract at the Windows binary
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    pass

def _ocr_load() -> dict:
    try:
        if _OCR_TEMP_FILE.exists():
            return json.loads(_OCR_TEMP_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _ocr_save(data: dict):
    _OCR_TEMP_FILE.write_text(json.dumps(data), encoding="utf-8")

# Wipe any leftover sessions from a previous run
if _OCR_TEMP_FILE.exists():
    _OCR_TEMP_FILE.unlink()


def _run_ocr_on_file(filepath: str) -> str:
    """
    Extract text from a file.
    - Images: always OCR.
    - PDFs: try direct text extraction first; only OCR pages that come back empty
      (i.e. scanned/image-only pages).
    """
    import os
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    if ext != ".pdf":
        # Pure image — always OCR
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return "[ERROR] pytesseract / Pillow not installed. Run: pip install pytesseract pillow"
        return pytesseract.image_to_string(Image.open(filepath)).strip()

    # ---- PDF path ----
    page_texts = []

    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    page_texts.append(text)
                else:
                    # Page has no selectable text — OCR it
                    page_texts.append(_ocr_pdf_page(page))
    except ImportError:
        # Fallback: pypdf for text, then OCR blank pages
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[ERROR] Install pdfplumber or pypdf: pip install pdfplumber"

        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts.append(text)
            else:
                page_texts.append(_ocr_pdf_page_by_index(filepath, i))

    return "\n\n".join(page_texts).strip()


def _ocr_pdf_page(pdfplumber_page) -> str:
    """OCR a single pdfplumber page by rendering it to an image."""
    try:
        import pytesseract
        from PIL import Image
        img = pdfplumber_page.to_image(resolution=200).original
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        return f"[OCR ERROR on page] {e}"


def _ocr_pdf_page_by_index(filepath: str, page_index: int) -> str:
    """OCR a single PDF page by index using pdf2image (fallback when pdfplumber unavailable)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages = convert_from_path(filepath, dpi=200, first_page=page_index + 1, last_page=page_index + 1)
        if not pages:
            return ""
        return pytesseract.image_to_string(pages[0]).strip()
    except Exception as e:
        return f"[OCR ERROR on page {page_index}] {e}"


@app.route("/api/upload-ocr", methods=["POST", "OPTIONS"])
def upload_ocr():
    if request.method == "OPTIONS":
        return "", 204

    import os

    ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf", ".txt"}

    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files provided"}), 400

        session_id = request.form.get("session_id") or str(_uuid.uuid4())
        results = []

        for f in files:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED_EXTS:
                results.append({"filename": f.filename, "error": f"Unsupported type: {ext}", "text": None})
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name

            try:
                text = _run_ocr_on_file(tmp_path)
            except Exception as e:
                text = None
                results.append({"filename": f.filename, "error": str(e), "text": None, "chars": 0})
                continue
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            results.append({"filename": f.filename, "text": text, "chars": len(text), "error": None})

            with _ocr_store_lock:
                data = _ocr_load()
                data.setdefault(session_id, {})[f.filename] = text
                _ocr_save(data)

            debug_log(f"📷 OCR [{session_id}] {f.filename}: {len(text)} chars")

        total_chars = sum(r.get("chars") or 0 for r in results)
        return jsonify({"success": True, "session_id": session_id, "results": results, "total_chars": total_chars})

    except Exception as e:
        debug_log(f"❌ OCR upload error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-ocr/retrieve", methods=["POST", "OPTIONS"])
def retrieve_ocr():
    if request.method == "OPTIONS":
        return "", 204

    data_req = request.json or {}
    session_id = data_req.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    with _ocr_store_lock:
        store = _ocr_load()
        session = store.get(session_id)

    if session is None:
        return jsonify({"error": "Session not found (server may have restarted)"}), 404

    filename = data_req.get("filename")
    if filename:
        text = session.get(filename)
        if text is None:
            return jsonify({"error": f"File '{filename}' not found in session"}), 404
        return jsonify({"session_id": session_id, "filename": filename, "text": text})

    return jsonify({"session_id": session_id, "files": session})


@app.route("/api/upload-ocr/clear", methods=["POST", "OPTIONS"])
def clear_ocr_session():
    if request.method == "OPTIONS":
        return "", 204

    data_req = request.json or {}
    session_id = data_req.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    with _ocr_store_lock:
        store = _ocr_load()
        if session_id not in store:
            return jsonify({"error": "Session not found"}), 404
        del store[session_id]
        _ocr_save(store)

    debug_log(f"🗑️ Cleared OCR session: {session_id}")
    return jsonify({"success": True, "session_id": session_id})

@app.route("/api/models/list", methods=["GET"])
def list_models():
    """Get list of available models"""
    try:
        from yan import MODEL_MANAGER

        models = MODEL_MANAGER.get_available_models()
        current = MODEL_MANAGER.get_current_model_info()

        return jsonify(
            {
                "success": True,
                "models": models,
                "current": current,
                "count": len(models),
            }
        )

    except Exception as e:
        logging.error(f"Error listing models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model"""
    try:
        from yan import MODEL_MANAGER

        data = request.json
        model_key = data.get("model_key")
        n_ctx = data.get("n_ctx", 8000)
        n_gpu_layers = data.get("n_gpu_layers", 0)

        if not model_key:
            return jsonify({"success": False, "error": "No model_key provided"}), 400

        logging.info(f"Switching to model: {model_key}")

        # Load the model
        success = MODEL_MANAGER.load_model(
            model_key, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
        )

        if success:
            current_info = MODEL_MANAGER.get_current_model_info()

            return jsonify(
                {
                    "success": True,
                    "message": f'✅ Switched to {current_info["display_name"]}',
                    "current": current_info,
                }
            )
        else:
            return jsonify({"success": False, "error": "Failed to load model"}), 500

    except Exception as e:
        logging.error(f"Model switch error: {e}")
        import traceback

        traceback.print_exc()

        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/current", methods=["GET"])
def get_current_model():
    """Get currently loaded model info"""
    try:
        from yan import MODEL_MANAGER

        current = MODEL_MANAGER.get_current_model_info()

        return jsonify(
            {"success": True, "current": current, "has_model": current is not None}
        )

    except Exception as e:
        logging.error(f"Error getting current model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/scan", methods=["POST"])
def rescan_models():
    """Rescan models directory for new models"""
    try:
        from yan import MODEL_MANAGER

        MODEL_MANAGER._scan_models()
        models = MODEL_MANAGER.get_available_models()

        return jsonify(
            {
                "success": True,
                "message": f"Found {len(models)} models",
                "models": models,
            }
        )

    except Exception as e:
        logging.error(f"Error rescanning models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------
# ROOT ENDPOINT
# --------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "name": "Project Elixer API",
            "version": "2.1",
            "status": "running",
            "streaming": "enabled",
            "endpoints": [
                "/health",
                "/api/chat",
                "/api/chat/stream",
                "/api/upload-pdfs",
                "/api/pdf-status",
                "/api/rag-status",
                "/api/semantic-search",
            ],
        }
    )


import io
import tempfile
import os
from pathlib import Path


# ============================================================
# IMPROVED VOICE OUTPUT ENDPOINTS
# ============================================================
@app.route("/api/voice/synthesize", methods=["POST", "OPTIONS"])
def synthesize_voice():
    """
    Convert text to speech and return audio file
    Now with simplified Piper support (WAV output - no conversion needed!)
    """
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json()
        text = data.get("text", "")
        voice = data.get("voice", "southern_english_female")
        speed = data.get("speed", 1.0)
        engine = data.get("engine", "piper")

        debug_log(f"\n{'='*60}")
        debug_log(f"🔊 TTS REQUEST RECEIVED")
        debug_log(f"📝 Text: {text[:100]}...")
        debug_log(f"🎤 Engine: {engine}")
        debug_log(f"🗣️ Voice: {voice}")
        debug_log(f"⚡ Speed: {speed}")
        debug_log(f"{'='*60}\n")

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        # Limit text length
        if len(text) > 50000:
            text = text[:50000] + "... text truncated for voice output."

        # Try engines in order of preference
        audio_file = None
        errors = []

        # ✅ Try Piper first (recommended - fast and offline!)
        if engine == "piper":
            audio_file, error = synthesize_with_piper(text, voice, speed)
            if error:
                errors.append(f"Piper: {error}")

        # Fallback to other engines if Piper fails
        if not audio_file and engine == "gtts":
            audio_file, error = synthesize_with_gtts(text, speed)
            if error:
                errors.append(f"gTTS: {error}")

        elif not audio_file and engine == "edge-tts":
            audio_file, error = synthesize_with_edge_tts(text, voice, speed)
            if error:
                errors.append(f"Edge TTS: {error}")

        elif not audio_file and engine == "pyttsx3":
            audio_file, error = synthesize_with_pyttsx3(text, voice, speed)
            if error:
                errors.append(f"pyttsx3: {error}")

        # Final fallback: Try all engines
        if not audio_file:
            debug_log(f"⚠️ Primary engine failed, trying all fallbacks...")

            # Try Piper if not already tried
            if engine != "piper":
                audio_file, error = synthesize_with_piper(text, voice, speed)
                if error:
                    errors.append(f"Piper fallback: {error}")

            # Try gTTS
            if not audio_file:
                audio_file, error = synthesize_with_gtts(text, speed)
                if error:
                    errors.append(f"gTTS fallback: {error}")

            # Try pyttsx3 as last resort
            if not audio_file:
                audio_file, error = synthesize_with_pyttsx3(text, voice, speed)
                if error:
                    errors.append(f"pyttsx3 fallback: {error}")

        if audio_file and os.path.exists(audio_file):
            debug_log(f"✅ TTS successful: {audio_file}")

            try:
                # ✅ Determine MIME type based on file extension
                mime_type = "audio/wav" if audio_file.endswith(".wav") else "audio/mpeg"
                download_name = (
                    "speech.wav" if audio_file.endswith(".wav") else "speech.mp3"
                )

                debug_log(f"📤 Sending {mime_type} file: {download_name}")

                # Send file
                response = send_file(
                    audio_file,
                    mimetype=mime_type,
                    as_attachment=False,
                    download_name=download_name,
                )

                # Schedule cleanup after a delay
                import threading

                def delayed_cleanup():
                    import time

                    time.sleep(2)
                    try:
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                            debug_log(f"🗑️ Cleaned up temp file: {audio_file}")
                    except Exception as e:
                        debug_log(f"⚠️ Cleanup error: {e}")

                cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
                cleanup_thread.start()

                return response

            except Exception as e:
                # Cleanup on error
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                raise
        else:
            error_msg = "; ".join(errors)
            debug_log(f"❌ All TTS engines failed: {error_msg}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "All TTS engines failed",
                        "details": error_msg,
                    }
                ),
                500,
            )

    except Exception as e:
        debug_log(f"❌ Voice synthesis error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


'''
def synthesize_with_pyttsx3(text, voice, speed):
    """
    Fast, offline TTS using pyttsx3
    Returns: (audio_file_path, error_message)
    """
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', int(150 * speed))
        engine.setProperty('volume', 0.9)
        
        # Set voice if specified
        if voice and voice != 'default':
            voices = engine.getProperty('voices')
            for v in voices:
                if voice in v.id or voice.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Verify file exists and has content
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path, None
        else:
            return None, "Audio file was not created"
        
    except ImportError:
        return None, "pyttsx3 not installed. Install: pip install pyttsx3"
    except Exception as e:
        return None, str(e)


def synthesize_with_gtts(text, speed):
    """
    Google Text-to-Speech (requires internet)
    Returns: (audio_file_path, error_message)
    """
    try:
        from gtts import gTTS
        
        # Adjust speed
        slow = speed < 1.0
        
        tts = gTTS(text=text, lang='en', slow=slow)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio
        tts.save(temp_path)
        
        # Verify file exists and has content
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path, None
        else:
            return None, "Audio file was not created"
        
    except ImportError:
        return None, "gTTS not installed. Install: pip install gtts"
    except Exception as e:
        return None, str(e)


def synthesize_with_edge_tts(text, voice, speed):
    """
    Microsoft Edge TTS (high quality, requires internet)
    Returns: (audio_file_path, error_message)
    """
    try:
        import edge_tts
        import asyncio
        
        async def generate():
            try:
                # Use default voice if not specified
                voice_name = voice if voice and voice != 'default' else "en-US-AriaNeural"
                
                # Calculate rate
                rate_value = f"{int((speed - 1) * 100):+d}%"
                
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice_name,
                    rate=rate_value
                )
                
                # Create temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb')
                temp_path = temp_file.name
                temp_file.close()
                
                # Save audio
                await communicate.save(temp_path)
                
                return temp_path
            except Exception as e:
                debug_log(f"Edge TTS async error: {e}")
                return None
        
        # Run async function
        temp_path = asyncio.run(generate())
        
        # Verify file exists and has content
        if temp_path and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path, None
        else:
            return None, "Audio file was not created"
        
    except ImportError:
        return None, "edge-tts not installed. Install: pip install edge-tts"
    except Exception as e:
        return None, str(e)
'''


@app.route("/api/voice/config", methods=["GET", "OPTIONS"])
def get_voice_config():
    """
    Get available voices and TTS engines (including Piper)
    """
    if request.method == "OPTIONS":
        return "", 204

    try:
        config = {"engines": [], "voices": {}}

        # ✅ Check Piper (priority #1)
        piper_voices = get_piper_voices()
        if piper_voices and any(v["available"] for v in piper_voices):
            config["engines"].append("piper")
            config["voices"]["piper"] = piper_voices
            debug_log(f"✅ Piper available with {len(piper_voices)} voice(s)")
        else:
            debug_log("⚠️ Piper not available")

        # Check other engines...
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            config["engines"].append("pyttsx3")
            config["voices"]["pyttsx3"] = [
                {
                    "id": v.id,
                    "name": v.name.split(" - ")[0] if " - " in v.name else v.name,
                    "available": True,
                }
                for v in voices[:10]
            ]
            engine.stop()
            debug_log("✅ pyttsx3 available")
        except Exception as e:
            debug_log(f"⚠️ pyttsx3 not available: {e}")

        try:
            import gtts

            config["engines"].append("gtts")
            config["voices"]["gtts"] = [
                {"id": "default", "name": "Google TTS (English)", "available": True}
            ]
            debug_log("✅ gTTS available")
        except Exception as e:
            debug_log(f"⚠️ gTTS not available: {e}")

        try:
            import edge_tts

            config["engines"].append("edge-tts")
            config["voices"]["edge-tts"] = [
                {
                    "id": "en-US-AriaNeural",
                    "name": "Aria (US Female)",
                    "available": True,
                },
                {"id": "en-US-GuyNeural", "name": "Guy (US Male)", "available": True},
                {
                    "id": "en-GB-SoniaNeural",
                    "name": "Sonia (UK Female)",
                    "available": True,
                },
                {"id": "en-GB-RyanNeural", "name": "Ryan (UK Male)", "available": True},
            ]
            debug_log("✅ Edge TTS available")
        except Exception as e:
            debug_log(f"⚠️ Edge TTS not available: {e}")

        if not config["engines"]:
            debug_log("❌ No TTS engines available!")
            return jsonify({"success": False, "error": "No TTS engines available"}), 503

        return jsonify(
            {
                "success": True,
                "config": config,
                "default_engine": (
                    "piper" if "piper" in config["engines"] else config["engines"][0]
                ),
                "piper_status": {
                    "executable_found": PIPER_EXECUTABLE.exists(),
                    "voices_found": len(piper_voices),
                    "available_voices": [v for v in piper_voices if v["available"]],
                },
            }
        )

    except Exception as e:
        debug_log(f"❌ Voice config error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/test", methods=["POST", "OPTIONS"])
def test_voice():
    """
    Quick test endpoint for TTS
    """
    if request.method == "OPTIONS":
        return "", 204

    try:
        test_text = "Hello! This is a test of the voice output system."

        # Try each engine
        results = {}

        # Test pyttsx3
        audio, error = synthesize_with_pyttsx3(test_text, "default", 1.0)
        results["pyttsx3"] = {"available": audio is not None, "error": error}
        if audio and os.path.exists(audio):
            os.remove(audio)

        # Test gTTS
        audio, error = synthesize_with_gtts(test_text, 1.0)
        results["gtts"] = {"available": audio is not None, "error": error}
        if audio and os.path.exists(audio):
            os.remove(audio)

        # Test Edge TTS
        audio, error = synthesize_with_edge_tts(test_text, "default", 1.0)
        results["edge-tts"] = {"available": audio is not None, "error": error}
        if audio and os.path.exists(audio):
            os.remove(audio)

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/transcribe", methods=["POST", "OPTIONS"])
def transcribe_audio():
    """
    Transcribe audio using speech recognition (backend option)
    """
    if request.method == "OPTIONS":
        return "", 204

    try:
        # Check if audio file was uploaded
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        audio_file = request.files["audio"]

        # Save temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_file.save(temp_audio.name)
        temp_audio.close()

        # Try to transcribe using SpeechRecognition library
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio.name) as source:
                audio_data = recognizer.record(source)

            # Try Google Speech Recognition (free)
            try:
                text = recognizer.recognize_google(audio_data)
                os.unlink(temp_audio.name)

                return jsonify({"success": True, "text": text})
            except sr.UnknownValueError:
                os.unlink(temp_audio.name)
                return (
                    jsonify({"success": False, "error": "Could not understand audio"}),
                    400,
                )
            except sr.RequestError as e:
                os.unlink(temp_audio.name)
                return jsonify({"success": False, "error": f"API error: {str(e)}"}), 500

        except ImportError:
            os.unlink(temp_audio.name)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "SpeechRecognition library not installed",
                    }
                ),
                500,
            )

    except Exception as e:
        debug_log(f"❌ Transcription error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# --------------------
# PERSISTENT CHAT SESSIONS (cross-device)
# --------------------
SAVED_CHATS_FILE = CACHE_DIR / "saved_chats.json"

def _load_saved_chats():
    try:
        if SAVED_CHATS_FILE.exists():
            with open(SAVED_CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        debug_log(f"Error loading saved chats: {e}")
    return []

def _write_saved_chats(chats):
    try:
        SAVED_CHATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVED_CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False)
    except Exception as e:
        debug_log(f"Error writing saved chats: {e}")

@app.route("/api/chats", methods=["GET", "OPTIONS"])
def get_chats():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({"chats": _load_saved_chats()})

@app.route("/api/chats/save", methods=["POST", "OPTIONS"])
def save_chat():
    if request.method == "OPTIONS":
        return "", 204
    try:
        chat = request.json or {}
        if not chat.get("id"):
            return jsonify({"error": "Missing chat id"}), 400
        chats = _load_saved_chats()
        idx = next((i for i, c in enumerate(chats) if str(c["id"]) == str(chat["id"])), None)
        if idx is not None:
            chats[idx] = chat
        else:
            chats.append(chat)
        _write_saved_chats(chats)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chats/delete", methods=["POST", "OPTIONS"])
def delete_chat():
    if request.method == "OPTIONS":
        return "", 204
    try:
        data = request.json or {}
        chat_id = data.get("id")
        if not chat_id:
            return jsonify({"error": "Missing id"}), 400
        chats = _load_saved_chats()
        chats = [c for c in chats if str(c["id"]) != str(chat_id)]
        _write_saved_chats(chats)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/api/news/status", methods=["GET", "OPTIONS"])
def news_status():
    if request.method == "OPTIONS":
        return "", 204
    if not tech_feeder:
        return jsonify({"available": False, "message": "Tech news feeder not running"}), 200
    return jsonify({"available": True, **tech_feeder.status()})


@app.route("/api/news/refresh", methods=["POST", "OPTIONS"])
def news_refresh():
    if request.method == "OPTIONS":
        return "", 204
    if not tech_feeder:
        return jsonify({"success": False, "message": "Tech news feeder not running"}), 503
    try:
        count = tech_feeder.refresh_now()
        return jsonify({"success": True, "new_articles": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# --------------------
# ERROR HANDLERS
# --------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# --------------------
# STARTUP
# --------------------
def print_startup_info():
    """Print startup information"""
    print("\n" + "=" * 60)
    print("🚀 Project Elixer - Flask Server (Streaming Enabled)")
    print("=" * 60)
    print(f"✅ Server Status: Running")
    print(f"✅ LLM Loaded: {llm is not None}")
    print(f"✅ Embedding Model: {emb_model is not None}")
    print(f"✅ RAG Engine: {RAG_ENGINE is not None}")
    print(f"✅ Streaming: Enabled")
    print(f"📁 Upload Folder: {UPLOAD_FOLDER.absolute()}")
    print(f"🔗 Server URL: http://localhost:5000")
    print(f"🔗 Health Check: http://localhost:5000/health")
    print("📂 Templates folder:", app.template_folder)

    print("=" * 60)

    sources = persistent_data.get("sources") or []
    if sources:
        unique_sources = set(sources)
        print(f"📚 Loaded {len(unique_sources)} PDF(s) from cache")

    print("\n✨ Server is ready! Open index.html in your browser.\n")


# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    from werkzeug.serving import run_simple


    run_simple(
        hostname="0.0.0.0",
        port=5000,
        application=app,
        use_reloader=False,
        use_debugger=False,
        threaded=True,
    )


# print_startup_info()
