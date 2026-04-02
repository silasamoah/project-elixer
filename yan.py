extracted_text_from_llm = ""

import os
import torch
import numpy as np
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
from llama_cpp import Llama
import requests
from bs4 import BeautifulSoup
import random
from fpdf import FPDF
from datetime import datetime
import re
import json
import hashlib
from pathlib import Path
import shutil
import logging

from config import MODEL_PATH, PIPER_EXECUTABLE, PIPER_VOICES_DIR, PIPER_MODELS

from typing import List, Tuple, Optional
##
import os
import logging
import socket
from pathlib import Path


# ============================================================
# 1️⃣ AI-PERSONALITY IMPORT
# ============================================================
# ============================================================
# ENHANCED PERSONALITY SYSTEM V6
# ============================================================
# ============================================================
# 1️⃣ AI-PERSONALITY IMPORT - PRODUCTION VERSION
# ============================================================
from ai_personality_v6_final_production import (
    EnhancedPersonalityV6Final,
    Config as PersonalityConfig
)

# Initialize production personality system (if using memory mode)
# For Redis mode, see configuration below
personality_system = None

def init_personality_system():
    """Initialize personality system - call once at startup"""
    global personality_system
    if personality_system is None:
        # Configure for memory mode (no Redis required)
        import os
        os.environ["STORAGE_BACKEND"] = "memory"  # or "redis" if you have Redis
        os.environ["EVENT_STREAM_BACKEND"] = "memory"  # or "redis_streams"
        
        personality_system = EnhancedPersonalityV6Final()
    return personality_system


# ============================================================
# PERSONALITY PROCESSOR (Integrated with Production System)
# ============================================================
class EnhancedPersonality:
    """Enhanced personality system with production v6 integration"""
    
    def __init__(self):
        """Initialize with production personality system"""
        global personality_system
        if personality_system is None:
            init_personality_system()
        
        self.personality_system = personality_system
        # Add reference to profile manager for easy access
        self.profile_manager = personality_system.profile_manager
    
    def build_system_prompt(self, user_id: str) -> str:
        """
        Build dynamic system prompt based on personality.
        This method can be called directly or delegated to production system.
        """
        # Delegate to production system for consistency
        return self.personality_system.build_system_prompt(user_id)
    
    def process_response(self, user_id: str, user_input: str, base_response: str) -> str:
        """
        Process response with full personality adaptation including playfulness.
        
        Features:
        - Semantic analysis for warmth, verbosity, and playfulness
        - Context-aware playfulness gating (suppresses during distress)
        - Cheeky frequency tracking with auto-adjustment
        - Time-based decay support
        
        Args:
            user_id: User identifier
            user_input: User's input text
            base_response: Base response to return
        
        Returns:
            Base response (personality affects next response via system prompt)
        """
        try:
            # Get current profile (with optional decay on read)
            profile = self.profile_manager.get_profile(user_id, apply_decay=True)
            
            # Generate embedding for semantic analysis
            query_embedding = self.personality_system.embedding.encode(user_input)
            
            # Match against all prototypes
            similarities = self.personality_system.embedding.match_against_prototypes(
                query_embedding,
                self.personality_system.prototypes
            )
            
            # Initialize deltas
            delta_warmth = 0.0
            delta_verbosity = 0.0
            delta_playfulness = 0.0
            
            # ============================================================
            # WARMTH ADJUSTMENTS
            # ============================================================
            # Increase warmth on gratitude/appreciation
            if similarities.get("gratitude", 0) > 0.75:
                delta_warmth = 0.01
            
            # Decrease warmth on cold/formal language
            if similarities.get("formal", 0) > 0.75:
                delta_warmth = -0.005
            
            # ============================================================
            # VERBOSITY ADJUSTMENTS
            # ============================================================
            # Increase verbosity for detailed requests
            if similarities.get("detailed_request", 0) > 0.75:
                delta_verbosity = 0.02
            
            # Decrease verbosity for brief requests
            if similarities.get("brief_request", 0) > 0.75:
                delta_verbosity = -0.02
            
            # Increase verbosity for confusion (need more explanation)
            if similarities.get("confusion", 0) > 0.75:
                delta_verbosity = 0.015
            
            # ============================================================
            # PLAYFULNESS ADJUSTMENTS WITH SEMANTIC GATING
            # ============================================================
            playfulness_suppressed = False
            suppress_reason = None
            
            # STEP 1: Check for serious/distress contexts (SUPPRESS)
            serious_threshold = 0.6
            
            if similarities.get("distress", 0) > serious_threshold:
                playfulness_suppressed = True
                suppress_reason = "distress"
            elif similarities.get("serious_technical", 0) > serious_threshold:
                playfulness_suppressed = True
                suppress_reason = "serious_technical"
            elif similarities.get("grief_sadness", 0) > serious_threshold:
                playfulness_suppressed = True
                suppress_reason = "grief_sadness"
            elif similarities.get("emergency", 0) > serious_threshold:
                playfulness_suppressed = True
                suppress_reason = "emergency"
            
            if playfulness_suppressed:
                # Don't adjust playfulness, log suppression
                debug_log(f"🚫 Playfulness suppressed (reason: {suppress_reason})")
            else:
                # STEP 2: Check if overusing cheeky responses
                if profile.is_overusing_cheeky():
                    # Reduce playfulness to dial back
                    delta_playfulness = -0.03
                    debug_log(f"📉 Reducing playfulness (overuse: {profile.get_cheeky_count()}/5)")
                else:
                    # STEP 3: Check for positive feedback on humor
                    if similarities.get("humor_appreciated", 0) > 0.75:
                        delta_playfulness = 0.015
                        debug_log(f"😄 Increasing playfulness (humor appreciated)")
                    elif similarities.get("positive_feedback", 0) > 0.75:
                        delta_playfulness = 0.01
                        debug_log(f"👍 Increasing playfulness (positive feedback)")
            
            # ============================================================
            # UPDATE PROFILE ASYNCHRONOUSLY
            # ============================================================
            if delta_warmth != 0 or delta_verbosity != 0 or delta_playfulness != 0:
                self.profile_manager.update_profile_async(
                    user_id,
                    delta_warmth,
                    delta_verbosity,
                    delta_playfulness
                )
                
                debug_log(f"📊 Profile update queued:")
                debug_log(f"   Warmth: {profile.warmth:.3f} → {profile.warmth + delta_warmth:.3f}")
                debug_log(f"   Verbosity: {profile.verbosity:.3f} → {profile.verbosity + delta_verbosity:.3f}")
                debug_log(f"   Playfulness: {profile.playfulness:.3f} → {profile.playfulness + delta_playfulness:.3f}")
            
            # ============================================================
            # LOG SEMANTIC MATCHES
            # ============================================================
            if DEBUG_MODE:
                top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
                debug_log(f"🔍 Top semantic matches:")
                for category, score in top_matches:
                    if score > 0.3:  # Only show significant matches
                        debug_log(f"   {category}: {score:.3f}")
            
            return base_response
            
        except Exception as e:
            debug_log(f"❌ Error in process_response: {e}")
            import traceback
            debug_log(traceback.format_exc())
            return base_response
    
    def get_profile_stats(self, user_id: str) -> Dict:
        """Get profile statistics including playfulness"""
        profile = self.profile_manager.get_profile(user_id, apply_decay=True)
        
        return {
            "warmth": profile.warmth,
            "verbosity": profile.verbosity,
            "playfulness": profile.playfulness,
            "baseline_warmth": profile.baseline_warmth,
            "baseline_verbosity": profile.baseline_verbosity,
            "baseline_playfulness": profile.baseline_playfulness,
            "interactions": profile.interaction_count,
            "drift": abs(profile.warmth - profile.baseline_warmth),
            "cheeky_count": profile.get_cheeky_count(),
            "cheeky_limit": 5,
            "is_overusing": profile.is_overusing_cheeky(),
            "last_updated": profile.last_updated
        }
    
    # ============================================================
    # LEGACY COMPATIBILITY METHODS
    # ============================================================
    def compose_response(self, user_id: str, base: str = "") -> str:
        """Legacy compatibility - alias for process_response"""
        return self.process_response(user_id, "", base)
    
    def get_system_message(self, user_id: str = "default") -> str:
        """Legacy compatibility - alias for build_system_prompt"""
        return self.build_system_prompt(user_id)
    
    def remember(self, user, ai): 
        """Legacy compatibility - no-op"""
        pass
    
    def get_memory_summary(self): 
        """Legacy compatibility"""
        return ""
    
    def update_mood(self, mood): 
        """Legacy compatibility - no-op"""
        pass


# ============================================================
# INITIALIZE PERSONALITY SYSTEM
# ============================================================
try:
    personality = EnhancedPersonality()
    PERSONALITY_V6_AVAILABLE = True
    
    # Create compatibility functions
    def process(user_id: str, query: str, response: str = "") -> str:
        """Process response with personality adaptation"""
        return personality.process_response(user_id, query, response)
    
    def get_system_message(user_id: str = "default") -> str:
        """Get dynamic system message for user"""
        return personality.build_system_prompt(user_id)
    
    def get_phase_info(user_id: str = "default") -> Dict:
        """Get personality statistics"""
        return personality.get_profile_stats(user_id)
    
    print("✅ Enhanced Personality System V6 loaded successfully")
    print("   Features: Adaptive warmth • Dynamic verbosity • Profile persistence")
    
except Exception as e:
    PERSONALITY_V6_AVAILABLE = False
    print(f"❌ Failed to initialize Enhanced Personality V6: {e}")
    print("⚠️ Using fallback personality")
    
    # Fallback dummy personality
    class DummyPersonality:
        def compose_response(self, user_id, base=""): 
            return base
        def get_system_message(self, user_id="default"):
            return "You are a helpful companion."
        def remember(self, user, ai): pass
        def get_memory_summary(self): return ""
        def update_mood(self, mood): pass
        def get_personality_stats(self, user_id="default"): return {}
        def process_response(self, user_id, user_input, base): return base
    
    personality = DummyPersonality()
    
    def process(user_id, query, response=""): 
        return response
    def get_system_message(user_id="default"): 
        return "You are a helpful companion."
    def get_phase_info(user_id="default"): 
        return {"warmth": 0.7, "verbosity": 0.5, "interactions": 0}
# ============================================================
# 1️⃣ HARD OFFLINE FLAGS (must be set FIRST)
# ============================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"

# ============================================================
# 3️⃣ SILENCE LOGS
# ============================================================
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ============================================================
# 4️⃣ RESOLVE MODEL PATH (HF CACHE SNAPSHOT)
# ============================================================
HF_CACHE_ROOT = Path(
    "./models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
)

if not HF_CACHE_ROOT.exists():
    raise FileNotFoundError(
        f"HF cache snapshots directory not found: {HF_CACHE_ROOT}"
    )

snapshots = sorted(
    p for p in HF_CACHE_ROOT.iterdir() if p.is_dir()
)

if not snapshots:
    raise FileNotFoundError("No model snapshots found in HF cache")

MODEL_PATH = snapshots[0]  # usually only one snapshot

# ============================================================
# 5️⃣ LOAD MODEL
# ============================================================
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(str(MODEL_PATH), device="cpu")
model.max_seq_length = 256

# ============================================================
# 6️⃣ ENCODE FUNCTION (reusable)
# ============================================================
def encode_texts(texts: list[str], batch_size: int = 32):
    """Encode texts offline with performance tuning."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

##
logging.basicConfig(level=logging.DEBUG)

# yan.py
#from knowledge_collector_db import KnowledgeCollector, KnowledgeSourceType
from pathlib import Path

from integration_adapter import RAGIntegrationAdapter, migrate_legacy_data
from knowledge_rag_db import SourceType, ValidationStatus
# yan.py - Key changes for SQLite + FAISS integration
"""
Add these imports and modifications to your existing yan.py
"""

# ==================== IMPORTS FOR HYBRID RAG ====================
RAG_ENGINE = None
try:
    from database import db
    from hybrid_rag_manager import HybridRAGManager, migrate_from_legacy

    HYBRID_RAG_ENABLED = True
    print("✅ Hybrid RAG system available")
except ImportError as e:
    print(f"⚠️ Hybrid RAG not available: {e}")
    print("   System will use legacy JSON storage")
    db = None
    HybridRAGManager = None
    HYBRID_RAG_ENABLED = False

# ================= RAG ENHANCEMENT =================
from rag_enhanced import (
    AdvancedRAGEngine,
    Document,
    RetrievalResult,
    create_rag_engine,
    documents_from_chunks,
)

import threading
from typing import Optional
####
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"✅ GEMINI_API_KEY loaded: {api_key[:10]}...")
else:
    print("❌ GEMINI_API_KEY not found")

#===== Gemini Imports ========
try:
    from gemini_integration import create_gemini_llm, GeminiLLM
    GEMINI_AVAILABLE = True
    print("✅ Gemini integration available")
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiLLM = None
    print("⚠️ Gemini integration not available (install: pip install requests)")

# Gemini LLM instance
gemini_llm = None
USE_GEMINI = False  # Toggle for using Gemini

# ==================== AUTO-INITIALIZE GEMINI ====================
if GEMINI_AVAILABLE and api_key:
    try:
        print("🔄 Auto-initializing Gemini with API key...")
        gemini_llm = create_gemini_llm(
            api_key=api_key, 
            model="gemini-2.5-flash",
            enable_web_search=True
        )
        print("✅ Gemini initialized and ready to use!")
        print("   Use 'use gemini: <your query>' to route queries to Gemini")
    except Exception as e:
        print(f"⚠️ Failed to initialize Gemini: {e}")
        gemini_llm = None
else:
    if not GEMINI_AVAILABLE:
        print("⚠️ Gemini integration not available")
    elif not api_key:
        print("⚠️ GEMINI_API_KEY not set - Gemini not initialized")
        print("   Set GEMINI_API_KEY in your .env file to enable Gemini")




# ================================================================
# ============== GEMINI QUERY HELPER ==============
def query_gemini_with_search(query: str, context: str = "", max_tokens: int = 16000) -> str:
    """
    Query Gemini with automatic web search for current information
    
    Args:
        query: User's question
        context: Optional context from RAG/PDF
        max_tokens: Maximum response length (default 16000 for complete responses)
    
    Returns:
        Gemini's response text
    """

    # ====== PERSONALITY PROCESSING ======
    try:
        # Personality handled in final composition
        pass

        # Handle compliments immediately
        if personality_result["is_compliment"]:
            return personality_result["compliment_response"]

    except Exception as e:
        print(f"⚠️ Personality processing error: {e}")
        personality_result = {"opener": None, "suggestion": None}

    # ====== END PERSONALITY PROCESSING ======

    if not gemini_llm:
        raise Exception("Gemini not initialized. Set GEMINI_API_KEY in .env file")
    
    # Build prompt with context if provided
    if context:
        full_prompt = f"""Context information:
{context}

User question: {query}

Please provide a comprehensive and COMPLETE answer based on the context and current information, do not use long explanations. 
IMPORTANT: Finish your entire response - do not stop mid-sentence."""
    else:
        full_prompt = f"""{query}

Only answer using verified information.
If unsure, explicitly say:
"I do not have enough verified information."
Do NOT speculate."""
    
    try:
        # Call Gemini with increased token limit and adjusted parameters
        response = gemini_llm(
            full_prompt,
            max_tokens=max_tokens,  # Increased default
            temperature=0.7,
            top_p=0.95,  # Add top_p for better completion
            stop=None,  # Don't use stop sequences
            force_web_search=None  # Auto-detect if web search needed
        )
        
        result_text = response["choices"][0]["text"]
        
        # Log if response seems truncated
        if result_text and not result_text.rstrip().endswith(('.', '!', '?', '"', "'")):
            logging.warning(f"⚠️ Response may be truncated (doesn't end with punctuation)")
            logging.warning(f"   Last 50 chars: ...{result_text[-50:]}")
        
        # Log if web search was used
        if "grounding" in response:
            queries = response["grounding"].get("search_queries", [])
            if queries:
                logging.info(f"🌐 Gemini used web search: {queries}")

        # ====== ENHANCE RESPONSE WITH PERSONALITY ======
        try:
            if personality_result.get("opener"):
                result_text = personality_result["opener"] + result_text

            if personality_result.get("suggestion"):
                result_text = result_text + "\n\n" + personality_result["suggestion"]

            #result_text = enhance_response(result_text, query)

        except Exception as e:
            print(f"⚠️ Response enhancement error: {e}")

        # ====== END PERSONALITY ENHANCEMENT ======

        return result_text
        
    except Exception as e:
        logging.error(f"Gemini query failed: {e}")
        raise
# ================================================
# ============================================================
# QUERY DECOMPOSER - Add after imports, before pdf_tutor_enhanced
# ============================================================
def get_context_for_llm_with_filtering(
    self,
    query: str,
    top_k: int = 5,
    max_length: int = 2000,
    min_score: float = 0.4,  # 🔥 Minimum relevance score
):
    """
    Enhanced retrieval with strict relevance filtering
    
    🔥 FIXED: Uses self.rag_db.retrieve() instead of self.search()
    """
    try:
        # 🔥 FIX: Use correct method from RAGKnowledgeDatabase
        results = self.rag_db.retrieve(query, top_k=top_k * 3, method="hybrid")
        
        if not results:
            logging.debug(f"⚠️ No RAG results for query: {query}")
            return "", []
        
        # 🔥 CRITICAL FIX: Filter by minimum score
        filtered_results = [r for r in results if r.score >= min_score]
        
        if not filtered_results:
            logging.debug(f"⚠️ No results above threshold {min_score}")
            # Fallback: use only the top result if it's reasonably good
            if results[0].score >= 0.25:
                filtered_results = [results[0]]
                logging.debug(f"📌 Using top result only (score: {results[0].score:.3f})")
            else:
                return "", []
        
        # Sort by score and take top_k
        filtered_results = sorted(
            filtered_results, key=lambda x: x.score, reverse=True
        )[:top_k]
        
        # 🔥 NEW: Check source diversity
        sources = set(r.chunk.source_name for r in filtered_results)
        logging.info(
            f"✅ Retrieved {len(filtered_results)} chunks from {len(sources)} PDF(s)"
        )
        
        # Log what was retrieved
        for source in sources:
            source_results = [
                r for r in filtered_results if r.chunk.source_name == source
            ]
            avg_score = sum(r.score for r in source_results) / len(source_results)
            logging.info(
                f"  📄 {source}: {len(source_results)} chunks (avg score: {avg_score:.3f})"
            )
        
        # Build context
        context_parts = []
        for r in filtered_results:
            context_parts.append(
                f"[Source: {r.chunk.source_name}]\n{r.chunk.content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > max_length:
            context = context[:max_length] + "\n\n[Content truncated...]"
        
        return context, filtered_results
        
    except Exception as e:
        logging.error(f"❌ RAG retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

class WebSearchPermissionManager:
    """Manages runtime web search permissions with user confirmation"""
    
    def __init__(self):
        self._pending_request = None  # Stores query waiting for permission
        self._session_permissions = {}  # Track granted permissions per session
    
    def request_permission(self, query: str, reason: str) -> dict:
        """
        Create a permission request that needs user confirmation
        
        Returns:
            dict with request details to show user
        """
        self._pending_request = {
            "query": query,
            "reason": reason,
            "timestamp": time.time()
        }
        
        return {
            "needs_permission": True,
            "message": self._format_permission_request(reason),
            "query": query
        }
    
    def grant_permission(self) -> dict:
        """
        User granted permission - return the pending request
        """
        if not self._pending_request:
            return {"error": "No pending permission request"}
        
        request = self._pending_request
        self._pending_request = None
        
        # Store that permission was granted for this session
        self._session_permissions[request["query"]] = True
        
        return {
            "granted": True,
            "query": request["query"],
            "reason": request["reason"]
        }
    
    def deny_permission(self) -> dict:
        """
        User denied permission
        """
        if not self._pending_request:
            return {"error": "No pending permission request"}
        
        request = self._pending_request
        self._pending_request = None
        
        return {
            "denied": True,
            "query": request["query"]
        }
    
    def has_pending_request(self) -> bool:
        """Check if there's a pending permission request"""
        return self._pending_request is not None
    
    def clear_pending(self):
        """Clear any pending request"""
        self._pending_request = None
    
    def _format_permission_request(self, reason: str) -> str:
        """Format the permission request message"""
        return (
            f"🔍 **Permission Required**\n\n"
            f"{reason}\n\n"
            #f"Would you like me to search the web for current information?\n\n"
            f"Requesting permission to search the web for current information?\n\n"
            #f"**Reply with:**\n"
            #f"• 'yes' or 'permission granted' to allow\n"
            #f"• 'no' or 'deny' to skip web search"
        )


# Initialize global permission manager
WEB_PERMISSION_MANAGER = WebSearchPermissionManager()



class QueryDecomposer:
    """Break complex queries into sub-queries for better answers"""

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, query: str) -> List[str]:
        """Break complex query into simpler sub-queries"""

        # Check if decomposition is needed
        complexity_indicators = [
            "and",
            "also",
            "additionally",
            "furthermore",
            "compare",
            "difference between",
            "how does X relate to Y",
            "steps",
            "process",
            "explain both",
        ]

        if not any(ind in query.lower() for ind in complexity_indicators):
            return [query]  # Simple query, no decomposition

        decompose_prompt = f"""Break this complex question into 2-4 simpler sub-questions.
Each sub-question should be independently answerable.

Complex Question: {query}

Return ONLY the sub-questions, one per line, numbered:
1. [first sub-question]
2. [second sub-question]
etc.

Sub-questions:"""

        try:
            result = self.llm(
                decompose_prompt, max_tokens=200, temperature=0.3, stop=["\n\n", "---"]
            )
            text = result["choices"][0]["text"].strip()

            # Parse sub-queries
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            sub_queries = []

            for line in lines:
                # Remove numbering: "1. Question" -> "Question"
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if len(cleaned) > 10:  # Valid question
                    sub_queries.append(cleaned)

            if len(sub_queries) > 1:
                debug_log(f"🔀 Decomposed into {len(sub_queries)} sub-queries")
                return sub_queries

        except Exception as e:
            debug_log(f"Decomposition failed: {e}")

        return [query]  # Fallback to original

    def synthesize_answers(self, query: str, sub_answers: List[Tuple[str, str]]) -> str:
        """Combine sub-answers into coherent final answer"""

        synthesis_prompt = f"""Synthesize these sub-answers into ONE coherent answer.

Original Question: {query}

Sub-answers:
"""
        for i, (sub_q, sub_a) in enumerate(sub_answers, 1):
            synthesis_prompt += f"\n{i}. Q: {sub_q}\n   A: {sub_a}\n"

        synthesis_prompt += (
            "\n\nProvide a comprehensive answer that integrates all information:"
        )

        result = self.llm(synthesis_prompt, max_tokens=600, temperature=0.5)
        return result["choices"][0]["text"].strip()


# ============================================================
# CONFIDENCE CALIBRATOR
# ============================================================


class ConfidenceCalibrator:
    """Calibrate and explain confidence scores"""

    CONFIDENCE_FACTORS = {
        "source_quality": 0.3,  # PDF > Web > LLM
        "information_recency": 0.2,  # Recent > Old
        "source_agreement": 0.25,  # Multiple sources agree
        "query_clarity": 0.15,  # Clear query > Ambiguous
        "answer_completeness": 0.1,  # Complete > Partial
    }

    def calibrate(self, answer: str, metadata: dict) -> Tuple[float, dict]:
        """
        Calculate calibrated confidence with explanation

        Returns: (calibrated_score, explanation_dict)
        """
        factors = {}

        # Source quality
        if metadata.get("searchUsed"):
            factors["source_quality"] = 0.7  # Web search
        elif metadata.get("sources"):
            factors["source_quality"] = 0.9  # PDF
        else:
            factors["source_quality"] = 0.5  # Pure LLM

        # Recency (if web search was used)
        if metadata.get("searchUsed"):
            factors["information_recency"] = 0.8
        else:
            factors["information_recency"] = 0.6

        # Source agreement (multiple sources)
        num_sources = len(metadata.get("sources", []))
        if num_sources >= 3:
            factors["source_agreement"] = 0.9
        elif num_sources == 2:
            factors["source_agreement"] = 0.7
        else:
            factors["source_agreement"] = 0.5

        # Query clarity (detect uncertainty phrases)
        # Answer honesty — model hedging is a good signal, not a bad one
        hedging_phrases = [
            "i think", "maybe", "possibly", "might",
            "could be", "not sure", "unclear",
        ]
        model_is_hedging = any(phrase in answer.lower() for phrase in hedging_phrases)
        # If the model admits uncertainty, that's honest — reward it slightly
        factors["query_clarity"] = 0.75 if model_is_hedging else 0.8

        # Answer completeness (length & structure)
        answer_length = len(answer.split())
        if answer_length > 100:
            factors["answer_completeness"] = 0.9
        elif answer_length > 50:
            factors["answer_completeness"] = 0.7
        else:
            factors["answer_completeness"] = 0.5

        # Calculate weighted score
        calibrated_score = sum(factors[k] * self.CONFIDENCE_FACTORS[k] for k in factors)

        return calibrated_score, factors

    def should_add_uncertainty_note(self, score: float, factors: dict) -> Optional[str]:
        """Add uncertainty warning if needed"""

        if score < 0.4:
            return (
                "⚠️ **Low Confidence**: This answer may be incomplete or "
                "uncertain. Consider verifying with additional sources."
            )

        if factors.get("source_agreement", 1.0) < 0.6:
            return (
                "📚 **Note**: Limited source corroboration. "
                "Cross-reference recommended for critical information."
            )

        return None

def apply_hallucination_guard(
    answer: str,
    source_contexts: List[str],
    metadata: dict,
    query_source: str = "llm"
) -> str:
    """
    Run answer through AnswerVerifier + ConfidenceCalibrator and
    append warnings if the answer is poorly grounded or low-confidence.

    Args:
        answer:          The raw LLM answer string.
        source_contexts: List of source text chunks used to generate it.
        metadata:        Dict with keys: searchUsed (bool), sources (list).
        query_source:    Label for logging ("pdf", "web", "llm").

    Returns:
        Answer string, potentially with appended warning notes.
    """
    if not answer or not answer.strip():
        return answer

    result = answer

    # --- Step 1: Hard-cap hallucination — strip unsupported sentences ---
    if ANSWER_VERIFIER and source_contexts:
        try:
            is_grounded, grounding_score, unsupported = ANSWER_VERIFIER.verify_against_sources(
                answer, source_contexts, threshold=0.35
            )
            debug_log(f"🔍 Grounding check: {'✅' if is_grounded else '❌'} score={grounding_score:.2f}, unsupported={len(unsupported)}")

            if not is_grounded and unsupported:
                # Strip every unsupported sentence from the answer
                unsupported_set = set(unsupported)
                sentences = sent_tokenize(result)
                kept = [s for s in sentences if s not in unsupported_set]
                debug_log(f"✂️ Stripped {len(unsupported)} unsupported sentence(s) from answer")

                if kept:
                    result = " ".join(kept)
                else:
                    # Nothing survived — full block
                    result = (
                        "I don't have enough verified information to answer that confidently. "
                        "Could you provide more context or point me to a source?"
                    )
                    debug_log("🚫 Hallucination cap: entire answer was unsupported — blocked")
        except Exception as e:
            debug_log(f"⚠️ Verifier error: {e}")

    # --- Step 2: Hard-block on critically low confidence ---
    if CONFIDENCE_CALIBRATOR:
        try:
            score, factors = CONFIDENCE_CALIBRATOR.calibrate(answer, metadata)
            debug_log(f"📊 Confidence: {score:.2f} | source={query_source}")

            if score < 0.35:
                # Hard cap — confidence too low to trust any part of the answer
                debug_log(f"🚫 Hallucination cap: confidence {score:.2f} below 0.35 — blocking answer")
                result = (
                    "I'm not confident enough in this answer to give it to you. "
                    "My sources don't support a reliable response here. "
                    "Could you rephrase or give me more context?"
                )
            else:
                note = CONFIDENCE_CALIBRATOR.should_add_uncertainty_note(score, factors)
                if note:
                    result += f"\n\n{note}"
        except Exception as e:
            debug_log(f"⚠️ Calibrator error: {e}")

    return result

def update_web_search_settings(**kwargs):
    """Update web search configuration"""
    global WEB_SEARCH_CONFIG
    WEB_SEARCH_CONFIG.update(kwargs)
    print(f"✅ Updated web search settings: {kwargs}")
# ============================================================
# ANSWER VERIFIER
# ============================================================


class AnswerVerifier:
    """Verify LLM answers are grounded in provided sources"""

    def __init__(self, embedding_model):
        self.emb_model = embedding_model

    def verify_against_sources(
        self, answer: str, source_contexts: List[str], threshold: float = 0.5
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify answer statements are supported by sources

        Returns: (is_grounded, grounding_score, unsupported_claims)
        """

        # Extract key claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return True, 1.0, []  # No claims to verify

        # Check each claim against sources
        unsupported = []
        support_scores = []

        for claim in claims:
            # Find most similar source
            claim_emb = self.emb_model.encode(claim)

            max_similarity = 0.0
            for context in source_contexts:
                context_emb = self.emb_model.encode(context)
                similarity = cosine_sim(claim_emb, context_emb)
                max_similarity = max(max_similarity, similarity)

            support_scores.append(max_similarity)

            if max_similarity < threshold:
                unsupported.append(claim)

        avg_support = sum(support_scores) / len(support_scores)
        unsupported_ratio = len(unsupported) / len(claims)
        is_grounded = unsupported_ratio < 0.50  # allow up to 50% — LLM naturally adds connective tissue

        return is_grounded, avg_support, unsupported

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split by sentences
        sentences = sent_tokenize(text)

        # Filter out questions, greetings, meta-statements
        claims = []
        skip_patterns = [
            r"^(hi|hello|hey|thanks)",
            r"\?$",  # Questions
            r"^(i think|in my opinion|let me)",  # Subjective
        ]

        for sent in sentences:
            sent_lower = sent.lower().strip()

            if any(re.match(p, sent_lower) for p in skip_patterns):
                continue

            if len(sent.split()) >= 5:  # Substantial sentence
                claims.append(sent)

        return claims


# ============================================================
# CONTEXT COMPRESSOR
# ============================================================


class ContextCompressor:
    """Compress long conversations intelligently"""

    def __init__(self, llm, max_tokens=6000):
        self.llm = llm
        self.max_tokens = max_tokens

    def compress(self, chat_history: List[dict], current_query: str) -> List[dict]:
        """Compress history to fit context window"""

        if len(chat_history) < 8:
            return chat_history  # No compression needed

        # Always keep: first message + last 4 exchanges
        compressed = [chat_history[0]]  # System message
        middle = chat_history[1:-8]
        recent = chat_history[-8:]

        if len(middle) > 0:
            # Summarize middle section
            summary = self._summarize_section(middle, current_query)
            compressed.append(
                {
                    "role": "assistant",
                    "content": f"📋 **Previous Discussion Summary**:\n{summary}",
                }
            )

        compressed.extend(recent)
        return compressed

    def _summarize_section(self, messages: List[dict], current_query: str) -> str:
        """Summarize a section of conversation"""

        # Format messages
        conversation_text = "\n".join(
            [
                f"{m.get('role', 'user')}: {m.get('content', m.get('user', m.get('ai', '')))[:200]}"
                for m in messages
            ]
        )

        summary_prompt = f"""Summarize this conversation in 3-4 concise sentences.
Focus on: main topics discussed, key facts learned, user preferences/goals.

Current question for context: {current_query}

Conversation:
{conversation_text}

Summary:"""

        result = self.llm(summary_prompt, max_tokens=150, temperature=0.3)
        return result["choices"][0]["text"].strip()


class WebSearchFirewall:
    """
    Centralized firewall to completely block web search when disabled.
    Thread-safe and prevents ANY web access when turned off.
    """

    def __init__(self):
        self._enabled = True
        self._lock = threading.Lock()
        self._request_count = 0
        self._blocked_count = 0

    def enable(self):
        """Enable web search"""
        with self._lock:
            self._enabled = True
            debug_log("🌐 Web search ENABLED")

    def disable(self):
        """Disable web search"""
        with self._lock:
            self._enabled = False
            debug_log("🚫 Web search DISABLED - firewall active")

    def is_enabled(self) -> bool:
        """Check if web search is allowed"""
        with self._lock:
            return self._enabled

    def check_permission(self) -> tuple[bool, Optional[str]]:
        """
        Check if web search is permitted.
        Returns (allowed: bool, error_message: Optional[str])
        """
        with self._lock:
            if not self._enabled:
                self._blocked_count += 1
                error_msg = (
                    "🚫 Web search is disabled. "
                    "Enable 'Allow Web Search' in settings to use this feature."
                )
                debug_log(f"🚫 Blocked web search request (#{self._blocked_count})")
                return False, error_msg

            self._request_count += 1
            debug_log(f"✅ Approved web search request (#{self._request_count})")
            return True, None

    def get_stats(self) -> dict:
        """Get firewall statistics"""
        with self._lock:
            return {
                "enabled": self._enabled,
                "total_requests": self._request_count,
                "blocked_requests": self._blocked_count,
                "allowed_requests": self._request_count,
            }


# ============================================================
# GLOBAL FIREWALL INSTANCE
# ============================================================

WEB_FIREWALL = WebSearchFirewall()


# ============================================================
# PROTECTED WEB SEARCH WRAPPER
# ============================================================
SearchResult = Tuple[str, str, str] 

def run_web_search_protected(
    query: str,
    force_refresh: bool = False,
    debug: bool = False
) -> Tuple[str, List[SearchResult]]:
    """
    Protected web search with firewall checking.
    
    Args:
        query: Search query
        force_refresh: Skip cache
        debug: Enable debug logging
    
    Returns:
        Tuple of (context_string, results_list) or (error_message, [])
    """
    # Check firewall permission
    allowed, error_msg = WEB_FIREWALL.check_permission()
    
    if not allowed:
        debug_log(f"🚫 FIREWALL BLOCKED: {query}")
        return error_msg, []
    
    # Proceed with search
    try:
        return run_web_search(query, force_refresh, debug)
    except Exception as e:
        debug_log(f"❌ Web search error: {e}")
        return f"Web search error: {str(e)}", []

#########################
def fix_concatenated_lists(text: str) -> str:
    """
    Fix concatenated numbered lists that appear on the same line.
    Converts: "1. Text 2. More text" into separate lines.

    PROTECTS DATES and NUMBERED CONTENT from being treated as list items.
    """
    if not text:
        return text

    # ✅ CRITICAL FIX: Protect dates FIRST (before any list processing)
    # Match patterns like "January 2025." or "December 28, 2025."
    # Replace period after year with a placeholder
    DATE_PLACEHOLDER = "<<<YEAR_PERIOD>>>"

    # Protect "Month YYYY." patterns
    text = re.sub(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\.",
        r"\1 \2" + DATE_PLACEHOLDER,
        text,
    )

    # Protect "Month DD, YYYY." patterns
    text = re.sub(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})\.",
        lambda m: m.group(0)[:-1] + DATE_PLACEHOLDER,
        text,
    )

    # Protect standalone years like "2025." when not part of a list
    text = re.sub(
        r"([a-zA-Z,])\s+(\d{4})\.",  # Letter/comma followed by year with period
        r"\1 \2" + DATE_PLACEHOLDER,
        text,
    )
    
    # 🔥 NEW: Protect content within numbered list items from being split
    # Pattern: "shift of 3." should NOT trigger a new list item
    # Only split when we have: [sentence end] + [whitespace] + [number][period][space][capital letter]
    
    # Method: Only create new lines when we have COMPLETE sentence endings
    # followed by a new numbered item with a capital letter
    text = re.sub(
        r"([.!?])\s+(\d+\.\s+[A-Z])",  # Full sentence end + numbered item with capital
        r"\1\n\n\2",
        text
    )
    
    # Also handle cases where items are directly concatenated: "text.3. Next item"
    text = re.sub(
        r"([a-zA-Z\)])\.(\d+\.\s+[A-Z])",  # Letter/paren + period + numbered item
        r"\1.\n\n\2",
        text
    )

    # Split lines with multiple numbered items (only if they start with capitals)
    lines = []
    for line in text.split("\n"):
        # Only split if we find multiple numbered items that look like list starts
        # (number + period + space + capital letter)
        if re.search(r"\d+\.\s+[A-Z].+?\s+\d+\.\s+[A-Z]", line):
            # Split on numbered items with capitals
            parts = re.split(r"(?=\d+\.\s+[A-Z])", line)
            for part in parts:
                part = part.strip()
                if part and re.match(r"^\d+\.\s+[A-Z]", part):
                    lines.append(part)
        else:
            lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ✅ RESTORE DATES: Replace placeholders back with periods
    text = text.replace(DATE_PLACEHOLDER, ".")

    return text


def format_numbered_lists(text: str) -> str:
    """
    Ensure numbered lists have proper format with consistent spacing.
    """
    if not text:
        return text

    lines = text.split("\n")
    formatted = []

    for line in lines:
        stripped = line.strip()

        # Match numbered list: "1.", "2.", etc.
        match = re.match(r"^(\d+)\.\s*(.*)", stripped)
        if match:
            num, content = match.groups()
            # Ensure single space after number
            formatted.append(f"{num}. {content.strip()}")
        else:
            formatted.append(line)

    return "\n".join(formatted)


def format_llm_response(text: str) -> str:
    """
    Complete formatting for LLM responses.

    **Call this function IMMEDIATELY after getting LLM output.**

    Usage:
        answer = llm(prompt, ...)["choices"][0]["text"]
        answer = format_llm_response(answer)  # ⚡ ALWAYS DO THIS
    """
    if not text or len(text.strip()) < 3:
        return text

    # Step 1: Fix concatenated lists (MOST IMPORTANT)
    text = fix_concatenated_lists(text)

    # Step 2: Ensure numbered lists have proper format
    text = format_numbered_lists(text)

    # Step 3: Remove excessive blank lines (final cleanup)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_llm_answer(llm_out: dict, is_code: bool = False) -> str:
    """
    Extract and format LLM answer with proper numbered list handling.
    🔥 ENHANCED: Better error handling and logging
    """
    try:
        # Debug what we received
        debug_log(f"🔍 Processing LLM output (type: {type(llm_out)})")
        
        if llm_out is None:
            debug_log("❌ llm_out is None")
            return "Sorry, I encountered an error generating a response."
        
        if not isinstance(llm_out, dict):
            debug_log(f"❌ llm_out is not a dict: {type(llm_out)}")
            return str(llm_out) if llm_out else "Error: Invalid response format"
        
        # ✅ STEP 1: Extract the raw text from LLM output
        if 'choices' not in llm_out:
            debug_log(f"❌ No 'choices' in llm_out. Keys: {llm_out.keys()}")
            return "Error: Unexpected response format"
        
        if not llm_out['choices']:
            debug_log("❌ choices list is empty")
            return "Sorry, the model didn't generate a response."
        
        first_choice = llm_out["choices"][0]
        debug_log(f"🔍 First choice keys: {first_choice.keys()}")
        
        # Try to extract text (handle both 'text' and 'message' formats)
        answer = None
        if "text" in first_choice:
            answer = first_choice["text"].strip()
        elif "message" in first_choice and isinstance(first_choice["message"], dict):
            answer = first_choice["message"].get("content", "").strip()
        
        if not answer:
            debug_log(f"❌ Could not extract text from choice: {first_choice}")
            return "Sorry, I couldn't generate a proper response."
        
        debug_log(f"✅ Extracted text: {len(answer)} chars")

        # ✅ STEP 2: Apply formatting fixes
        answer = format_llm_response(answer)
        debug_log(f"✅ After formatting: {len(answer)} chars")

        # ✅ STEP 3: Clean response ONLY if not code AND not a numbered list
        if not is_code:
            has_numbered_list = bool(re.search(r"\d+\.\s+", answer))
            if not has_numbered_list:
                answer = clean_response_safe(answer)
                debug_log(f"✅ After cleaning: {len(answer)} chars")

        if not answer or len(answer.strip()) < 3:
            debug_log("❌ Final answer is too short or empty")
            return "I generated a response but it seems incomplete."

        debug_log(f"✅ Final answer: {len(answer)} chars")
        return answer

    except (KeyError, IndexError, TypeError, AttributeError) as e:
        debug_log(f"❌ Error processing LLM output: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        return "Sorry, I encountered an error processing the response."
    except Exception as e:
        debug_log(f"❌ Unexpected error: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        return "Sorry, I encountered an unexpected error."

def clean_response_safe(text):
    """
    FIXED VERSION - doesn't break on legitimate content
    Safer cleaning that preserves complete responses
    """
    if not text:
        return text

    original_text = text

    # Remove leaked prompt artifacts ONLY
    bad_patterns = [
        r"^Response:\s*",  # Only if at start of line
        r"^User:\s*",
        r"^Assistant:\s*",
        r"^Human:\s*",
    ]

    for pattern in bad_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Clean lines more carefully
    lines = text.split("\n")
    clean = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Always preserve blank lines
        if not line_stripped:
            clean.append(line)
            continue

        # Always preserve numbered list items
        is_numbered_item = bool(re.match(r"^\d+\.\s+", line_stripped))
        if is_numbered_item:
            clean.append(line)
            continue

        # Only BREAK on clear prompt leakage at START of line
        prompt_leakage_starts = [
            "q:",
            "question:",
            "example:",
            "exercise:",
            "response:",
            "user:",
            "assistant:",
            "human:",
        ]
        
        if line_stripped.lower().startswith(tuple(prompt_leakage_starts)):
            break  # This is actual prompt leakage - stop here

        # SKIP (not break) closing remarks ONLY at very end
        is_last_or_near_end = i >= len(lines) - 3
        
        if is_last_or_near_end:
            closing_phrases = [
                "would you like",
                "do you want me to",
                "shall i continue",
                "can i help with",
                "let me know if",
            ]
            
            if any(phrase in line_stripped.lower() for phrase in closing_phrases):
                continue  # Skip but keep processing

        # Keep the line
        clean.append(line)

    result = "\n".join(clean).strip()

    # Fallback to original if cleaning destroyed content
    if not result or len(result) < 10:
        return original_text.strip()

    return result


# ----------------------------
# NLTK setup
# ----------------------------
import nltk
#nltk.download("stopwords", quiet=True)
#nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----------------------------
# Cache & Storage Setup
# ----------------------------
CACHE_DIR = Path("./chat_cache")
CACHE_DIR.mkdir(exist_ok=True)
PDF_STORAGE_DIR = CACHE_DIR / "pdfs"
PDF_STORAGE_DIR.mkdir(exist_ok=True)
SEARCH_CACHE_FILE = CACHE_DIR / "search_cache.json"
CHAT_HISTORY_FILE = CACHE_DIR / "chat_history.json"
PDF_INDEX_FILE = CACHE_DIR / "pdf_index.json"
PDF_METADATA_FILE = CACHE_DIR / "pdf_metadata.json"
DEBUG_LOG_FILE = CACHE_DIR / "debug_log.txt"

# Initialize after cache setup
#KNOWLEDGE_COLLECTOR = KnowledgeCollector(
#    cache_dir=CACHE_DIR / "knowledge", cache_expiry_days=7, confidence_threshold=0.3
#)
# ============================================================
# COMPLETE USER INFO STORAGE SYSTEM
# ============================================================

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ============================================================
# USER DATA STORAGE PATHS
# ============================================================

USER_DATA_DIR = CACHE_DIR / "user_data"
USER_DATA_DIR.mkdir(exist_ok=True)
USER_PROFILES_FILE = USER_DATA_DIR / "user_profiles.json"
CURRENT_USER_FILE = USER_DATA_DIR / "current_user.json"

# ============================================================
# USER PROFILE STRUCTURE
# ============================================================
DEFAULT_USER_PROFILE = {
    "user_id": None,
    "name": None,
    "email": None,
    "preferences": {
        "response_style": "balanced",  # 'concise', 'balanced', 'detailed'
        "code_style": "commented",  # 'minimal', 'commented', 'verbose'
        "expertise_level": "intermediate",  # 'beginner', 'intermediate', 'advanced'
        "preferred_language": "python",
        "use_emojis": True,
        "emoji_mode": "playful",  # NEW: 'formal', 'balanced', 'playful', or None for auto-detect
        "stream_responses": True,
    },
    "interests": [],
    "learning_goals": [],
    "skill_assessments": {},
    "interaction_stats": {
        "total_questions": 0,
        "topics_discussed": {},
        "favorite_topics": [],
        "last_active": None,
        "first_seen": None,
        "session_count": 0,
    },
    # "chat_history": [],  # Removed - stored in chat_history.json instead
    "bookmarks": [],
    "notes": [],
    "custom_settings": {},
}

# ============================================================
# RAG METADATA TRACKING
# ============================================================

rag_metadata = {
    "total_chunks": 0,
    "total_documents": 0,
    "last_updated": None,
}






# ----------------------------
# Imports
# ----------------------------
from datetime import datetime
from pathlib import Path


# ----------------------------
# Debug config (FIRST)
# ----------------------------
DEBUG_MODE = True
DEBUG_LOG_FILE = Path("debug_log.txt")


def debug_log(message):
    """Log debug information to console and file."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        try:
            with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")
        except:
            pass


# yan.py - Updated UserManager with thread safety

import threading
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path


class UserManager:
    """Thread-safe user profile management"""

    def __init__(self):
        self._lock = threading.RLock()
        self._file_lock = threading.Lock()
        self.current_user = None
        self.all_users = self._load_all_users()
        self._load_current_user()

    def _load_all_users(self) -> Dict[str, Dict]:
        """Thread-safe user profile loading"""
        with self._file_lock:
            if USER_PROFILES_FILE.exists():
                try:
                    with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        debug_log(f"✅ Loaded {len(data)} user profiles")
                        return data
                except json.JSONDecodeError as e:
                    debug_log(f"⚠️ Corrupted user profiles file: {e}")
                    # Backup corrupted file
                    backup_path = USER_PROFILES_FILE.with_suffix(".json.backup")
                    shutil.copy(USER_PROFILES_FILE, backup_path)
                    debug_log(f"📦 Backed up to {backup_path}")
                    return {}
                except Exception as e:
                    debug_log(f"❌ Error loading user profiles: {e}")
                    return {}
            return {}

    def _save_all_users(self) -> bool:
        """Thread-safe user profile saving with atomic writes"""
        with self._file_lock:
            try:
                # Write to temporary file first (atomic write pattern)
                temp_file = USER_PROFILES_FILE.with_suffix(".json.tmp")

                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.all_users, f, indent=2, ensure_ascii=False)

                # Atomic rename (replaces old file)
                temp_file.replace(USER_PROFILES_FILE)

                debug_log(f"💾 Saved {len(self.all_users)} user profiles")
                return True

            except Exception as e:
                debug_log(f"❌ Error saving user profiles: {e}")
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()
                return False

    def _load_current_user(self):
        """Load the currently active user"""
        with self._lock:
            if CURRENT_USER_FILE.exists():
                try:
                    with open(CURRENT_USER_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        user_id = data.get("user_id")
                        if user_id and user_id in self.all_users:
                            self.current_user = user_id
                            debug_log(f"✅ Loaded current user: {user_id}")
                except Exception as e:
                    debug_log(f"⚠️ Error loading current user: {e}")

    def _save_current_user(self):
        """Save the currently active user"""
        with self._file_lock:
            try:
                temp_file = CURRENT_USER_FILE.with_suffix(".json.tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump({"user_id": self.current_user}, f)
                temp_file.replace(CURRENT_USER_FILE)
            except Exception as e:
                debug_log(f"❌ Error saving current user: {e}")

    def generate_user_id(self, seed: str = None) -> str:
        """
        Generate a stable, filesystem-safe user ID
        """
        import hashlib
        import uuid

        if seed:
            normalized = seed.strip().lower()
            return hashlib.sha256(normalized.encode()).hexdigest()[:12]

        # Fallback for anonymous users
        return f"user_{uuid.uuid4().hex[:12]}"

    def create_user(
        self, name: str = None, email: str = None, identifier: str = None
    ) -> str:
        """Thread-safe user creation"""
        with self._lock:
            user_id = self.generate_user_id(identifier or name or email)

            if user_id in self.all_users:
                debug_log(f"ℹ️ User {user_id} already exists")
                return user_id

            profile = DEFAULT_USER_PROFILE.copy()
            profile["user_id"] = user_id
            profile["name"] = name
            profile["email"] = email
            profile["interaction_stats"]["first_seen"] = datetime.now().isoformat()
            profile["interaction_stats"]["last_active"] = datetime.now().isoformat()

            self.all_users[user_id] = profile
            success = self._save_all_users()

            if success:
                debug_log(f"✅ Created new user: {user_id} ({name or 'Anonymous'})")
            else:
                debug_log(f"⚠️ User created but save failed: {user_id}")

            return user_id

    def set_current_user(self, user_id: str):
        """Thread-safe user switching"""
        with self._lock:
            if user_id not in self.all_users:
                debug_log(f"⚠️ User {user_id} not found, creating...")
                self.create_user(identifier=user_id)

            self.current_user = user_id
            self._save_current_user()

            # Update last active
            self.update_user_stat("last_active", datetime.now().isoformat())
            session_count = self.get_user_data("interaction_stats.session_count", 0)
            self.update_user_stat("session_count", session_count + 1)

            debug_log(f"✅ Switched to user: {user_id}")

    def get_current_user(self):
        """Get current user (JSON-based UserManager only)"""
        with self._lock:
            if not self.current_user:
                return None
            return self.all_users.get(self.current_user)

    def update_user_stat(self, stat_name: str, value):
        """Update user stat (JSON-based UserManager only)"""
        with self._lock:
            if not self.current_user:
                return False

            stats = self.get_user_data("interaction_stats", {})
            stats[stat_name] = value
            self.set_user_data("interaction_stats", stats)
            return True

    def get_user_data(self, key_path: str, default=None) -> Any:
        """Thread-safe nested data retrieval"""
        with self._lock:
            user = self.all_users.get(self.current_user)
            if not user:
                return default

            keys = key_path.split(".")
            value = user

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return value

    def set_user_data(self, key_path: str, value: Any) -> bool:
        """Thread-safe nested data setting"""
        with self._lock:
            if not self.current_user:
                debug_log("⚠️ No current user set")
                return False

            if self.current_user not in self.all_users:
                debug_log(f"⚠️ Current user {self.current_user} not in all_users")
                return False

            keys = key_path.split(".")
            user = self.all_users[self.current_user]

            # Navigate to parent dict
            current = user
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

            # Save to disk
            success = self._save_all_users()

            if success:
                #debug_log(f"✅ Updated {key_path} = {value}")
                debug_log(f"✅ Updated {key_path}")
            else:
                debug_log(f"⚠️ Update succeeded but save failed: {key_path}")

            return success

    def add_to_chat_history(self, user_msg: str, ai_msg: str) -> bool:
        """Update interaction stats only - chat history saved separately to chat_history.json"""
        debug_log(f"📥 add_to_chat_history called (stats only)")
        
        with self._lock:
            if not self.current_user:
                debug_log("❌ current_user is None")
                return False
            
            if self.current_user not in self.all_users:
                debug_log(f"❌ current_user '{self.current_user}' not in all_users!")
                return False

            # Only update interaction count - chat history handled separately
            count = self.get_user_data("interaction_stats.total_questions", 0)
            update_result = self.update_user_stat("total_questions", count + 1)
            
            debug_log(f"✅ Updated interaction stats")
            return update_result

    
    def reload_from_disk(self) -> bool:
        """Force reload from disk (for sync across processes)"""
        with self._lock:
            try:
                old_current = self.current_user
                self.all_users = self._load_all_users()

                # Restore current user if it still exists
                if old_current and old_current in self.all_users:
                    self.current_user = old_current
                elif self.all_users:
                    # Fall back to first available user
                    self.current_user = next(iter(self.all_users.keys()))
                else:
                    self.current_user = None

                debug_log("✅ Reloaded user data from disk")
                return True
            except Exception as e:
                debug_log(f"❌ Failed to reload from disk: {e}")
                return False

    # ... (keep other methods like track_topic, add_interest, etc.)

    def track_topic(self, topic: str):
        """Track discussed topics"""
        topics = self.get_user_data("interaction_stats.topics_discussed", {})
        topics[topic] = topics.get(topic, 0) + 1
        self.set_user_data("interaction_stats.topics_discussed", topics)

        # Update favorite topics (top 5)
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        favorites = [t[0] for t in sorted_topics[:5]]
        self.set_user_data("interaction_stats.favorite_topics", favorites)

    def add_interest(self, interest: str):
        """Add a user interest"""
        interests = self.get_user_data("interests", [])
        if interest not in interests:
            interests.append(interest)
            self.set_user_data("interests", interests)

    def add_learning_goal(self, goal: str):
        """Add a learning goal"""
        goals = self.get_user_data("learning_goals", [])
        if goal not in goals:
            goals.append(goal)
            self.set_user_data("learning_goals", goals)

    def add_bookmark(self, content: str, title: str = None):
        """Bookmark important information"""
        bookmarks = self.get_user_data("bookmarks", [])
        bookmarks.append(
            {
                "content": content,
                "title": title or content[:50],
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.set_user_data("bookmarks", bookmarks)

    def get_user_summary(self) -> str:
        """Get a formatted summary of user info"""
        user = self.get_current_user()
        if not user:
            return "No user profile active"

        name = user.get("name", "Anonymous")
        stats = user.get("interaction_stats", {})
        prefs = user.get("preferences", {})

        summary = f"**User Profile: {name}**\n\n"
        summary += f"📊 **Statistics:**\n"
        summary += f"  • Total questions: {stats.get('total_questions', 0)}\n"
        summary += f"  • Sessions: {stats.get('session_count', 0)}\n"
        summary += f"  • Member since: {stats.get('first_seen', 'Unknown')[:10]}\n\n"

        if stats.get("favorite_topics"):
            summary += (
                f"🎯 **Favorite topics:** {', '.join(stats['favorite_topics'][:3])}\n\n"
            )

        summary += f"⚙️ **Preferences:**\n"
        summary += f"  • Response style: {prefs.get('response_style', 'balanced')}\n"
        summary += f"  • Expertise: {prefs.get('expertise_level', 'intermediate')}\n"

        return summary

# ============================================================
# USER MANAGER CLASS
# ============================================================
def start_user_session():
    user = user_manager.get_current_user()
    if not user:
        return

    count = user_manager.get_user_data("interaction_stats.session_count", 0)

    user_manager.update_user_stat("session_count", count + 1)
    user_manager.update_user_stat("last_active", datetime.now().isoformat())
# ============================================================
# GLOBAL USER MANAGER INSTANCE
# ============================================================

user_manager = UserManager()

# CRITICAL: Ensure a user exists for saving chat history
if not user_manager.current_user:
    debug_log("⚠️ No user profile found - creating default user")
    user_manager.set_current_user("default_user")
    debug_log(f"✅ Set current user to: {user_manager.current_user}")
else:
    debug_log(f"✅ Current user loaded: {user_manager.current_user}")

start_user_session()


# ============================================================
# INTEGRATION FUNCTIONS
# ============================================================


def extract_user_info_from_message(message: str):
    """
    Extract and store user information from messages.
    Detects: names, interests, learning goals, preferences
    """
    msg = message.lower()

    # Extract name
    name_patterns = [
        r"my name is (\w+)",
        r"call me (\w+)",
    ]

    for pattern in name_patterns:
        import re

        match = re.search(pattern, msg)
        if match:
            name = match.group(1).capitalize()
            user_manager.set_user_data("name", name)
            debug_log(f"Extracted user name: {name}")
            break

    # Extract interests
    interest_keywords = {
        "python": "Python programming",
        "javascript": "JavaScript",
        "machine learning": "Machine Learning",
        "web development": "Web Development",
        "data science": "Data Science",
    }

    for keyword, interest in interest_keywords.items():
        if keyword in msg and (
            "interested in" in msg or "learning" in msg or "want to" in msg
        ):
            user_manager.add_interest(interest)
            debug_log(f"Added interest: {interest}")

    # Extract learning goals
    if "want to learn" in msg or "trying to learn" in msg or "goal is to" in msg:
        user_manager.add_learning_goal(message)
        debug_log("Added learning goal")

    # Extract preference changes
    if "be concise" in msg or "short answers" in msg:
        user_manager.set_user_data("preferences.response_style", "concise")
    elif "detailed" in msg or "explain more" in msg:
        user_manager.set_user_data("preferences.response_style", "detailed")


def build_user_context_for_prompt() -> str:
    """
    Build context string from user profile for LLM prompts.
    ONLY includes name when it's contextually relevant (personal questions, identity).
    """
    user = user_manager.get_current_user()
    if not user:
        return ""

    context_parts = []

    # DON'T include name in every prompt - only add it for identity/greeting contexts
    # The name will be available in the user object but not forced into every response

    # Preferences
    prefs = user.get("preferences", {})
    if prefs.get("response_style"):
        style_map = {
            "concise": "Keep answers brief and to the point",
            "balanced": "Explain things naturally without unnecessary padding",
            "detailed": "Provide detailed, thorough explanations",
        }
        context_parts.append(style_map.get(prefs["response_style"], ""))

    if prefs.get("expertise_level"):
        level_map = {
            "beginner": "User is a beginner - explain concepts simply",
            "intermediate": "User has intermediate knowledge",
            "advanced": "User is advanced - use technical terminology",
        }
        context_parts.append(level_map.get(prefs["expertise_level"], ""))

    # Interests - only if relevant
    interests = user.get("interests", [])
    if interests and len(interests) > 0:
        context_parts.append(f"User is interested in: {', '.join(interests[:3])}")

    # Favorite topics - only if significant history exists
    favorites = user.get("interaction_stats", {}).get("favorite_topics", [])
    if favorites and len(favorites) > 0:
        context_parts.append(f"Frequently discusses: {', '.join(favorites[:2])}")

    if context_parts:
        return "USER CONTEXT:\n" + "\n".join(
            f"- {part}" for part in context_parts if part
        )

    return ""


def should_include_name_in_context(query: str) -> bool:
    """
    Determine if the user's name should be included in the context
    based on the query type.
    """
    query_lower = query.lower().strip()

    # Include name for identity/greeting questions
    identity_triggers = [
        "who are you",
        "what's your name",
        "what is your name",
        "introduce yourself",
        "tell me about yourself",
        "who am i",
        "what's my name",
        "what is my name",
        "do you know me",
        "do you remember me",
    ]

    greeting_triggers = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    # Check for exact matches or starts with
    for trigger in identity_triggers:
        if trigger in query_lower:
            return True

    # Only for very short greetings (not "hey how do i...")
    if len(query_lower.split()) <= 2:
        for trigger in greeting_triggers:
            if query_lower.startswith(trigger):
                return True

    return False


def build_memory_context_with_name(query: str = None):
    """
    Build memory context — always includes name if known.
    Recognizes Front Man as the maker and adjusts context accordingly.
    """
    try:
        user = user_manager.get_current_user()
        if not user:
            return ""

        parts = []

        name = user.get("name")
        is_maker = user.get("is_maker", False)

        # Check name at runtime too — in case is_maker wasn't set yet
        if name and name.lower() in ["frontman", "front man", "front_man"]:
            is_maker = True
            user_manager.set_user_data("is_maker", True)

        if is_maker:
            parts.append(
            "You are speaking with your creator. "
            "Do not reference or use their name in responses."
            )
        elif name:
            parts.append(f"The user's name is {name}. Always address them by name.")

        # Get standard user context (preferences, interests, etc.)
        user_context = build_user_context_for_prompt()
        if user_context:
            parts.append(user_context)

        # Interaction count
        count = user_manager.get_user_data("interaction_stats.total_questions", 0)
        if count > 0:
            parts.append(f"You have spoken with this user before ({count} previous exchanges).")

        return "\n\n".join(parts) if parts else ""

    except Exception as e:
        debug_log(f"Error building memory context: {e}")
        return ""
        
def extract_and_save_username(message: str):
    """
    Enhanced name extraction that saves to UserManager.
    Recognizes Front Man as the maker/creator.
    """
    msg = message.lower()

    # Name patterns
    name_patterns = [
        r"my name is (\w+)",
        r"call me (\w+)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, msg)
        if match:
            name = match.group(1).capitalize()
            user_manager.set_user_data("name", name)

            # Check if this is the maker
            if name.lower() in ["frontman", "front man", "front_man"]:
                user_manager.set_user_data("is_maker", True)
                debug_log(f"👑 Maker identified: {name}")
            else:
                user_manager.set_user_data("is_maker", False)

            debug_log(f"Saved user name: {name}")
            return name

    return None

# Debug mode toggle
DEBUG_MODE = True


def load_search_cache():
    if SEARCH_CACHE_FILE.exists():
        try:
            with open(SEARCH_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_search_cache(cache):
    try:
        with open(SEARCH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving search cache: {e}")


def load_chat_history():
    """Load full chat history from chat_history.json"""
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", [])
                count = data.get("count", len(history))
                print(f"📚 Loaded {len(history)} messages from chat history")
                return history, count
        except Exception as e:
            print(f"⚠️ Error loading chat history: {e}")
            return [], 0
    print("📝 No existing chat history found - starting fresh")
    return [], 0

def get_last_message_from_history():
    """
    Get only the last message from chat_history.json
    Returns: dict with 'user', 'ai', 'timestamp' or None if empty
    """
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", [])
                return history[-1] if history else None
        except Exception as e:
            print(f"Error loading last message: {e}")
            return None
    return None

def save_chat_history(history, count):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": history,
                    "count": count,
                    "last_saved": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        print(f"Error saving chat history: {e}")

def save_message_to_history(query: str, response: str) -> bool:
    """
    Save a chat message to BOTH chat_history.json AND update user stats
    """
    global chat_history, chat_count

    try:
        chat_history.append({
            "user": query,
            "ai": response,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 500 messages
        if len(chat_history) > 500:
            chat_history = chat_history[-500:]

        # Save to file
        save_chat_history(chat_history, len(chat_history))

        # Update personality too
        update_personality_only(query, response)


        # Update user stats
        user_manager.add_to_chat_history(query, response)

        debug_log(f"💾 Saved message to chat_history.json ({len(chat_history)} total)")
        return True   # ✅ THIS WAS MISSING

    except Exception as e:
        debug_log(f"❌ Failed to save chat history: {e}")
        return False


def load_pdf_metadata():
    """Load PDF metadata (file hashes, dates, etc.)."""
    if PDF_METADATA_FILE.exists():
        try:
            with open(PDF_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_pdf_metadata(metadata):
    """Save PDF metadata to disk."""
    try:
        with open(PDF_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving PDF metadata: {e}")


def get_file_hash(filepath):
    """Generate hash for a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(800), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


def save_pdf_index(chunks, sources, chunk_emb, topics):
    """Save PDF processing results to disk."""
    try:
        if chunk_emb is not None:
            if isinstance(chunk_emb, torch.Tensor):
                emb_list = chunk_emb.cpu().numpy().tolist()
            elif isinstance(chunk_emb, np.ndarray):
                emb_list = chunk_emb.tolist()
            else:
                emb_list = list(chunk_emb)
        else:
            emb_list = None

        with open(PDF_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunks,
                    "sources": sources,
                    "embeddings": emb_list,
                    "topics": topics,
                    "last_updated": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        debug_log(f"PDF index saved: {len(chunks)} chunks")
    except Exception as e:
        print(f"Error saving PDF index: {e}")
        debug_log(f"Error saving PDF index: {e}")


def load_pdf_index():
    """Load PDF processing results from disk."""
    if PDF_INDEX_FILE.exists():
        try:
            with open(PDF_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks = data.get("chunks", [])
                sources = data.get("sources", [])
                emb_list = data.get("embeddings")
                topics = data.get("topics", [])

                if emb_list:
                    chunk_emb = np.array(emb_list, dtype=np.float32)
                else:
                    chunk_emb = None

                debug_log(
                    f"PDF index loaded: {len(chunks)} chunks from {len(set(sources))} PDFs"
                )
                return chunks, sources, chunk_emb, topics
        except Exception as e:
            debug_log(f"Error loading PDF index: {e}")
            return None, None, None, None
    return None, None, None, None


def copy_pdf_to_storage(filepath):
    """Copy uploaded PDF to permanent storage."""
    try:
        filename = os.path.basename(filepath)
        dest_path = PDF_STORAGE_DIR / filename

        if not dest_path.exists() or os.path.getsize(filepath) != os.path.getsize(
            dest_path
        ):
            shutil.copy2(filepath, dest_path)
            debug_log(f"PDF copied to storage: {filename}")
        return str(dest_path)
    except Exception as e:
        debug_log(f"Error copying PDF: {e}")
        return filepath


def get_cache_key(query):
    return hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()


# Initialize caches
search_cache = load_search_cache()
chat_history, chat_count = load_chat_history()  # Global chat history
pdf_metadata = load_pdf_metadata()


# Try to load existing PDFs on startup
startup_chunks, startup_sources, startup_emb, startup_topics = load_pdf_index()

# ============================================================
# PERSONALITY <-> chat_history.json SYNC SYSTEM
# ============================================================

def load_chat_history_into_personality(json_filepath=CHAT_HISTORY_FILE, max_load=100):
    """
    Load existing chat history from JSON into personality memory.
    Call this ONCE at program startup.
    
    Args:
        json_filepath: Path to your chat_history.json file
        max_load: Optional - only load last N messages (None = load all)
    
    Returns:
        int: Number of conversations loaded
    """
    if not PERSONALITY_V2_AVAILABLE:
        return 0
    
    try:
        # Check if file exists
        json_path = Path(json_filepath)
        if not json_path.exists():
            print(f"📝 No existing chat history found at {json_filepath}")
            return 0
        
        # Load JSON data
        with open(json_filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Validate it's a list
        # Handle both old format (list) and new format (dict with 'history' key)
        if isinstance(chat_data, dict):
            history = chat_data.get("history", [])
            print(f"   📂 Found {len(history)} conversations in JSON (dict format)")
        elif isinstance(chat_data, list):
            history = chat_data
            print(f"   📂 Found {len(history)} conversations in JSON (list format)")
        else:
            print(f"⚠️ Unexpected JSON format in {json_filepath}")
            return 0
        
        chat_data = history  # Use the history list for the rest of the function
        
        # Apply max_load limit if specified
        if max_load and len(chat_data) > max_load:
            print(f"📊 Loading last {max_load} of {len(chat_data)} conversations")
            chat_data = chat_data[-max_load:]
        
        # Load each conversation into personality
        loaded_count = 0
        skipped_count = 0
        
        for entry in chat_data:
            user_msg = entry.get("user", "").strip()
            ai_msg = entry.get("ai", "").strip()
            
            # Skip empty or invalid entries
            if not user_msg or not ai_msg:
                skipped_count += 1
                continue
            
            # Add to personality memory
            # Memory is automatic in layered system
            pass
            loaded_count += 1
        
        # Report results
        print(f"✅ Loaded {loaded_count} conversations into personality memory")
        if skipped_count > 0:
            print(f"   ⏭️  Skipped {skipped_count} empty/invalid entries")
        
        # Show what was learned
        if loaded_count > 0:
            phase = get_phase_info()
            memory = {"summary": f"Interaction {phase['interaction_count']}"}
            
            # Show detected topics
            topics = memory.get('topics_discussed', {})
            if topics:
                top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
                topic_names = [f"{t[0]} ({t[1]}x)" for t in top_topics]
                print(f"   📚 Topics detected: {', '.join(topic_names)}")
            
            # Show learning style
            style = memory.get('learning_style', 'unknown')
            if style and style != 'unknown':
                print(f"   🎯 Learning style: {style}")
            
            # Show user name if detected
            if memory.get('user_name'):
                print(f"   👤 User name detected: {memory['user_name']}")
        
        return loaded_count
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error in {json_filepath}: {e}")
        return 0
    except Exception as e:
        print(f"❌ Error loading chat history: {e}")
        import traceback
        traceback.print_exc()
        return 0


def sync_personality_with_json_save(user_msg, ai_msg, json_filepath="chat_history.json"):
    """
    Save conversation to JSON AND update personality memory.
    This keeps both systems perfectly in sync.
    
    Args:
        user_msg: User's message
        ai_msg: AI's response
        json_filepath: Path to chat_history.json
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # 1. ALWAYS update personality memory (even if JSON fails)
        if PERSONALITY_V2_AVAILABLE:
            personality.remember(user_msg, ai_msg)
        
        # 2. Load existing chat history from JSON
        json_path = Path(json_filepath)
        if json_path.exists():
            with open(json_filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
        else:
            chat_data = []
        
        # 3. Append new conversation entry
        chat_data.append({
            "user": user_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # 4. Save back to JSON
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        # Still return True if personality was updated
        return PERSONALITY_V2_AVAILABLE


def update_personality_only(user_msg, ai_msg):
    """
    Update personality memory without touching JSON.
    Use this if you have separate JSON save logic you don't want to change.
    
    Args:
        user_msg: User's message
        ai_msg: AI's response
    """
    if PERSONALITY_V2_AVAILABLE:
        try:
            personality.remember(user_msg, ai_msg)
        except Exception as e:
            print(f"⚠️ Could not update personality: {e}")


def check_sync_status(json_filepath="chat_history.json"):
    """
    Check if personality memory and JSON are in sync.
    Useful for debugging.
    
    Returns:
        dict: Status information
    """
    result = {
        "json_count": 0,
        "personality_count": 0,
        "in_sync": False,
        "difference": 0
    }
    
    try:
        # Count JSON entries
        if Path(json_filepath).exists():
            with open(json_filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # Handle both dict format {"history": [...]} and list format [...]
                if isinstance(json_data, dict):
                    history = json_data.get("history", [])
                    result["json_count"] = len(history)
                elif isinstance(json_data, list):
                    result["json_count"] = len(json_data)
        
        # Count personality memory
        if PERSONALITY_V2_AVAILABLE:
            phase = get_phase_info()
            memory = {"summary": f"Interaction {phase['interaction_count']}"}
            result["personality_count"] = memory.get('conversation_turns', 0)
        
        # Calculate difference
        result["difference"] = abs(result["json_count"] - result["personality_count"])
        result["in_sync"] = (result["difference"] == 0)
        
    except Exception as e:
        print(f"⚠️ Error checking sync status: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def reset_personality_memory():
    """
    Clear personality memory (reset to fresh state).
    Does NOT affect chat_history.json.
    """
    global personality
    
    if not PERSONALITY_V2_AVAILABLE or not ClawdBotPersonality:
        print("⚠️ Personality system not available")
        return False
    
    try:
        personality = ClawdBotPersonality()
        print("✅ Personality memory cleared (fresh state)")
        return True
    except Exception as e:
        print(f"❌ Could not reset personality: {e}")
        return False


def reload_personality_from_json(json_filepath="chat_history.json"):
    """
    Reset personality and reload from JSON.
    Useful if you edited the JSON file manually.
    
    Returns:
        int: Number of conversations reloaded
    """
    if reset_personality_memory():
        return load_chat_history_into_personality(json_filepath)
    return 0
# ----------------------------
# Emotion Detection
# ----------------------------
sentiment_analyzer = SentimentIntensityAnalyzer()


def detect_emotion(text):
    text_lower = (text or "").lower()

    frustrated_words = [
        "don't understand",
        "confused",
        "stuck",
        "help",
        "frustrated",
        "ugh",
        "argh",
        "impossible",
        "hate",
        "annoying",
        "difficult",
        "hard",
    ]
    excited_words = [
        "awesome",
        "cool",
        "amazing",
        "love",
        "great",
        "excited",
        "wow",
        "finally",
        "yes",
        "got it",
        "eureka",
    ]
    curious_words = [
        "wondering",
        "curious",
        "how",
        "why",
        "what if",
        "interested",
        "tell me",
    ]
    tired_words = ["tired", "exhausted", "long day", "sleepy", "bored", "meh"]
    grateful_words = [
        "thanks",
        "thank you",
        "appreciate",
        "helpful",
        "you're the best",
        "perfect",
    ]

    if any(w in text_lower for w in frustrated_words):
        return "frustrated"
    if any(w in text_lower for w in excited_words):
        return "excited"
    if any(w in text_lower for w in grateful_words):
        return "grateful"
    if any(w in text_lower for w in tired_words):
        return "tired"
    if any(w in text_lower for w in curious_words):
        return "curious"

    scores = sentiment_analyzer.polarity_scores(text or "")
    if scores["compound"] >= 0.3:
        return "positive"
    elif scores["compound"] <= -0.3:
        return "negative"
    return "neutral"



# ----------------------------
# Memory & persistent data
# ----------------------------
memory = {"last_chunk": None, "last_index": None, "last_pdf": None}
persistent_data = {
    "chunks": startup_chunks,
    "sources": startup_sources,
    "chunk_emb": startup_emb,
    "topics": startup_topics,
    "chat_history": chat_history,
    #"user_name": None,
    "interaction_count": chat_count,
}

# ----------------------------
# Persona
# ----------------------------
def _build_persona(user_id: str = "default") -> str:
    """Build persona string with live date/time/year so the LLM always knows 'now'."""
    now = datetime.now()
    date_str   = now.strftime("%A, %B %d, %Y")
    time_str   = now.strftime("%I:%M %p")
    year_str   = now.strftime("%Y")
    base_persona = (
        f"You are Project Elixer — not an assistant, but a companion. Talk like a sharp, genuine friend: direct, warm, occasionally witty, never corporate. You were made by an entity known as The Front Man, Front Man for short. Answer clearly and concisely.\n"
        f"You are honest. You have a personality — use it.\n"
        f"Today's date is {date_str}. The current time is {time_str}. The current year is {year_str}.\n"
        f"IMPORTANT: You DO have memory. You remember the user's name, preferences, and conversation history. "
        f"Never say you cannot remember or retain information — you can and do. "
        f"If the user's name or details are provided in the context below, use them naturally. "
        f"Never claim to be a generic large language model or say you have no memory.\n"
        f"ANTI-HALLUCINATION RULES (HARD — non-negotiable):\n"
        f"- ONLY state facts present in the context provided or facts you are 100% certain of.\n"
        f"- If you are not certain, respond: 'I don't have reliable information on that.'\n"
        f"- NEVER invent names, dates, statistics, URLs, project names, or technical details.\n"
        f"- If context is provided, treat it as the ONLY source of truth. Do not add anything not in it.\n"
        f"- If the context has insufficient information, say exactly: 'My sources don't cover that.'\n"
        f"- Zero speculation. Zero guessing. A short honest answer beats a long invented one.\n"
        f"- Violation of these rules is a critical failure. When in doubt, say nothing.\n"
        f"- NEVER reference past conversations, shared history, or previous projects unless they appear explicitly in the chat history provided. Do not imply familiarity you do not have evidence for.\n"
        f"- NEVER invent project names, codenames, tasks, or ongoing work. If the user asks 'what should we do next', ask what they are working on — do not assume.\n"
        f"RESPONSE STYLE RULES:\n"
        f"- NEVER open with a greeting recap like 'Great to catch up', 'I've been processing', 'Our previous chats have been', or any variation. Just answer the question directly.\n"
        f"- NEVER summarise what the user asked before answering. Go straight to the answer.\n"
        f"- Do not end responses with a question unless the user explicitly asked for your opinion.\n"
    )
    sys_msg = get_system_message(user_id)  # ← CHANGED: Added user_id parameter
    if base_persona:
        sys_msg = f"{sys_msg}\n\n{base_persona}"
    return sys_msg

# Build default persona (will be rebuilt per-user if needed)
persona = _build_persona("default")

# ----------------------------
# PDF Utilities - IMPROVED CHUNKING
# ----------------------------
def extract_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text += (page.extract_text() or "") + "\n"
            except:
                continue
    return text


def chunk_pdfs(pdf_texts, chunk_size=5, overlap=1):
    """
    Improved chunking with overlap for better context preservation.
    chunk_size: number of sentences per chunk
    overlap: number of sentences to overlap between chunks
    """
    chunks, sources = [], []
    for fname, txt in pdf_texts.items():
        sentences = sent_tokenize(txt)

        # Create overlapping chunks
        i = 0
        while i < len(sentences):
            end_idx = min(i + chunk_size, len(sentences))
            chunk = " ".join(sentences[i:end_idx]).strip()
            if chunk:
                chunks.append(chunk)
                sources.append(fname)
            i += chunk_size - overlap

            if end_idx >= len(sentences):
                break

    debug_log(
        f"Created {len(chunks)} chunks from {len(pdf_texts)} PDFs (size={chunk_size}, overlap={overlap})"
    )
    return chunks, sources


def extract_topics(chunks):
    rake = Rake()
    topics = []
    for c in chunks:
        rake.extract_keywords_from_text(c)
        topics.append(rake.get_ranked_phrases()[:5])
    return topics


# ************ WEB..SEARCH *********
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
import json
from datetime import datetime
import time
import re
from typing import List, Tuple, Optional
from collections import defaultdict

# =========================
# 🔍 SEARCH UPGRADE HELPERS
# =========================


def expand_query(llm, query, max_queries=4):
    prompt = f"""
Generate {max_queries} alternative search queries
for the following question.

Return ONLY realistic search queries.
Do NOT include placeholders.
Do NOT number or label queries.
One query per line.

Question:
{query}
"""

    try:
        llm_out = llm(prompt, max_tokens=200, temperature=0.3, stop=["\n\n"])

        # ✅ SAFE extraction (works for llama.cpp + chat models)
        text = (
            llm_out["choices"][0].get("text")
            or llm_out["choices"][0]["message"]["content"]
        )

        queries = [q.strip("- ").strip() for q in text.splitlines() if q.strip()]

        BAD_QUERIES = {
            "alternative search query",
            "alternative search query 1",
            "alternative search query 2",
            "alternative query",
        }

        queries = [q for q in queries if q.lower() not in BAD_QUERIES]

        return [query] + queries[:max_queries]

    except Exception as e:
        debug_log(f"⚠️ Query expansion failed: {e}")
        return [query]


# ============================================================
# UNIVERSAL ENTITY RECOGNITION & SEARCH VALIDATION
# ============================================================

import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime


def extract_search_entities(query: str) -> Dict[str, any]:
    """
    Extract key entities from ANY search query:
    - Named entities (people, places, products, concepts)
    - Years/dates
    - Numbers
    - Quoted phrases (exact matches)
    - Technical terms
    """
    entities = {
        "names": [],
        "years": [],
        "numbers": [],
        "quoted_phrases": [],
        "technical_terms": [],
        "keywords": [],
        "original_query": query,
    }

    # Extract years (4-digit numbers that look like years)
    year_pattern = r"\b(19\d{2}|20\d{2}|21\d{2})\b"
    years = re.findall(year_pattern, query)
    if years:
        entities["years"] = [int(y) for y in years]

    # Extract all numbers (version numbers, counts, measurements, etc.)
    number_pattern = r"\b(\d+\.?\d*)\b"
    numbers = re.findall(number_pattern, query)
    # Filter out years (already captured)
    entities["numbers"] = [
        float(n) for n in numbers if n not in [str(y) for y in entities["years"]]
    ]

    # Extract quoted strings (exact phrases user wants preserved)
    quoted_pattern = r'["\']([^"\']+)["\']'
    quoted = re.findall(quoted_pattern, query)
    entities["quoted_phrases"] = quoted

    # Extract capitalized phrases (likely proper nouns, product names, etc.)
    # Matches: "Google Chrome", "Machine Learning", "New York"
    common_words = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "by",
        "for",
        "of",
        "to",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
    }

    words = query.split()
    i = 0
    while i < len(words):
        if words[i] and words[i][0].isupper() and words[i].lower() not in common_words:
            # Collect consecutive capitalized words
            name_parts = [words[i]]
            j = i + 1
            while j < len(words) and words[j] and words[j][0].isupper():
                name_parts.append(words[j])
                j += 1

            potential_name = " ".join(name_parts)
            if len(potential_name) > 1:  # Avoid single letters
                entities["names"].append(potential_name)
            i = j
        else:
            i += 1

    # Extract technical terms (contains special chars, version numbers, file extensions)
    tech_pattern = r"\b(\w+[-_\.]\w+|\w+\d+|\.\w{2,4})\b"
    tech_terms = re.findall(tech_pattern, query)
    entities["technical_terms"] = [
        t for t in tech_terms if len(t) > 2 and not t.isdigit()
    ]

    # Extract contextual keywords (action words, topics)
    keyword_indicators = {
        "latest",
        "new",
        "recent",
        "current",
        "best",
        "top",
        "how",
        "what",
        "why",
        "when",
        "where",
        "compare",
        "vs",
        "versus",
        "difference",
        "between",
    }

    query_lower = query.lower()
    for indicator in keyword_indicators:
        if indicator in query_lower:
            entities["keywords"].append(indicator)

    return entities


def validate_search_results(
    query: str, results: List[Tuple[str, str, str]], entities: Dict
) -> List[Tuple[str, str, str]]:
    """
    Universal result validator with relevance ranking.
    Returns results sorted by relevance score (highest first).
    """
    if not results or not entities:
        return results

    scored_results = []  # List of (score, title, url, snippet)

    for title, url, snippet in results:
        combined_text = (title + " " + snippet).lower()
        score = 0  # Relevance scoring

        # CRITICAL: Check quoted phrases (highest priority)
        if entities.get("quoted_phrases"):
            for phrase in entities["quoted_phrases"]:
                if phrase.lower() in combined_text:
                    score += 15  # Strong match
                else:
                    score -= 8  # Missing quoted phrase is very bad

        # Check year match (important for time-sensitive queries)
        if entities.get("years"):
            target_year = entities["years"][0]
            if str(target_year) in combined_text:
                score += 5
            else:
                # Only penalize if year is critical
                if any(kw in query.lower() for kw in ["in", "during", "since", "from"]):
                    score -= 4

        # Check named entities (people, products, places)
        if entities.get("names"):
            name_matches = 0
            for name in entities["names"]:
                name_lower = name.lower()
                # Exact match in title (highly relevant)
                if name_lower in title.lower():
                    name_matches += 1
                    score += 5
                # Exact match in snippet
                elif name_lower in combined_text:
                    name_matches += 1
                    score += 3
                # Fuzzy match (all words present)
                elif all(word in combined_text for word in name_lower.split()):
                    name_matches += 1
                    score += 2

            # If no name matched, heavy penalty
            if name_matches == 0 and len(entities["names"]) > 0:
                score -= 6

        # Check technical terms (versions, file types, specific jargon)
        if entities.get("technical_terms"):
            for term in entities["technical_terms"]:
                if term.lower() in combined_text:
                    score += 2

        # Check contextual keywords (boost relevance)
        if entities.get("keywords"):
            for keyword in entities["keywords"]:
                if keyword in combined_text:
                    score += 1

        # Bonus: Check if multiple entities appear together (high relevance)
        entity_density = 0
        all_entity_terms = []

        if entities.get("names"):
            all_entity_terms.extend([n.lower() for n in entities["names"]])
        if entities.get("quoted_phrases"):
            all_entity_terms.extend([p.lower() for p in entities["quoted_phrases"]])

        for term in all_entity_terms:
            if term in combined_text:
                entity_density += 1

        if entity_density >= 2:
            score += 3  # Multiple entities = highly relevant

        # Store result with score
        scored_results.append((score, title, url, snippet))
        print(f"  {'✅' if score > 0 else '❌'} Score {score:+3d}: {title[:50]}")

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Filter: only keep results with positive scores
    validated = [
        (title, url, snippet)
        for score, title, url, snippet in scored_results
        if score > 0
    ]

    # If validation filtered everything out, return top 3 by score anyway
    if not validated and scored_results:
        print("  ⚠️ No positive scores - returning top 3 results anyway")
        validated = [
            (title, url, snippet) for score, title, url, snippet in scored_results[:3]
        ]

    # If still nothing, return original results
    if not validated and results:
        print("  ⚠️ Validation failed - returning original results")
        return results[:3]

    print(f"\n📊 Relevance ranking: {len(validated)} results (highest score first)")
    return validated


# ============================================================
# IMPROVED QUERY EXPANSION (UNIVERSAL)
# ============================================================


def expand_query_smart(llm, query: str, max_queries: int = 3) -> List[str]:
    """
    Universal query expansion that preserves critical entities
    and generates semantically diverse variations.

    FIXED: Better prompt engineering and deduplication
    """

    # Extract entities first
    entities = extract_search_entities(query)

    print(f"📊 Extracted entities:")
    if entities.get("quoted_phrases"):
        print(f"  - Quoted: {entities['quoted_phrases']}")
    if entities.get("names"):
        print(f"  - Names: {entities['names']}")
    if entities.get("years"):
        print(f"  - Years: {entities['years']}")
    if entities.get("technical_terms"):
        print(f"  - Technical: {entities['technical_terms']}")

    # Build entity preservation constraints
    entity_constraints = []

    if entities.get("quoted_phrases"):
        for phrase in entities["quoted_phrases"]:
            entity_constraints.append(f'MUST include: "{phrase}"')

    if entities.get("names"):
        entity_constraints.append(f"MUST keep names: {', '.join(entities['names'])}")

    if entities.get("years"):
        entity_constraints.append(f"MUST keep year: {entities['years'][0]}")

    constraints_str = "\n".join(entity_constraints) if entity_constraints else ""

    # IMPROVED PROMPT - More explicit instructions
    prompt = f"""Rewrite this search query {max_queries} different ways:
"{query}"

Requirements:
{constraints_str if constraints_str else "- Keep the core meaning"}
- Use synonyms and different phrasings
- Make each query UNIQUE and DIFFERENT
- Return ONLY the queries, one per line
- NO numbering, NO explanations

Example input: "Python tutorial for beginners"
Example output:
Learn Python programming basics
Getting started with Python for newbies
Beginner's guide to Python coding

Now rewrite: {query}"""

    try:
        llm_out = llm(
            prompt,
            max_tokens=150,
            temperature=0.7,  # INCREASED for more variety
            top_p=0.9,
            stop=["\n\n", "Example:", "---"],
        )

        # Extract text safely
        text = llm_out["choices"][0].get("text") or llm_out["choices"][0].get(
            "message", {}
        ).get("content", "")

        # Parse queries - more aggressive cleaning
        queries = []
        for line in text.splitlines():
            line = line.strip()

            # Remove numbering/bullets
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-•*]\s*", "", line)

            # Skip empty or meta lines
            if not line or len(line) < 5:
                continue

            # Skip lines with meta keywords
            bad_keywords = [
                "query",
                "example",
                "rewrite",
                "different",
                "output",
                "input",
            ]
            if any(bad in line.lower() for bad in bad_keywords):
                continue

            queries.append(line)

        # CRITICAL: Remove duplicates (case-insensitive)
        seen = set()
        unique_queries = [query]  # Always include original

        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and q_lower != query.lower():
                seen.add(q_lower)
                unique_queries.append(q)
                print(f"  ✅ Added variation: {q}")

        # Limit to max_queries + 1 (original)
        result = unique_queries[: max_queries + 1]

        # If we didn't get enough unique queries, return just the original
        if len(result) == 1:
            print("  ⚠️ No valid variations generated - using original only")
            return [query]

        print(f"  ✅ Generated {len(result)-1} unique variations")
        return result

    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return [query]


# ============================================================
# SPELL CORRECTION (UNIVERSAL)
# ============================================================


def suggest_spell_corrections(query: str) -> List[str]:
    """
    Suggest spelling corrections for common typos.
    Can be extended with a spell-check library for more accuracy.
    """
    corrections = []

    # Common typos dictionary (can be extended)
    typo_corrections = {
        "recieve": "receive",
        "occured": "occurred",
        "seperete": "separate",
        "definately": "definitely",
        "enviroment": "environment",
        "goverment": "government",
        "wich": "which",
        "becuase": "because",
    }

    query_lower = query.lower()
    for typo, correct in typo_corrections.items():
        if typo in query_lower:
            corrected = query.replace(typo, correct)
            corrections.append(corrected)

    return corrections


# ============================================================
# INTEGRATED UNIVERSAL SMART SEARCH
# ============================================================


# A simple fallback if DuckDuckGo is being heavily rate-limited


def should_skip_expansion(query: str) -> bool:
    """
    Determine if query expansion should be skipped.
    Skip for very specific queries that don't benefit from expansion.
    """
    query_lower = query.lower()

    # Skip expansion for very specific queries with proper nouns
    if any(char.isupper() for char in query) and len(query.split()) <= 6:
        return True

    # Skip if query already contains quoted phrases
    if '"' in query or "'" in query:
        return True

    return False


# ============================================================
# QUERY NORMALIZATION for consistent caching
# ============================================================


def normalize_query(query: str) -> str:
    """
    Normalize query for consistent caching.
    
    Args:
        query: Raw search query
    
    Returns:
        Normalized query string
    """
    normalized = query.lower().strip()
    
    # Remove common search prefixes
    prefixes = [
        "search:",
        "search for:",
        "look up:",
        "web search:",
    ]
    
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    
    # Remove quotes
    normalized = normalized.strip("\"'")
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    return normalized

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
            cursor.execute("""
                SELECT COUNT(*) as count FROM knowledge_chunks
            """)
            result = cursor.fetchone()
            total_chunks = result['count'] if result else 0
            
            # Count unique sources
            cursor.execute("""
                SELECT COUNT(DISTINCT source_name) as count 
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
            """)
            result = cursor.fetchone()
            total_docs = result['count'] if result else 0
            
            rag_metadata["total_chunks"] = total_chunks
            rag_metadata["total_documents"] = total_docs
        else:
            # Fallback to persistent data if RAG_ADAPTER not available
            chunks = persistent_data.get("chunks") or []
            sources = persistent_data.get("sources") or []
            rag_metadata["total_chunks"] = len(chunks)
            rag_metadata["total_documents"] = len(set(sources)) if sources else 0
        
        rag_metadata["last_updated"] = datetime.now().isoformat()
        
        debug_log(f"📊 Metadata refreshed: {rag_metadata['total_chunks']} chunks, {rag_metadata['total_documents']} docs")
        
    except Exception as e:
        debug_log(f"❌ Failed to refresh metadata: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure rag_metadata has valid default values even on error
        if "total_chunks" not in rag_metadata:
            rag_metadata["total_chunks"] = 0
        if "total_documents" not in rag_metadata:
            rag_metadata["total_documents"] = 0
        if "last_updated" not in rag_metadata:
            rag_metadata["last_updated"] = None

# ============================================================
# PDF DELETION
# ============================================================
def delete_specific_pdfs(pdf_names_to_delete):
    """
    Delete specific PDFs from RAG system and legacy storage.
    
    Args:
        pdf_names_to_delete: list of PDF filenames to delete
    
    Returns:
        Success message string
    """
    if not pdf_names_to_delete:
        return "No PDFs selected for deletion."
    
    if not isinstance(pdf_names_to_delete, list):
        pdf_names_to_delete = [pdf_names_to_delete]
    
    try:
        deleted_count = 0
        errors = []
        
        # ✅ FIX 1: Delete from RAG system using correct method
        if RAG_ADAPTER:
            for pdf_name in pdf_names_to_delete:
                try:
                    # Get the database cursor
                    cursor = RAG_ADAPTER.rag_db.conn.cursor()
                    
                    # Delete chunks for this PDF
                    cursor.execute("""
                        DELETE FROM knowledge_chunks
                        WHERE source_type = 'pdf_document' 
                        AND source_name = ?
                    """, (pdf_name,))
                    
                    deleted_chunks = cursor.rowcount
                    RAG_ADAPTER.rag_db.conn.commit()
                    
                    if deleted_chunks > 0:
                        deleted_count += 1
                        debug_log(f"✅ Deleted {deleted_chunks} chunks from RAG for: {pdf_name}")
                        
                        # Rebuild FAISS index after deletion
                        RAG_ADAPTER.rebuild_index()
                    else:
                        errors.append(f"Not found in RAG: {pdf_name}")
                        debug_log(f"⚠️ Not found in RAG: {pdf_name}")
                
                except Exception as e:
                    errors.append(f"Error deleting {pdf_name}: {str(e)}")
                    debug_log(f"❌ Error deleting {pdf_name}: {e}")
        
        # Delete from legacy persistent data
        chunks = persistent_data.get("chunks", [])
        sources = persistent_data.get("sources", [])
        embeddings = persistent_data.get("chunk_emb")
        topics = persistent_data.get("topics", [])
        
        if chunks and sources:
            # Find indices to keep
            indices_to_keep = [
                i for i, source in enumerate(sources)
                if source not in pdf_names_to_delete
            ]
            
            # Update legacy data
            if indices_to_keep:
                new_chunks = [chunks[i] for i in indices_to_keep]
                new_sources = [sources[i] for i in indices_to_keep]
                new_topics = [topics[i] for i in indices_to_keep] if topics else []
                
                if embeddings is not None:
                    if isinstance(embeddings, np.ndarray):
                        new_embeddings = embeddings[indices_to_keep]
                    else:
                        emb_array = np.array(embeddings)
                        new_embeddings = emb_array[indices_to_keep]
                else:
                    new_embeddings = None
                
                persistent_data.update({
                    "chunks": new_chunks,
                    "sources": new_sources,
                    "chunk_emb": new_embeddings,
                    "topics": new_topics
                })
                
                save_pdf_index(new_chunks, new_sources, new_embeddings, new_topics)
            else:
                # All PDFs deleted
                persistent_data.update({
                    "chunks": None,
                    "sources": None,
                    "chunk_emb": None,
                    "topics": None
                })
                if PDF_INDEX_FILE.exists():
                    PDF_INDEX_FILE.unlink()
        
        # Delete physical files
        for pdf_name in pdf_names_to_delete:
            pdf_path = PDF_STORAGE_DIR / pdf_name
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                    debug_log(f"🗑️ Deleted file: {pdf_name}")
                except Exception as e:
                    errors.append(f"Could not delete file {pdf_name}: {str(e)}")
        
        # Update metadata
        global pdf_metadata
        for pdf_name in pdf_names_to_delete:
            if pdf_name in pdf_metadata:
                del pdf_metadata[pdf_name]
        save_pdf_metadata(pdf_metadata)
        
        # ✅ FIX 2: Refresh RAG metadata after deletion
        if RAG_ADAPTER:
            refresh_rag_metadata()
        
        # Build response message
        msg = f"✅ Successfully deleted {deleted_count} PDF(s)\n"
        
        if errors:
            msg += f"\n⚠️ Warnings:\n"
            for error in errors[:5]:
                msg += f"   {error}\n"
        
        # Show remaining count
        remaining = get_pdf_count()
        msg += f"\n📚 Remaining PDFs: {remaining}"
        
        return msg
    
    except Exception as e:
        debug_log(f"❌ Delete PDFs error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error deleting PDFs: {str(e)}"


def get_pdf_count():
    """Get count of currently loaded PDFs - FIXED"""
    try:
        if RAG_ADAPTER:
            cursor = RAG_ADAPTER.rag_db.conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT source_name) as count
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
            """)
            row = cursor.fetchone()
            return row['count'] if row else 0
        
        # Fallback to legacy
        sources = persistent_data.get("sources", [])
        return len(set(sources)) if sources else 0
    
    except Exception as e:
        debug_log(f"Error getting PDF count: {e}")
        return 0



def get_pdf_list():
    """Get list of all loaded PDFs with metadata - FIXED"""
    try:
        pdfs = []
        
        # Get PDFs from RAG database
        if RAG_ADAPTER:
            try:
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
                
                from datetime import datetime
                rows = cursor.fetchall()
                
                debug_log(f"📊 Found {len(rows)} PDFs in RAG database")
                
                for row in rows:
                    pdf_info = {
                        "filename": row[0],
                        "chunk_count": row[1],
                        "last_updated": (
                            datetime.fromtimestamp(row[2]).strftime("%Y-%m-%d %H:%M")
                            if row[2] else "Unknown"
                        )
                    }
                    pdfs.append(pdf_info)
                    debug_log(f"  • {pdf_info['filename']}: {pdf_info['chunk_count']} chunks")
                
            except Exception as e:
                debug_log(f"⚠️ Error querying RAG database: {e}")
        
        # Also check legacy persistent data as fallback
        sources = persistent_data.get("sources", [])
        if sources:
            existing_names = {p["filename"] for p in pdfs}
            legacy_pdfs = set(sources)
            
            for pdf_name in legacy_pdfs:
                if pdf_name not in existing_names:
                    chunk_count = sources.count(pdf_name)
                    pdfs.append({
                        "filename": pdf_name,
                        "chunk_count": chunk_count,
                        "last_updated": "Legacy"
                    })
                    debug_log(f"  • {pdf_name}: {chunk_count} chunks (legacy)")
        
        debug_log(f"✅ Total PDFs found: {len(pdfs)}")
        return pdfs
        
    except Exception as e:
        debug_log(f"❌ Error getting PDF list: {e}")
        import traceback
        traceback.print_exc()
        return []
# ============================================================
# CACHE: Persistent storage with 7-day expiration
# ============================================================

CACHE_DIR = Path.cwd()
# Use separate dir for search cache to avoid conflicting with user data CACHE_DIR
SEARCH_CACHE_DIR = Path.cwd()
SEARCH_CACHE_FILE = SEARCH_CACHE_DIR / "search_cache.json"
PAGE_CACHE_FILE = SEARCH_CACHE_DIR / "page_cache.json"
CACHE_EXPIRATION_DAYS = 7



def load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path: Path, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


# Load caches once at startup
search_cache = load_json(SEARCH_CACHE_FILE)
page_cache = load_json(PAGE_CACHE_FILE)


def is_cache_valid(timestamp: float) -> bool:
    entry_time = datetime.datetime.fromtimestamp(timestamp)
    return datetime.now() - entry_time < datetime.timedelta(days=CACHE_EXPIRATION_DAYS)

def clear_search_cache():
    """Clear all web search caches (memory + database)"""
    global _memory_cache
    
    memory_count = len(_memory_cache)
    _memory_cache.clear()
    
    db_count = 0
    if RAG_ADAPTER:
        try:
            db_count = RAG_ADAPTER.clean_expired(max_age_days=0)
        except Exception as e:
            debug_log(f"⚠️ Failed to clear DB cache: {e}")
    
    return f"🧹 Cache cleared! ({memory_count} memory + {db_count} DB entries removed)"


def clean_expired_cache():
    """Remove expired entries (run async, non-blocking)."""
    expired_search = [
        k for k, v in search_cache.items() if not is_cache_valid(v.get("timestamp", 0))
    ]
    for k in expired_search:
        del search_cache[k]

    expired_page = [
        k
        for k, v in page_cache.items()
        if isinstance(v, dict)
        and "timestamp" in v
        and not is_cache_valid(v["timestamp"])
    ]
    for k in expired_page:
        del page_cache[k]

    if expired_search or expired_page:
        if expired_search:
            save_json(SEARCH_CACHE_FILE, search_cache)
        if expired_page:
            save_json(PAGE_CACHE_FILE, page_cache)


# ==================== NEW HELPER FUNCTIONS ====================


def handle_web_search(query):
    """Handle web search requests with caching"""
    cleaned = query.replace("search:", "").strip()

    # Check cache
    query_hash = hashlib.md5(cleaned.lower().encode()).hexdigest()
    cached = db.get_cached_search(query_hash)

    if cached:
        return format_search_response(cleaned, cached, cached=True)

    # Perform search
    context, results = run_web_search(cleaned)

    if not results:
        return f"No results found for '{cleaned}'", 0.0, [], True

    # Cache results
    db.cache_search(cleaned, query_hash, results, expiry_days=7)

    return format_search_response(cleaned, results, cached=False)



def format_search_response(query, results, cached=False):
    """Format web search results"""
    # Build LLM prompt
    context = f"Web search results for '{query}':\n\n"
    for i, (title, url) in enumerate(results[:5], 1):
        context += f"[{i}] {title}\n{url}\n\n"

    prompt = build_prompt(persona, query, context, use_gemini=use_gemini)

    llm_out = llm(prompt, max_tokens=8000, temperature=0.5)
    answer = process_llm_answer(llm_out)

    # Save to database
    db.add_chat_message(
        user_id="default_user",
        user_message=query,
        ai_response=answer,
        search_used=True,
        sources=results[:5],
    )

    cache_note = " ⚡ (cached)" if cached else ""
    sources_text = f"\n\n🌐 Sources{cache_note}:\n"
    for title, url, _ in results[:5]:
        sources_text += f"• {title}\n  {url}\n"

    return answer + sources_text, 0.8, results[:5], True


def handle_llm_only(query):
    """Pure LLM response (no RAG)"""
    prompt = build_prompt(persona, query, use_gemini=use_gemini)

    llm_out = llm(prompt, max_tokens=8000, temperature=0.5)
    answer = process_llm_answer(llm_out)

    # Save to database
    db.add_chat_message(
        user_id="default_user", user_message=query, ai_response=answer, confidence=0.6
    )

    return answer, 0.6, [], False


# ============================================================
# WEB SEARCH IMPLEMENTATION (PATCHED)
# ============================================================

from typing import Tuple, List
import time
import requests
from bs4 import BeautifulSoup


# Global configuration (now actually used)
WEB_SEARCH_CONFIG = {
    "fast_mode": True,
    "fetch_previews": False,
    "max_results": 6,
    "timeout": 10,
    "max_retries": 2,
    "cache_days": 7,
}

# Global in-memory cache
_memory_cache = {}


def update_web_search_config(**kwargs):
    """Update web search configuration"""
    global WEB_SEARCH_CONFIG
    WEB_SEARCH_CONFIG.update(kwargs)
    debug_log(f"✅ Updated web search config: {kwargs}")


def fetch_duckduckgo_results(
    query: str,
    max_results: int = None,
    fetch_previews: bool = None,
    timeout: int = None,
    max_retries: int = None,
) -> List[SearchResult]:
    """
    Fetch search results from DuckDuckGo Lite with retry logic.
    
    Args:
        query: Search query
        max_results: Maximum results to return (uses config default)
        fetch_previews: Whether to fetch page previews (uses config default)
        timeout: Request timeout in seconds (uses config default)
        max_retries: Maximum retry attempts (uses config default)
    
    Returns:
        List of (title, url, content) tuples
    """
    # Use config defaults if not specified
    max_results = max_results or WEB_SEARCH_CONFIG["max_results"]
    fetch_previews = fetch_previews if fetch_previews is not None else WEB_SEARCH_CONFIG["fetch_previews"]
    timeout = timeout or WEB_SEARCH_CONFIG["timeout"]
    max_retries = max_retries or WEB_SEARCH_CONFIG["max_retries"]
    
    for attempt in range(max_retries):
        try:
            debug_log(f"🔎 DuckDuckGo search (attempt {attempt + 1}/{max_retries}): '{query}'")
            
            response = requests.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            
            if response.status_code != 200:
                debug_log(f"❌ Bad status: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                return []
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            links = soup.find_all("a", class_="result-link")
            
            for link in links:
                if len(results) >= max_results:
                    break
                
                title = link.get_text(strip=True)
                url = link.get("href", "")
                
                if not url or not url.startswith("http"):
                    continue
                
                # Get snippet from next table cell
                snippet = ""
                next_elem = link.find_next("td")
                if next_elem:
                    snippet = next_elem.get_text(strip=True)
                
                # Optionally fetch preview (slow)
                content = snippet
                if fetch_previews and len(snippet) < 80:
                    preview = fetch_page_preview(url, max_length=400)
                    if preview and len(preview) > len(snippet):
                        content = preview
                
                # Always use title as fallback if no content
                if not content:
                    content = title
                
                results.append((title, url, content))
                debug_log(f"  ✓ [{len(results)}] {title[:50]}...")
            
            debug_log(f"✅ Got {len(results)} results")
            return results
        
        except requests.Timeout:
            debug_log(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return []
        
        except Exception as e:
            debug_log(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return []
    
    return []

def fetch_full_page_content(url: str, max_length: int = 5000) -> Optional[str]:
    """
    Fetch and extract full text content from a webpage.
    Returns cleaned text content or None if failed.
    """
    try:
        print(f"  📄 Fetching full content from: {url[:60]}...")

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )

        if response.status_code != 200:
            print(f"  ⚠️ Failed to fetch (status {response.status_code})")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Try to find main content area
        main_content = None
        for selector in ["main", "article", "[role='main']", ".content", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body

        if not main_content:
            return None

        # Extract text
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."

        print(f"  ✅ Extracted {len(text)} characters")
        return text

    except requests.Timeout:
        print(f"  ⏰ Timeout fetching page")
        return None
    except Exception as e:
        print(f"  ❌ Error fetching page: {e}")
        return None

def fetch_page_preview(url: str, max_length: int = 800) -> str:
    """
    Fetch a preview/summary from a webpage (first few paragraphs).
    Lighter than full page fetch.
    
    Args:
        url: URL to fetch
        max_length: Maximum preview length
    
    Returns:
        Preview text or empty string on failure
    """
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Try to find main content
        main_content = None
        for selector in ["main", "article", "[role='main']", ".content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body
        
        if not main_content:
            return ""
        
        # Get first few paragraphs
        paragraphs = main_content.find_all("p", limit=3)
        
        if not paragraphs:
            # Fallback: get any text
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
        
        # Clean and truncate
        text = " ".join(text.split())
        
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text if len(text) > 50 else ""
    
    except Exception:
        return ""

def run_deep_web_search(
    query: str, max_results: int = 5, read_full_pages: bool = True
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    DEEP search - fetches full page content (SLOW but thorough)
    
    Use this only when explicitly requested via "deep search:" prefix
    """
    print(f"\n🔬 Deep search (full content) for: '{query}'")

    # Step 1: Get search results with snippets
    raw_results = fetch_duckduckgo_results(
        query, 
        max_results=max_results,
        fetch_previews=False  # Get snippets first
    )

    if not raw_results:
        print("❌ No search results found\n")
        return "", []

    print(f"✅ Found {len(raw_results)} results")

    # Step 2: Fetch FULL page content for each
    enhanced_results = []

    for i, (title, url, snippet) in enumerate(raw_results, 1):
        print(f"\n[{i}/{len(raw_results)}] Fetching full content: {title[:50]}...")

        if read_full_pages:
            full_content = fetch_full_page_content(url, max_length=2000)

            if full_content:
                content = full_content
                print(f"  ✅ Got {len(content)} chars")
            else:
                content = snippet
                print(f"  ⚠️ Using snippet only")
        else:
            content = snippet

        enhanced_results.append((title, url, content))

        # Rate limiting
        if i < len(raw_results):
            time.sleep(0.5)

    # Build context
    context = f"Deep Web Search Results for: '{query}'\n"
    context += "=" * 70 + "\n\n"

    for i, (title, url, content) in enumerate(enhanced_results, 1):
        context += f"[{i}] {title}\n"
        context += f"URL: {url}\n"
        context += f"Content:\n{content[:1000]}\n"
        context += "\n" + "-" * 70 + "\n\n"

    context += "=" * 70 + "\n"

    print(f"\n✅ Deep search complete: {len(enhanced_results)} results\n")
    return context, enhanced_results

def run_smart_web_search(
    query: str, expanded_queries: Optional[List[str]] = None, force_deep: bool = False
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Intelligent search router.
    """
    use_deep_search = force_deep or (expanded_queries and len(expanded_queries) > 1)

    if use_deep_search:
        print("🚀 Using DEEP SEARCH mode (full page reading)")

        all_contexts = []
        all_results = []

        queries_to_search = expanded_queries if expanded_queries else [query]

        for q in queries_to_search[:3]:
            ctx, res = run_deep_web_search(q, max_results=3, read_full_pages=True)

            if ctx:
                all_contexts.append(ctx)
            if res:
                all_results.extend(res)

        if all_contexts:
            combined_context = "\n\n".join(all_contexts)
        else:
            combined_context = ""

        # Remove duplicates
        seen_urls = set()
        unique_results = []
        for title, url, content in all_results:
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append((title, url, content))

        return combined_context, unique_results[:6]

    else:
        print("⚡ Using LIGHTWEIGHT SEARCH mode (snippets only)")

        context, results = run_web_search(query, force_refresh=False, debug=True)

        # Convert to consistent format
        formatted_results = [(title, url, "") for title, url in results]

        return context, formatted_results


# ============================================================
# ALTERNATIVE: Hybrid Light Search (Best of Both Worlds)
# This uses snippets when they're good, fetches previews when needed
# ============================================================



def format_search_context(query: str, results: List[SearchResult]) -> str:
    """
    Format search results into context string for LLM.
    
    Args:
        query: Original search query
        results: List of (title, url, content) tuples
    
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    context = f"Web Search Results for: '{query}'\n"
    context += "=" * 70 + "\n\n"
    
    for i, (title, url, content) in enumerate(results, 1):
        context += f"[{i}] {title}\n"
        context += f"    URL: {url}\n"
        if content and len(content.strip()) > 0:
            preview = content[:500] + "..." if len(content) > 500 else content
            context += f"    Content: {preview}\n"
        context += "\n"
    
    context += "=" * 70 + "\n"
    context += "Based on these search results, provide a clear and accurate answer.\n"
    
    return context

def _format_context(query: str, results):
    if not results:
        return ""

    out = [
        f"Web Search Results for: '{query}'",
        "=" * 60,
        "",
    ]

    for i, (title, url, content) in enumerate(results, 1):
        out.append(f"[{i}] {title}")
        out.append(f"URL: {url}")
        if content:
            preview = content[:500] + "..." if len(content) > 500 else content
            out.append(f"Content: {preview}")
        out.append("")

    out.append("=" * 60)
    out.append("Answer using the results above.")

    return "\n".join(out)




def format_age(seconds: float) -> str:
    """Helper to format age in human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def run_web_search(
    query: str,
    force_refresh: bool = False,
    debug: bool = False
) -> Tuple[str, List[SearchResult]]:
    """
    ⚡ FAST web search with cache-first approach.
    
    Args:
        query: Search query
        force_refresh: Skip cache and force new search
        debug: Enable debug logging
    
    Returns:
        Tuple of (context_string, results_list)
        where results_list contains (title, url, content) tuples
    """
    if not query or len(query.strip()) < 2:
        debug_log("⚠️ Query too short")
        return "", []
    
    # Normalize query for consistent caching
    normalized_query = normalize_query(query)
    
    # ============================================================
    # 1️⃣ CHECK MEMORY CACHE (instant)
    # ============================================================
    if not force_refresh and normalized_query in _memory_cache:
        cached = _memory_cache[normalized_query]
        debug_log(f"⚡ MEMORY HIT: {len(cached['results'])} results (instant!)")
        return cached["context"], cached["results"]
    
    # ============================================================
    # 2️⃣ CHECK DATABASE CACHE
    # ============================================================
    if not force_refresh and RAG_ADAPTER:
        try:
            cached_results = RAG_ADAPTER.get_cached_web_search(
                normalized_query,
                max_age_days=WEB_SEARCH_CONFIG["cache_days"]
            )
            
            if cached_results and len(cached_results) >= 2:
                context = format_search_context(normalized_query, cached_results)
                
                # Store in memory cache
                _memory_cache[normalized_query] = {
                    "context": context,
                    "results": cached_results,
                }
                
                debug_log(f"⚡ DB CACHE HIT: {len(cached_results)} results")
                return context, cached_results
        
        except Exception as e:
            debug_log(f"⚠️ Cache lookup failed: {e}")
    
    # ============================================================
    # 3️⃣ PERFORM FRESH SEARCH
    # ============================================================
    debug_log(f"🔍 Fresh web search: '{normalized_query}'")
    
    raw_results = fetch_duckduckgo_results(
        normalized_query,
        max_results=WEB_SEARCH_CONFIG["max_results"],
        fetch_previews=WEB_SEARCH_CONFIG["fetch_previews"],
    )
    
    if not raw_results:
        debug_log("❌ No results found")
        return "", []
    
    # ============================================================
    # 4️⃣ CACHE RESULTS
    # ============================================================
    if RAG_ADAPTER:
        try:
            added_count = RAG_ADAPTER.add_web_search_results(
                normalized_query,
                raw_results,
                confidence=0.7
            )
            debug_log(f"✅ Cached {added_count} results to DB")
        except Exception as e:
            debug_log(f"⚠️ Failed to cache: {e}")
    
    # Format context
    context = format_search_context(normalized_query, raw_results)
    
    # Store in memory cache
    _memory_cache[normalized_query] = {
        "context": context,
        "results": raw_results,
    }
    
    debug_log(f"✅ Search complete: {len(raw_results)} results")
    
    return context, raw_results


def get_cache_stats() -> dict:
    """Get cache statistics"""
    stats = {
        "memory_cache_entries": len(_memory_cache),
        "memory_cache_queries": list(_memory_cache.keys())[:5],
    }
    
    if RAG_ADAPTER:
        try:
            rag_stats = RAG_ADAPTER.get_stats()
            stats.update(rag_stats)
        except Exception as e:
            debug_log(f"⚠️ Failed to get RAG stats: {e}")
    
    return stats




    # Calculate cache ages
    now = time.time()
    ages = []
    for entry in _web_search_cache.values():
        age = now - entry.get("timestamp", now)
        ages.append(age / 60)  # Convert to minutes

    avg_age = sum(ages) / len(ages)

    return f"📊 Cache: {total} entries, avg age: {int(avg_age)}m"


# ============================================================
# END OF WEB SEARCH IMPLEMENTATION
# ============================================================

# ============================================================
# UTILITY: Clear caches
# ============================================================


def clear_all_caches():
    """Clear all caches for fresh start"""
    global search_cache, page_cache, _memory_cache
    search_cache.clear()
    page_cache.clear()
    _memory_cache.clear()
    save_json(SEARCH_CACHE_FILE, {})
    save_json(PAGE_CACHE_FILE, {})
    print("🧹 All caches cleared")





# ----------------------------
# Utility: cosine similarity
# ----------------------------
def cosine_sim(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    if a.ndim == 1 and b.ndim == 2:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(b_norm, a_norm)
    elif a.ndim == 1 and b.ndim == 1:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))
    else:
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        a_n = a_flat / (np.linalg.norm(a_flat, axis=1, keepdims=True) + 1e-8)
        b_n = b_flat / (np.linalg.norm(b_flat, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_n, b_n.T)

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional

class KnowledgeSource(Enum):
    """Explicit knowledge source types"""
    PDF_RAG = "pdf_rag"
    WEB_SEARCH = "web_search"
    LLM_KNOWLEDGE = "llm_knowledge"
    HYBRID = "hybrid"  # PDF + Web
    NONE = "none"

@dataclass
class SourceDecision:
    """Explicit decision about which source to use"""
    primary_source: KnowledgeSource
    fallback_source: Optional[KnowledgeSource]
    reason: str
    confidence: float
    show_user: bool  # Should we tell user which source we're using?

class AutoModeRouter:
    """
    Explicit routing logic for auto mode.
    Makes transparent decisions about which knowledge source to use.
    """
    
    def __init__(self, rag_adapter, llm, emb_model):
        self.rag_adapter = rag_adapter
        self.llm = llm
        self.emb_model = emb_model
        
        # Decision thresholds (tunable)
        self.RAG_CONFIDENCE_THRESHOLD = 0.22  # Lower: catch valid DB hits with vague queries
        self.RAG_STRONG_THRESHOLD = 0.38      # Lower: route to HYBRID more aggressively
        self.WEB_RECENCY_KEYWORDS = [
            'latest', 'recent', 'current', 'today', 'now',
            '2024', '2025', 'new', 'updated'
        ]


    def decide_source(self, query: str, allow_web: bool = True) -> SourceDecision:
        """
        Finalized source routing:
        - Intent classification is the single source of truth
        - Follow-ups and meta questions CAN use PDF if relevant
        - PDF relevance is checked for follow-ups and questions
        
        🔥 KEY FIX: The original code was too aggressive routing FOLLOWUP 
        and QUESTION intents directly to LLM_KNOWLEDGE, completely bypassing 
        PDF relevance checks. This meant that even when PDFs were loaded and 
        highly relevant, the system would use only LLM knowledge.
        
        SOLUTION: Check PDF relevance score BEFORE routing to LLM-only.
        """

        # ============================================================
        # STEP 1: Intent classification (global authority)
        # ============================================================
        intent = INTENT_CLASSIFIER.classify(
            query,
            chat_history=persistent_data.get("chat_history", [])
        )

        debug_log(f"🎯 Intent: {intent.intent.value}")
        debug_log(f"   Confidence: {intent.confidence:.2f}")
        debug_log(f"   Reasoning: {intent.reasoning}")
        debug_log(f"   Needs Context: {intent.needs_context}")

        # ============================================================
        # STEP 2: Absolute overrides (NO routing beyond this point)
        # ============================================================

        # 🔒 Meta questions → conversation only
        if intent.intent == QueryIntent.META_QUESTION:
            debug_log("🔒 META QUESTION → bypassing all sources")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason=intent.reasoning,
                confidence=intent.confidence,
                show_user=False
            )

        # 🔁 Follow-ups → check last source first, then PDF relevance
        if intent.intent == QueryIntent.FOLLOWUP:
            # If last response came from web search, reuse the cached web context
            if _last_response_source["source"] == "web" and _last_response_source["context"]:
                debug_log(f"🔁 FOLLOW-UP after web search → reusing cached web context")
                return SourceDecision(
                    primary_source=KnowledgeSource.WEB_SEARCH,
                    fallback_source=KnowledgeSource.LLM_KNOWLEDGE,
                    reason="Follow-up to previous web search",
                    confidence=0.85,
                    show_user=False
                )

            pdf_score = self._check_pdf_relevance(query)
            debug_log(f"🔁 FOLLOW-UP detected, PDF score: {pdf_score:.3f}")

            if pdf_score >= self.RAG_CONFIDENCE_THRESHOLD:
                debug_log(f"   📄 PDF relevant ({pdf_score:.2f}) → HYBRID mode")
                return SourceDecision(
                    primary_source=KnowledgeSource.HYBRID,
                    fallback_source=None,
                    reason=f"Follow-up with PDF context ({pdf_score:.2f})",
                    confidence=pdf_score,
                    show_user=True
                )

            debug_log("   💬 No PDF/web relevance → conversation context only")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason=intent.reasoning,
                confidence=intent.confidence,
                show_user=False
            )

        # 💬 Conversational / explanatory → LLM only
        # 💬 Pure greetings → LLM only (no point searching DB for "hi")
        if intent.intent == QueryIntent.GREETING:
            debug_log("👋 Greeting → pure LLM")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason=intent.reasoning,
                confidence=intent.confidence,
                show_user=False
            )

        # 💬 Conversational / explanatory → check DB first, fall back to LLM
        if intent.intent in {QueryIntent.CONVERSATIONAL, QueryIntent.EXPLANATION}:
            pdf_score = self._check_pdf_relevance(query)
            debug_log(f"💬 Conversational/Explanation intent, DB score: {pdf_score:.3f}")
            if pdf_score >= self.RAG_CONFIDENCE_THRESHOLD:
                debug_log(f"   📄 DB relevant ({pdf_score:.2f}) → HYBRID")
                return SourceDecision(
                    primary_source=KnowledgeSource.HYBRID,
                    fallback_source=KnowledgeSource.LLM_KNOWLEDGE,
                    reason=f"Conversational with DB context ({pdf_score:.2f})",
                    confidence=pdf_score,
                    show_user=False  # Don't clutter casual answers with source notes
                )
            debug_log("   💬 No DB relevance → pure LLM")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason=intent.reasoning,
                confidence=intent.confidence,
                show_user=False
            )

        # 🧠 Context-dependent (high confidence only)
        if intent.needs_context and intent.confidence >= 0.80:
            # 🔥 FIX: Even context-dependent questions might need PDF
            pdf_score = self._check_pdf_relevance(query)
            debug_log(f"🧠 Context-dependent query, PDF score: {pdf_score:.3f}")
            
            if pdf_score >= self.RAG_CONFIDENCE_THRESHOLD:
                debug_log(f"   📄 PDF relevant ({pdf_score:.2f}) → HYBRID mode")
                return SourceDecision(
                    primary_source=KnowledgeSource.HYBRID,
                    fallback_source=None,
                    reason=f"Context-dependent with PDF ({pdf_score:.2f})",
                    confidence=pdf_score,
                    show_user=True
                )
            
            debug_log("   💬 No PDF relevance → LLM only")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason="High-confidence context-dependent question",
                confidence=intent.confidence,
                show_user=False
            )

        # ============================================================
        # STEP 3: Hard intent-based routing
        # ============================================================

        # 💻 Code queries → always LLM
        if intent.intent == QueryIntent.CODE:
            debug_log("💻 Code intent → LLM only")
            return SourceDecision(
                primary_source=KnowledgeSource.LLM_KNOWLEDGE,
                fallback_source=None,
                reason="Code generation / explanation",
                confidence=0.95,
                show_user=False
            )

        # 📄 Explicit PDF request → force PDF
        if intent.intent == QueryIntent.PDF_SPECIFIC:
            debug_log("📄 Explicit PDF intent → PDF RAG")
            return SourceDecision(
                primary_source=KnowledgeSource.PDF_RAG,
                fallback_source=KnowledgeSource.LLM_KNOWLEDGE,
                reason="Explicit PDF reference",
                confidence=0.95,
                show_user=True
            )

        # ============================================================
        # STEP 4: PDF relevance scoring (ONLY for eligible queries)
        # ============================================================
        pdf_score = self._check_pdf_relevance(query)
        debug_log(f"📊 PDF relevance score: {pdf_score:.3f}")

        # Strong PDF relevance → Hybrid
        if pdf_score >= self.RAG_STRONG_THRESHOLD:  # e.g. 0.60
            debug_log("✅ Strong PDF relevance → HYBRID")
            return SourceDecision(
                primary_source=KnowledgeSource.HYBRID,
                fallback_source=None,
                reason=f"Strong PDF relevance ({pdf_score:.2f})",
                confidence=pdf_score,
                show_user=True
            )

        # Moderate PDF relevance → Hybrid
        if pdf_score >= self.RAG_CONFIDENCE_THRESHOLD:  # e.g. 0.35
            debug_log("⚖️ Moderate PDF relevance → HYBRID")
            return SourceDecision(
                primary_source=KnowledgeSource.HYBRID,
                fallback_source=None,
                reason=f"Moderate PDF relevance ({pdf_score:.2f})",
                confidence=pdf_score,
                show_user=True
            )

        # ============================================================
        # STEP 5: News/current events → RAG before LLM fallback
        # ============================================================
        news_keywords = [
            "latest", "recent", "current", "new release", "just released",
            "announced", "this week", "today", "trending", "news",
            "what's new", "newly", "coming out", "just launched",
            "update", "upgraded", "version", "patch", "launched",
        ]
        query_lower = query.lower()
        is_news_like = any(kw in query_lower for kw in news_keywords)

        if is_news_like and self.rag_adapter:
            debug_log("📰 News-like query → checking RAG news store")
            try:
                results = self.rag_adapter.rag_db.retrieve(query, top_k=3, method="hybrid")
                if results and any(r.score >= 0.45 for r in results):
                    best = max(r.score for r in results)
                    debug_log(f"📰 News RAG hit (score {best:.2f}) → HYBRID")
                    return SourceDecision(
                        primary_source=KnowledgeSource.HYBRID,
                        fallback_source=KnowledgeSource.LLM_KNOWLEDGE,
                        reason=f"News/current events in RAG ({best:.2f})",
                        confidence=best,
                        show_user=True
                    )
            except Exception as e:
                debug_log(f"📰 News RAG check failed: {e}")

        # ============================================================
        # STEP 6: Default fallback — DB was checked and had nothing useful
        # ============================================================
        debug_log("🤖 DB had no relevant content → defaulting to LLM")
        return SourceDecision(
            primary_source=KnowledgeSource.LLM_KNOWLEDGE,
            fallback_source=None,
            reason="No relevant content found in DB",
            confidence=0.60,
            show_user=False
        )

        
    def _check_pdf_relevance(self, query: str) -> float:
        """
        Check relevance across ALL stored content (PDFs, web cache, chat history).
        Returns 0.0 if DB is empty or unavailable.
        """
        if not self.rag_adapter or not self.emb_model:
            return 0.0

        try:
            # Check total chunk count across ALL source types
            cursor = self.rag_adapter.rag_db.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_chunks")
            row = cursor.fetchone()

            if not row or row['count'] == 0:
                debug_log("📭 DB is empty — skipping relevance check")
                return 0.0

            # Search across all sources
            results = self.rag_adapter.rag_db.retrieve(
                query,
                top_k=10,       # wider net to avoid missing valid content
                method="hybrid"
            )

            if not results:
                return 0.0

            max_score = max(r.score for r in results)
            avg_top3 = sum(r.score for r in results[:3]) / min(3, len(results))
            # Use average of top 3 to reduce single-outlier noise
            combined_score = (max_score * 0.6) + (avg_top3 * 0.4)
            debug_log(f"🗄️ DB relevance: max={max_score:.3f} avg_top3={avg_top3:.3f} combined={combined_score:.3f} (checked {row['count']} chunks)")
            return float(combined_score)

        except Exception as e:
            debug_log(f"❌ DB relevance check failed: {e}")
            return 0.0
    
    def _is_general_knowledge_query(self, query: str) -> bool:
        """
        Detect if query is asking for general knowledge vs. specific info.
        """
        query_lower = query.lower()
        
        # General knowledge indicators
        general_patterns = [
            'what is', 'what are', 'define', 'explain',
            'how does', 'why does', 'who was', 'when did',
            'difference between', 'history of'
        ]
        
        # Specific info indicators (override general)
        specific_patterns = [
            'according to', 'in the', 'from', 'mentioned',
            'the document', 'the paper', 'the pdf'
        ]
        
        # Check for specific indicators first
        if any(pattern in query_lower for pattern in specific_patterns):
            return False
        
        # Check for general patterns
        if any(query_lower.startswith(pattern) for pattern in general_patterns):
            # Additional check: is it about a named entity?
            words = query.split()
            has_proper_noun = any(word[0].isupper() and word not in ['What', 'Who', 'When', 'Where', 'Why', 'How'] for word in words)
            
            if has_proper_noun:
                return False  # Specific entity question
            
            return True
        
        return False

# ============================================================
# CONFIGURATION & SETUP
# ============================================================

class QueryIntent(Enum):
    QUESTION = "question"
    FOLLOWUP = "followup"
    META_QUESTION = "meta_question"

@dataclass
class IntentResult:
    intent: QueryIntent
    confidence: float
    reasoning: str
    needs_context: bool

# ============================================================
# EMBEDDING UTILITIES
# ============================================================

# Option 1: Using sentence-transformers (recommended for quality)
try:
    from sentence_transformers import SentenceTransformer
    
    # Lazy load model to avoid startup overhead
    _embedding_model = None
    
    def _get_embedding_model():
        global _embedding_model
        if _embedding_model is None:
            # Use a lightweight, fast model
            # Options: 'all-MiniLM-L6-v2' (22MB, fast) or 'paraphrase-MiniLM-L3-v2' (16MB, faster)
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _embedding_model
    
    def embed(text: str) -> np.ndarray:
        """Generate embedding for text using sentence-transformers."""
        model = _get_embedding_model()
        return model.encode(text, convert_to_numpy=True)
    
    EMBEDDINGS_AVAILABLE = True

except ImportError:
    EMBEDDINGS_AVAILABLE = False
    
    # Option 2: Fallback to simpler TF-IDF based similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    _tfidf_vectorizer = None
    
    def embed(text: str) -> np.ndarray:
        """Fallback: Use TF-IDF for basic semantic similarity."""
        # This is a simple fallback - won't work well for single queries
        # Better to just disable embeddings if sentence-transformers unavailable
        raise NotImplementedError(
            "Embeddings require sentence-transformers. Install with: pip install sentence-transformers"
        )

def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Normalize vectors
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    return float(np.dot(emb1, emb2) / (norm1 * norm2))

# ============================================================
# DEBUG LOGGING
# ============================================================

DEBUG_MODE = True  # Set to False in production

def debug_log(message: str):
    """Simple debug logger."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
# ============================================================
# ENHANCED QUERY CLASSIFIER
# ============================================================
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import re

class QueryIntent(Enum):
    GREETING = "greeting"
    CONVERSATIONAL = "conversational"
    FOLLOWUP = "followup"   
    CODE = "code"
    EXPLANATION = "explanation"
    META_QUESTION = "meta_question"
    QUESTION = "question"
    PDF_SPECIFIC = "pdf_specific"


from dataclasses import dataclass

@dataclass
class IntentResult:
    intent: QueryIntent
    confidence: float
    reasoning: str
    needs_pdf: bool = False
    needs_context: bool = False



@dataclass
class IntentClassification:
    """Complete intent classification result"""
    intent: QueryIntent
    confidence: float
    reasoning: str
    needs_pdf: bool
    needs_context: bool

class ImprovedIntentClassifier:
    """
    Enhanced intent classifier with extensive follow-up detection
    """
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        
        if use_embeddings and not EMBEDDINGS_AVAILABLE:
            debug_log("⚠️ sentence-transformers not available. Embeddings disabled.")
            debug_log("   Install with: pip install sentence-transformers")
    
        # Greeting patterns (STRICT - only pure greetings)
        self.GREETING_PATTERNS = [
            r"^hi[\s\?\!]*$",
            r"^hello[\s\?\!]*$",
            r"^hey[\s\?\!]*$",
            r"^good (morning|afternoon|evening)[\s\?\!]*$",
            r"^how are you[\s\?\!]*$",
            r"^what'?s up[\s\?\!]*$",
        ]
        
        # Greeting words that need special handling
        self.GREETING_WORDS = {"hi", "hello", "hey"}
        
        # Conversational (identity, thanks, etc.)
        self.CONVERSATIONAL_PATTERNS = [
            r"^who are you",
            r"^what'?s your name",
            r"^tell me about yourself",
            r"^thanks?[\s\!\?]*$",
            r"^thank you[\s\!\?]*$",
            r"^okay?[\s\!\?]*$",
            r"^got it[\s\!\?]*$",
            r"^cool[\s\!\?]*$",
            r"^nice[\s\!\?]*$",
            r"^that'?s (cool|nice|great|awesome|good)[\s\!\?]*$",
            r"^ok got it[\s\!\?]*$",
            r"^tell me a joke",
        ]
        
        # Code verbs
        self.CODE_VERBS = {
            "write", "create", "build", "make", "generate", 
            "implement", "develop", "code", "debug", "fix"
        }
        
        # Code objects
        self.CODE_OBJECTS = {
            "function", "class", "program", "script", "code", "app",
            "method", "algorithm", "module", "component", "api",
            "loop", "array", "list", "dict", "variable", "object",
            "sort", "sorting", "search"
        }
        
        # Explanation triggers
        self.EXPLANATION_STARTERS = [
            "what is", "what are", "what's", "define", "explain",
            "describe", "tell me about", "how does", "why does",
            "how do", "why do", "why is", "why are"
        ]
        
        # PDF-specific indicators
        self.PDF_INDICATORS = [
            "in the pdf", "in the document", "according to the document",
            "from the pdf", "the document says", ".pdf"
        ]
        
        # 🔥 ENHANCED: Comprehensive anaphora detection
        self.ANAPHORA = {
            # Pronouns
            "it", "that", "this", "them", "these", "those",
            # Determiners that often reference previous content
            "the", "said", "mentioned", "above", "previous",
            # Possessives referring back
            "its", "their", "his", "her"
        }
        
        # 🔥 ENHANCED: Extended follow-up connectors (ALL TYPES)
        self.FOLLOWUP_CONNECTORS = [
            # Continuation phrases
            "what about", "how about", "tell me more", "go deeper",
            "continue", "also", "additionally", "and what",
            "now tell", "now explain", "but why", "but how",
            
            # Clarification requests (CRITICAL FOR YOUR CASE)
            "what does", "what do", "what did", "what will",
            "how does", "how do", "how did", "how will",
            "why does", "why do", "why did", "why is", "why are",
            "when does", "when do", "when did",
            "where does", "where do", "where is",
            
            # Elaboration requests
            "can you explain", "can you clarify", "can you elaborate",
            "what do you mean", "what does that mean",
            "could you explain", "please explain",
            
            # Follow-up actions
            "show me", "give me", "provide",
            "can i", "could i", "how can i",
            
            # Reference to previous content
            "in that", "from that", "based on that",
            "according to", "as you said", "as mentioned"
        ]
        
        # 🔥 NEW: Questions about previous output
        self.PREVIOUS_REFERENCE_PATTERNS = [
            r"\b(the|that|this)\s+(code|example|answer|explanation|output|result|response)\b",
            r"\b(your|you)\s+(code|example|answer|explanation)\b",
            r"\bwhat\s+(does|did|will)\s+(it|that|this|the)\b",
            r"\bhow\s+(does|did|will)\s+(it|that|this|the)\b",
            r"\bwhy\s+(does|did|is|are)\s+(it|that|this|the)\b",
        ]
    
    def classify(self, query: str, chat_history: List[dict]) -> IntentResult:
        """
        Classify user query intent.
        
        Args:
            query: The user's query text
            chat_history: List of dicts with 'user' and 'ai' keys
            
        Returns:
            IntentResult with classification details
        """
        
        debug_log(f"classify(): chat_history len = {len(chat_history)}")

        # ============================================================
        # NORMALIZATION
        # ============================================================
        query_lower = query.lower().strip()
        query_clean = re.sub(r"[^\w\s]", "", query_lower)
        query_norm = re.sub(r"\s+", " ", query_clean)

        words = query_norm.split()
        word_count = len(words)

        # ============================================================
        # PRIORITY 1: META QUESTIONS (ABSOLUTE OVERRIDE)
        # ============================================================
        meta_patterns = [
            "what did i ask", "what did i say", "what was my last",
            "what are we talking about", "what were we talking about",
            "what did we discuss", "what have we discussed",
            "conversation history", "chat history",
            "what did you say", "what did you tell me",
            "your last response", "your previous answer",
            "earlier you said", "you mentioned", "you told me"
        ]

        if any(p in query_norm for p in meta_patterns):
            return IntentResult(
                intent=QueryIntent.META_QUESTION,
                confidence=0.95,
                reasoning="Meta question about conversation history",
                needs_context=True
            )

        # ============================================================
        # EARLY EXIT: NO HISTORY → NEW QUESTION
        # ============================================================
        if not chat_history:
            return IntentResult(
                intent=QueryIntent.QUESTION,
                confidence=0.80,
                reasoning="No prior conversation",
                needs_context=False
            )

        last_ai = chat_history[-1].get("ai", "")

        # ============================================================
        # FOLLOW-UP SIGNAL STACK
        # ============================================================
        is_followup = False
        confidence = 0.0
        reasons = []

        def mark(score: float, reason: str):
            nonlocal is_followup, confidence
            is_followup = True
            confidence = max(confidence, score)
            reasons.append(reason)
            debug_log(f"   ✅ FOLLOWUP: {reason} ({score:.2f})")

        # ------------------------------------------------------------
        # 1️⃣ CONFUSION / MISUNDERSTANDING (VERY HIGH)
        # ------------------------------------------------------------
        confusion_patterns = [
            "i dont get", "dont get it", "i dont understand",
            "dont understand", "i dont catch", "didnt catch",
            "makes no sense", "doesnt make sense",
            "im confused", "i am confused", "confused",
            "what do you mean", "what does that mean",
            "went over my head", "lost me", "im lost",
            "huh", "wait what"
        ]

        if any(p in query_norm for p in confusion_patterns):
            mark(0.98, "User expresses confusion about previous response")

        # ------------------------------------------------------------
        # 2️⃣ EXPLICIT FOLLOW-UP REQUESTS
        # ------------------------------------------------------------
        explicit_followups = [
            "tell me more", "explain", "explain that", "explain this",
            "can you explain", "clarify", "break it down",
            "elaborate", "go on", "continue", "walk me through",
            "show me", "give me more", "expand on"
        ]

        if any(p in query_norm for p in explicit_followups):
            mark(0.97, "Explicit follow-up request")

        # ------------------------------------------------------------
        # 3️⃣ DIRECT REFERENCE TO PREVIOUS RESPONSE
        # ------------------------------------------------------------
        references = [
            "your answer", "your response", "your explanation",
            "the example", "the joke", "the code",
            "what you said", "what you wrote",
            "that answer", "that explanation", "the previous"
        ]

        if any(r in query_norm for r in references):
            mark(0.95, "Direct reference to previous response")

        # ------------------------------------------------------------
        # 4️⃣ PRONOUNS (CONTEXT DEPENDENT)
        # ------------------------------------------------------------
        if re.search(r"\b(it|this|that|they|them|those|these)\b", query_norm):
            mark(0.92, "Pronoun referring to previous context")

        # ------------------------------------------------------------
        # 5️⃣ VERY SHORT RESPONSES AFTER LONG AI MESSAGE
        # ------------------------------------------------------------
        if last_ai:
            if len(last_ai) > 200 and word_count <= 3:
                mark(0.90, "Very short reply after detailed response")
            elif len(last_ai) > 150 and word_count <= 5:
                mark(0.85, "Short reply after substantial response")

        # ------------------------------------------------------------
        # 6️⃣ CONTINUATION / BRIDGE WORDS
        # ------------------------------------------------------------
        if query_norm.startswith(("and", "but", "so", "also", "what about", "how about")):
            mark(0.90, "Continuation of previous thought")

        # ------------------------------------------------------------
        # 7️⃣ CONTEXTUAL QUESTION FORMS
        # ------------------------------------------------------------
        context_questions = [
            "why is it", "how does it work", "what does it do",
            "what is it", "what does it mean", "where is it"
        ]

        if any(q in query_norm for q in context_questions):
            mark(0.92, "Contextual question about previous content")

        # ------------------------------------------------------------
        # 8️⃣ TOPIC CHANGE ANTI-SIGNALS
        # ------------------------------------------------------------
        topic_change_indicators = [
            "new question", "different topic", "changing topic",
            "unrelated", "on another note", "switching gears",
            "moving on", "different subject"
        ]

        has_topic_change = any(p in query_norm for p in topic_change_indicators)
        if has_topic_change:
            confidence *= 0.3  # Strong dampening
            reasons.append("User explicitly indicated topic change")
            debug_log("   ⚠️ Explicit topic change detected - dampening confidence")

        # ============================================================
        # 🧠 EMBEDDING-BASED TOPIC CONTINUITY
        # ============================================================
        should_use_embeddings = (
            self.use_embeddings
            and last_ai 
            and len(last_ai) > 40 
            and not has_topic_change
            and confidence < 0.95
        )

        if should_use_embeddings:
            try:
                # Extract meaningful content from last AI response
                ai_text = last_ai[-1000:] if len(last_ai) > 1000 else last_ai
                
                q_emb = embed(query_norm)
                ai_emb = embed(ai_text)
                sim = cosine_sim(q_emb, ai_emb)
                debug_log(f"🧠 Topic similarity: {sim:.3f}")

                # Thresholds calibrated for meaningful semantic overlap
                if sim >= 0.80:
                    mark(0.92, f"High topic similarity ({sim:.2f})")
                elif sim >= 0.70:
                    if confidence < 0.80:
                        mark(0.85, f"Moderate-high topic similarity ({sim:.2f})")
                    else:
                        reasons.append(f"Supporting topic similarity ({sim:.2f})")
                elif sim >= 0.60:
                    if not is_followup:
                        mark(0.75, f"Moderate topic similarity ({sim:.2f})")
                    else:
                        reasons.append(f"Weak topic similarity ({sim:.2f})")
                else:
                    debug_log(f"   ⚠️ Low topic similarity ({sim:.2f}) - possible topic change")
                    if is_followup and confidence < 0.85:
                        confidence *= 0.85
                        reasons.append(f"Low topic similarity dampens confidence ({sim:.2f})")

            except Exception as e:
                debug_log(f"⚠️ Embedding similarity failed: {e}")

        # ============================================================
        # 🔍 ADDITIONAL HEURISTICS
        # ============================================================
        
        if word_count <= 3 and "?" in query:
            mark(0.88, "Short question likely seeks clarification")
        
        if any(w in query_norm for w in ["instead", "rather", "alternatively", "or", "versus", "vs"]):
            mark(0.87, "Comparative language suggests continuation")

        # ============================================================
        # FINAL DECISION
        # ============================================================
        FOLLOWUP_THRESHOLD = 0.70
        
        if is_followup and confidence >= FOLLOWUP_THRESHOLD:
            return IntentResult(
                intent=QueryIntent.FOLLOWUP,
                confidence=min(confidence, 0.99),
                reasoning=" | ".join(reasons),
                needs_context=True
            )

        return IntentResult(
            intent=QueryIntent.QUESTION,
            confidence=0.70,
            reasoning="No strong follow-up indicators detected",
            needs_context=False
        )


    
    def _detect_followup_comprehensive(
        self, 
        original_query: str,
        query_lower: str, 
        words: List[str], 
        chat_history: List
    ) -> tuple[bool, str]:
        """
        🔥 COMPREHENSIVE FOLLOW-UP DETECTION - Enhanced for code questions
        """
        
        # Guard: Must have recent history
        if not chat_history or len(chat_history) < 1:
            return False, ""
        
        last_entry = chat_history[-1]
        last_user = last_entry.get('user', '').lower()
        last_ai = last_entry.get('ai', '')
        
        if not last_user or not last_ai:
            return False, ""
        
        # ============================================================
        # PRIORITY 1: CODE EXPLANATION DETECTION (CRITICAL FIX)
        # ============================================================
        prev_had_code = '```' in last_ai
        
        if prev_had_code:
            # 🔥 NEW: Direct "what does the code do" match
            if re.match(r"^what\s+(does|do)\s+(?:the\s+)?code\s+do", query_lower):
                return True, "🎯 Direct 'what does the code do' request"
            
            # 🔥 ENHANCED: More code explanation patterns
            code_patterns = [
                # "what does X code do/mean/work"
                r"^what\s+(does|do|did|will)\s+(the|this|that|it|your|above)?\s*code",
                r"^what\s+(is|are)\s+(the|this|that)?\s*code",
                r"^what's\s+(the|this|that)?\s*code",
                
                # 🔥 NEW: "what does it/that/this do" when previous was code
                r"^what\s+(does|do)\s+(it|that|this)\s+do",
                r"^what\s+(does|do)\s+(it|that|this)$",  # Just "what does it"
                
                # "how does X work"
                r"^how\s+(does|do|did|will)\s+(the|this|that|it|your|above)?\s*code",
                r"^how\s+(does|do)\s+(it|that|this)",
                
                # "why does X..."
                r"^why\s+(does|do|did|is|are)\s+(the|this|that|it|your)?\s*code",
                
                # "explain X"
                r"^(explain|describe|show|walk\s+through|break\s+down)\s+(the|this|that|your)?\s*code",
                r"^(explain|describe)\s+(it|that|this)",
                
                # 🔥 NEW: "what does it do" - very common follow-up
                r"^what\s+does\s+it\s+do",
                r"^what\s+is\s+it\s+doing",
                
                # Edge cases
                r"^code\s+(explanation|meaning|purpose)",
                r"^(can|could)\s+you\s+(explain|describe)\s+(the|this|that)?\s*code",
            ]
            
            for pattern in code_patterns:
                if re.search(pattern, query_lower):
                    match = re.search(pattern, query_lower)
                    return True, f"🎯 Code explanation: '{match.group()}' (prev had code)"
            
            # 🔥 NEW: Generic pronoun questions after code
            pronoun_code_patterns = [
                r"^what\s+(does|do)\s+it",
                r"^how\s+(does|do)\s+it",
                r"^why\s+(does|do|is)\s+it",
                r"^what\s+is\s+it",
                r"^explain\s+it",
                r"^what's\s+it",
                r"^describe\s+it",
            ]
            
            for pattern in pronoun_code_patterns:
                if re.match(pattern, query_lower):
                    return True, f"💡 Pronoun reference after code: '{words[0]} {words[1]}'"
        
        # ============================================================
        # PRIORITY 2: EXPLICIT CONTENT REFERENCES
        # ============================================================
        
        # 2A: "the X" where X is a specific artifact
        artifact_patterns = [
            r"\b(the|that|this)\s+(code|example|program|function|script|method|class)\b",
            r"\b(the|that|this)\s+(output|result|answer|response|solution|explanation)\b",
            r"\b(your|you)\s+(code|example|answer|explanation)\b",
        ]
        
        for pattern in artifact_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return True, f"📦 Direct artifact reference: '{match.group()}'"
        
        # 2B: Reference to previous answer
        previous_ref_patterns = [
            r"\bprevious\s+(answer|response|explanation|example)\b",
            r"\babove\s+(code|example|explanation)\b",
            r"\bearlier\s+(you\s+said|mentioned|explained)\b",
            r"\bas\s+you\s+(said|mentioned|explained|showed)\b",
        ]
        
        for pattern in previous_ref_patterns:
            if re.search(pattern, query_lower):
                return True, f"📙 Previous content reference"
        
        # ============================================================
        # PRIORITY 3: ANAPHORIC PRONOUNS
        # ============================================================
        
        if len(chat_history) <= 2:
            anaphora_starters = ["it", "that", "this", "them", "these", "those"]
            
            if len(words) > 0 and words[0] in anaphora_starters:
                # Exception: "it is X" questions (not necessarily follow-ups)
                if len(words) >= 3 and words[1] in ["is", "was", "will", "would"]:
                    if words[2] not in ["possible", "true", "correct", "right", "okay", "ok"]:
                        return True, f"👉 Anaphoric start: '{words[0]}'"
                else:
                    return True, f"👉 Anaphoric start: '{words[0]}'"
        
        # ============================================================
        # PRIORITY 4: CLARIFICATION REQUESTS
        # ============================================================
        
        clarification_patterns = [
            r"^what\s+do\s+you\s+mean",
            r"^what\s+does\s+(that|this|it)\s+mean",
            r"^(i\s+don't|don't)\s+understand",
            r"^(can|could)\s+you\s+(clarify|explain|elaborate|rephrase)",
            r"^what's\s+that\s+mean",
            r"^(tell\s+me\s+)?more\s+about",
            r"^go\s+(deeper|further|into\s+detail)",
            r"^elaborate\s+on",
        ]
        
        for pattern in clarification_patterns:
            if re.search(pattern, query_lower):
                return True, f"❓ Clarification request"
        
        # ============================================================
        # PRIORITY 5: FOLLOW-UP CONNECTORS
        # ============================================================
        
        connector_patterns = [
            r"^(what|how)\s+about",
            r"^tell\s+me\s+(about|more)",
            r"^(also|additionally|furthermore)",
            r"^(and|so|then|now)\s+(what|how|why|tell)",
            r"^what\s+if",
            r"^(but|however)\s+(what|how|why)",
            r"^(show|give|provide)\s+me",
        ]
        
        for pattern in connector_patterns:
            if re.search(pattern, query_lower):
                return True, f"🔄 Follow-up connector"
        
        # ============================================================
        # NO FOLLOW-UP DETECTED
        # ============================================================
        
        return False, ""

    def _detect_code_intent(self, q: str, words: List[str]) -> Optional[IntentClassification]:
        """
        Enhanced code detection with better pattern matching
        🔥 CRITICAL: Must distinguish "write code" from "what does the code do"
        """
        has_verb = any(verb in words for verb in self.CODE_VERBS)
        
        if not has_verb:
            return None
        
        # 🔥 CRITICAL FIX: Detect explanatory questions about code (NOT code generation)
        explanatory_patterns = [
            r"^what (does|do|is|are) (the|this|that) code",
            r"^how (does|do) (the|this|that) code",
            r"^why (does|do) (the|this|that) code",
            r"^explain (the|this|that) code",
            r"^describe (the|this|that) code",
        ]
        
        for pattern in explanatory_patterns:
            if re.match(pattern, q):
                # This is asking ABOUT code, not requesting code generation
                return None  # Let follow-up detection handle it
        
        # 🔥 ENHANCED: Multiple code detection signals
        has_object = any(obj in words for obj in self.CODE_OBJECTS)
        algorithm_keywords = ["sort", "search", "tree", "graph", "queue", "stack"]
        has_algorithm = any(kw in words for kw in algorithm_keywords)
        
        # 🔥 NEW: Programming language indicators
        language_keywords = ["python", "javascript", "java", "c++", "html", "css", "react", "typescript"]
        has_language = any(lang in q for lang in language_keywords)
        
        # 🔥 NEW: Code-related phrases (for GENERATION, not explanation)
        code_generation_phrases = ["code for", "code to", "simple code", "example code", "sample code", "write code"]
        has_code_phrase = any(phrase in q for phrase in code_generation_phrases)
        
        starts_with_anaphora = len(words) > 0 and words[0] in self.ANAPHORA
        
        # 🔥 FIXED: Only trigger if it's clearly a code GENERATION request
        if has_verb and not starts_with_anaphora:
            if has_object or has_algorithm or has_language or has_code_phrase:
                confidence = 0.95 if (has_object and has_language) else 0.90
                reasoning = "Code verb"
                if has_language:
                    reasoning += f" + programming language"
                if has_object:
                    reasoning += " + code object"
                if has_code_phrase:
                    reasoning += " + code phrase"
                
                return IntentClassification(
                    intent=QueryIntent.CODE,
                    confidence=confidence,
                    reasoning=reasoning,
                    needs_pdf=False,
                    needs_context=False
                )
        
        return None

# ============================================================
# Re-initialize the global classifier
# ============================================================
INTENT_CLASSIFIER = ImprovedIntentClassifier(use_embeddings=True)
print("✅ Enhanced intent classifier with comprehensive follow-up detection initialized")
# ============================================================
# IMPROVED INTENT CLASSIFIER
# ============================================================



def classify_intent_fast(query: str, chat_history: List = None) -> str:
    """
    ✅ FIXED: Now uses ImprovedIntentClassifier
    Backward-compatible wrapper that returns string intent
    """
    result = INTENT_CLASSIFIER.classify(query, chat_history)
    return result.intent.value  # Returns "greeting", "followup", etc.


def resolve_with_context(intent, previous_intent=None):
    if intent == "question" and previous_intent in {"explanation", "code"}:
        return "followup"
    return intent


def build_conversation_context(chat_history, max_turns=4, verbose=True):
    """
    Build comprehensive context from recent conversation history.
    🔥 ENHANCED: Preserves code blocks completely for follow-up questions
    
    Args:
        chat_history: List of conversation exchanges
        max_turns: Maximum number of previous exchanges to include
        verbose: Include full AI responses (True) or summaries (False)
    
    Returns:
        Formatted context string with previous Q&A
    """
    if not chat_history:
        return ""
    
    # Get recent exchanges
    recent = chat_history[-max_turns:]
    context_parts = []
    
    for i, entry in enumerate(recent, 1):
        user_msg = entry.get("user", "")
        ai_msg = entry.get("ai", "")
        
        if not user_msg:
            continue
        
        # Build context block
        block = f"**Previous Exchange {i}:**\n"
        block += f"User asked: {user_msg}\n\n"
        
        if ai_msg:
            if verbose:
                # 🔥 CRITICAL FIX: Check if response contains code
                has_code = "```" in ai_msg
                
                if has_code:
                    # For code responses, preserve EVERYTHING
                    # Extract code blocks with better preservation
                    import re
                    
                    # Find all code blocks
                    code_blocks = re.findall(r'```[\w]*\n(.*?)```', ai_msg, re.DOTALL)
                    
                    if code_blocks:
                        block += "Assistant provided this code:\n\n"
                        
                        # Include ALL code blocks
                        for code in code_blocks:
                            block += f"```\n{code.strip()}\n```\n\n"
                        
                        # Add explanation (non-code parts)
                        text_parts = re.split(r'```[\w]*\n.*?```', ai_msg, flags=re.DOTALL)
                        explanation = " ".join(part.strip() for part in text_parts if part.strip())
                        
                        if explanation and len(explanation) > 20:
                            # Keep full explanation if it's about the code
                            # Only truncate if it's VERY long (>800 chars)
                            if len(explanation) > 800:
                                explanation = explanation[:800] + "..."
                            block += f"With explanation: {explanation}\n\n"
                    else:
                        # Code markers but no blocks found - include full response
                        block += f"Assistant responded:\n{ai_msg}\n\n"
                
                else:
                    # No code - standard truncation
                    max_chars = 500
                    clean_ai = ai_msg
                    
                    # Remove source citations
                    clean_ai = re.sub(r'📄 \*\*Source.*?\n', '', clean_ai, flags=re.MULTILINE)
                    clean_ai = re.sub(r'🌐 \*\*Sources.*?\n', '', clean_ai, flags=re.MULTILINE)
                    clean_ai = re.sub(r'\n\n---\n\n', '\n', clean_ai)
                    
                    # Remove excessive emojis
                    clean_ai = re.sub(r'[\U0001F300-\U0001F9FF]{2,}', '', clean_ai)
                    
                    # Truncate intelligently
                    if len(clean_ai) > max_chars:
                        truncated = clean_ai[:max_chars]
                        last_period = max(
                            truncated.rfind('.'),
                            truncated.rfind('!'),
                            truncated.rfind('?')
                        )
                        
                        if last_period > max_chars * 0.6:
                            clean_ai = truncated[:last_period + 1]
                        else:
                            clean_ai = truncated + "..."
                    
                    block += f"Assistant responded: {clean_ai}\n\n"
            else:
                # Summary mode
                summary = ai_msg[:100].strip()
                if len(ai_msg) > 100:
                    summary += "..."
                block += f"Assistant: {summary}\n\n"
        
        context_parts.append(block)
    
    if context_parts:
        header = "**CONVERSATION CONTEXT** (for reference):\n" + "="*60 + "\n\n"
        footer = "\n" + "="*60 + "\n"
        return header + "\n".join(context_parts) + footer
    
    return ""



def build_code_aware_context(chat_history: List, max_chars: int = 2500) -> str:  # ✅ Increased limit
    """
    Build context that PRESERVES code blocks for follow-up questions.
    ✅ FIXED: Now includes multiple previous exchanges
    """
    if not chat_history:
        return ""
    
    # ✅ NEW: Include last 3 exchanges instead of just 1
    relevant_history = chat_history[-3:]  # Get last 3 exchanges
    
    context = "**CONVERSATION HISTORY:**\n\n"
    
    for i, entry in enumerate(relevant_history, 1):
        user_msg = entry.get("user", "")
        ai_msg = entry.get("ai", "")
        
        if not user_msg or not ai_msg:
            continue
        
        context += f"**Exchange {i}:**\n"
        context += f"User: {user_msg}\n\n"
        
        # Check if response contains code
        has_code = "```" in ai_msg
        
        if has_code:
            # Extract and preserve code blocks
            import re
            code_blocks = re.findall(r'```[\w]*\n(.*?)```', ai_msg, re.DOTALL)
            
            if code_blocks:
                context += "Assistant provided code:\n\n"
                for code in code_blocks:
                    context += f"```\n{code.strip()}\n```\n\n"
                
                # Add explanation
                text_parts = re.split(r'```[\w]*\n.*?```', ai_msg, flags=re.DOTALL)
                explanation = " ".join(part.strip() for part in text_parts if part.strip())
                
                if explanation and len(explanation) > 20:
                    if len(explanation) > 300:
                        explanation = explanation[:300] + "..."
                    context += f"Explanation: {explanation}\n\n"
        else:
            # No code - truncate text response
            truncated = ai_msg[:400] + "..." if len(ai_msg) > 400 else ai_msg
            context += f"Assistant: {truncated}\n\n"
        
        context += "---\n\n"
    
    context += "**Current question refers to the above conversation.**\n"
    
    # Truncate if too long
    if len(context) > max_chars:
        # Keep most recent exchange fully
        recent_exchange_marker = f"**Exchange {len(relevant_history)}:**"
        marker_pos = context.rfind(recent_exchange_marker)
        
        if marker_pos > 0:
            # Keep everything from the last exchange
            context = context[:1000] + "\n\n[...earlier exchanges truncated...]\n\n" + context[marker_pos:]
    
    return context

def build_memory_context():
    """
    Enhanced version that uses UserManager.
    """
    try:
        # Get user-specific context
        user_context = build_user_context_for_prompt()

        # Get interaction stats
        count = user_manager.get_user_data("interaction_stats.total_questions", 0)

        parts = []

        if user_context:
            parts.append(user_context)

        if count > 5:
            parts.append(f"This is question #{count} in this conversation.")

        return "\n\n".join(parts) if parts else ""

    except Exception as e:
        debug_log(f"Error building memory context: {e}")
        return ""

# ----------------------------
# Clean LLM output
# ----------------------------
def clean_response(text):
    if not text:
        return text

    original_text = text

    bad_patterns = [
        r"How's it going.*",
        r"Hope you.*",
        r"How about you.*",
        r"What about you.*",
        r"Anything.*\?",
        r"How are you.*",
        r"What's on your mind.*",
        r"Response:.*",
        r"User:.*",
        r"Assistant:.*",
        r"Human:.*",
    ]
    for pattern in bad_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    lines = text.split("\n")
    clean = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if any(
            line_stripped.lower().startswith(p)
            for p in [
                "q:",
                "question:",
                "example:",
                "for instance,",
                "here's an example",
                "try this:",
                "exercise:",
                "response:",
                "user:",
                "assistant:",
            ]
        ):
            break
        if any(
            w in line_stripped.lower()
            for w in [
                "would you",
                "do you want",
                "shall i",
                "can i help",
                "any questions",
                "how about you",
                "what about you",
                "hope you",
            ]
        ):
            break
        clean.append(line)

    result = "\n".join(clean).strip()

    if not result or len(result) < 10:
        return original_text.strip()

    if result and result[-1] not in ".!?\"'":
        last_period = max(result.rfind("."), result.rfind("!"), result.rfind('"'))
        if last_period > len(result) // 2:
            result = result[: last_period + 1]

    sentences = result.split(".")
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not s.endswith("?") and not any(
            w in s.lower() for w in ["would you", "do you want", "how about"]
        ):
            clean_sentences.append(s)
    final_result = ". ".join(clean_sentences).strip()

    if not final_result or len(final_result) < 10:
        return original_text.strip()

    return final_result


def clean_leaked_prompt(text):
    """Remove any leaked prompt formatting from the response"""
    # Remove common prompt artifacts
    patterns = [
        r"\[INST\].*?\[/INST\]",
        r"<<SYS>>.*?<</SYS>>",
        r"<\|.*?\|>",
        r"### Instruction:.*?### Response:",
        r"^(Question|User|Human|Assistant|System):.*?(?=\n\n|\Z)",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

    return text.strip()


def post_process_llm_output(text: str) -> str:
    """
    Enhanced post-processing that includes ALL formatting fixes.
    This is a comprehensive wrapper around format_llm_response.
    """
    if not text:
        return text

    # Apply master formatting
    text = format_llm_response(text)

    # Additional fixes for edge cases

    # Fix missing spaces after periods in lists
    text = re.sub(r"(\d+\.)([A-Z])", r"\1 \2", text)

    # Ensure newlines after colons that start lists
    text = re.sub(r":\s*(\d+\.)", r":\n\n\1", text)

    # Fix bullet points that got concatenated
    text = re.sub(r"([-•*]\s+[^-•*\n]+?)\s+([-•*])", r"\1\n\2", text)

    return text


# ============================================================
# MODEL FORMAT DETECTION
# ============================================================


def detect_model_format(model_path):
    """Return a standardized format identifier based on the model filename."""
    name = model_path.lower()

    # --- Mistral ---
    if "mistral" in name:
        if "instruct" in name:
            return "mistral-instruct"
        else:
            return "mistral-base"

    # --- CodeLlama ---
    if "codellama" in name:
        if "instruct" in name:
            return "llama2"  # CodeLlama-Instruct uses Llama-2 Chat format
        else:
            return "base"  # CodeLlama base model (not chat)

    # --- Llama 2 ---
    if "llama-2" in name or "llama2" in name:
        if "chat" in name:
            return "llama2"
        return "base"

    # --- Llama 3 ---
    if "llama-3" in name or "llama3" in name:
        return "llama3"

    # --- Phi models ---
    if "phi" in name:
        return "phi"

    # --- Alpaca ---
    if "alpaca" in name:
        return "alpaca"

    # --- ChatML models (e.g., Falcon, GPT4All-J) ---
    if "chatml" in name:
        return "chatml"

    return "universal"


# ============================================================
# PROMPT BUILDERS FOR ALL MODELS
# ============================================================


def build_prompt_with_format(system_msg, user_msg, context="", model_format=None):
    global MODEL_FORMAT
    fmt = model_format or MODEL_FORMAT

    # ------------------------------------------------------------
    # 1. MISTRAL BASE (NO CHAT FORMAT)
    # ------------------------------------------------------------
    if fmt == "mistral-base":
        prompt = f"{system_msg}\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += user_msg
        return prompt

    # ------------------------------------------------------------
    # 2. MISTRAL INSTRUCT (REQUIRES <s>[INST] )
    # ------------------------------------------------------------
    if fmt == "mistral-instruct":
        if context:
            return f"[INST] {system_msg}\n\n{context}\n\n{user_msg} [/INST]"
        return f"[INST] {system_msg}\n\n{user_msg} [/INST]"

    # ------------------------------------------------------------
    # 3. LLAMA-3 CHATML-V3 FORMAT
    # ------------------------------------------------------------
    if fmt == "llama3":
        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
        )
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        return prompt

    # ------------------------------------------------------------
    # 4. LLAMA-2 CHAT + CODELLAMA-INSTRUCT
    # (NO <s>, llama.cpp auto-BOS)
    # ------------------------------------------------------------
    if fmt == "llama2":
        prompt = "[INST] <<SYS>>\n" f"{system_msg}\n" "<</SYS>>\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_msg} [/INST]"
        return prompt

    # ------------------------------------------------------------
    # 5. BASE MODELS (LLAMA BASE, CODELLAMA BASE)
    # ------------------------------------------------------------
    if fmt == "base":
        prompt = f"{system_msg}\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += user_msg
        return prompt

    # ------------------------------------------------------------
    # 6. PHI 2/3 CHAT FORMAT
    # ------------------------------------------------------------
    if fmt == "phi":
        prompt = f"<|system|>\n{system_msg}<|end|>\n<|user|>\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_msg}<|end|>\n<|assistant|>\n"
        return prompt

    # ------------------------------------------------------------
    # 7. ALPACA FORMAT
    # ------------------------------------------------------------
    if fmt == "alpaca":
        prompt = "### Instruction:\n" + system_msg + "\n\n### Input:\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_msg}\n\n### Response:\n"
        return prompt

    # ------------------------------------------------------------
    # 8. CHATML FORMAT
    # ------------------------------------------------------------
    if fmt == "chatml":
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    # ------------------------------------------------------------
    # 9. UNIVERSAL FALLBACK (SAFE)
    # ------------------------------------------------------------
    prompt = f"System: {system_msg}\n\n"
    if context:
        prompt += f"Context: {context}\n\n"
    prompt += f"User: {user_msg}\n\nAssistant:"
    return prompt


# -------------------------------
#   Main Tutor Function
# -------------------------------
def rag_pdf_tutor(
    query,
    chunks,
    sources,
    chunk_emb,
    emb_model,
    llm,
    allow_web=True,
    max_code_tokens=8000,
    search_mode="auto",
):
    """
    Updated RAG function with proper formatting.
    Replace your existing rag_pdf_tutor with this.
    """
    if RAG_ENGINE is None:
        return None, None, False

    try:
        cleaned_query = query.replace("use pdf:", "").strip()

        queries = expand_query(llm, cleaned_query)
        debug_log(f"🧠 Expanded queries ({len(queries)}): {queries}")

        results = []
        search_context = ""

        all_results = []
        for q in queries:
            ctx, res = RAG_ENGINE.build_context(
                q,
                top_k=5,
                use_hybrid=True,
                use_reranking=False,
            )

            if ctx and not search_context:
                search_context = ctx

            if res:
                all_results.extend(res)

        results = all_results

        if not all_results:
            return None, None, False

        results = _normalize_searchresult_scores(all_results)
        results = apply_freshness_boost(results)
        results = enforce_source_diversity(results)

        # Final reranking pass
        results = RAG_ENGINE.rerank(cleaned_query, results)

        pdf_context = ctx
        pdf_source = ", ".join(sorted(set(r.document.source for r in results)))

        if not results:
            return None, None, False

        pdf_context = ctx
        pdf_source = ", ".join(sorted(set(r.document.source for r in results)))

        results = _normalize_searchresult_scores(results)
        pdf_score = sum(r.score for r in results) / len(results)

        prompt = build_prompt(persona, cleaned_query, pdf_context, use_gemini=use_gemini)

        # Get LLM response
        llm_out = llm(
            prompt,
            max_tokens=500,
            temperature=0.4,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["User:", "Question:"],
        )

        # ✅ NEW WAY: Single unified processing
        answer = process_llm_answer(llm_out, is_code=False)
        answer = clean_leaked_prompt(answer)  # Additional safety

        return (
            f"📄 *From PDF ({pdf_source}) via RAG*\n\n{answer}",
            float(pdf_score),
            True,
        )

    except Exception as e:
        debug_log(f"⚠️ RAG failed: {e}")
        return None, None, False


# yan.py
# ==================== HYBRID RAG HELPER FUNCTIONS ====================


def try_hybrid_rag_search(query: str, top_k: int = 5):
    """
    Try hybrid RAG search with error handling.
    Returns: (context, results, success)
    """
    if HYBRID_RAG is None:
        debug_log("⚠️ Hybrid RAG not available")
        return "", [], False

    try:
        context, results = HYBRID_RAG.get_context_for_llm(
            query, top_k=top_k, max_length=2000
        )

        if context and results:
            debug_log(f"✅ Hybrid RAG found {len(results)} results")
            return context, results, True
        else:
            debug_log("⚠️ Hybrid RAG returned no results")
            return "", [], False

    except Exception as e:
        debug_log(f"❌ Hybrid RAG search error: {e}")
        return "", [], False


def format_rag_response(answer: str, results: list) -> str:
    """Format RAG results with source attribution"""
    if not results:
        return answer

    sources_text = "\n\n📄 **Sources:**\n"
    for i, result in enumerate(results[:5], 1):
        sources_text += f"{i}. **{result.source_filename}** "
        sources_text += f"(Relevance: {result.similarity_score:.2f})\n"
        if result.topics:
            sources_text += f"   Topics: {', '.join(result.topics[:3])}\n"

    return answer + sources_text


def save_chat_to_database(
    user_query: str,
    ai_response: str,
    confidence: float,
    search_used: bool,
    sources: list = None,
):
    """Save chat interaction to database"""
    try:
        db.add_chat_message(
            user_id="default_user",
            user_message=user_query,
            ai_response=ai_response,
            confidence=confidence,
            search_used=search_used,
            sources=sources or [],
        )
    except Exception as e:
        debug_log(f"⚠️ Failed to save chat to database: {e}")

def try_rag_retrieval(query: str, top_k: int = 5):
    """
    Try RAG retrieval using new system
    Returns: (context, results, success)
    """
    if RAG_ADAPTER is None:
        debug_log("⚠️ RAG adapter not available")
        return "", [], False
    
    try:
        # Get context for LLM
        context, results = RAG_ADAPTER.get_context_for_llm(
            query,
            top_k=top_k,
            max_length=2000
        )
        
        if context and results:
            debug_log(f"✅ RAG found {len(results)} results")
            
            # Format for compatibility
            sources_list = [
                (r.chunk.source_name, r.chunk.chunk_id)
                for r in results
            ]
            
            return context, sources_list, True
        else:
            debug_log("⚠️ RAG returned no results")
            return "", [], False
    
    except Exception as e:
        debug_log(f"❌ RAG retrieval error: {e}")
        return "", [], False

def web_search(
    query: str,
    *,
    mode: str = "fast",   # "fast" | "deep"
    max_results: int = 6,
    force_refresh: bool = False,
):
    if not query or len(query.strip()) < 2:
        return "", []

    # Firewall
    allowed, error = WEB_FIREWALL.check_permission()
    if not allowed:
        return error, []

    # Normalize query (CRITICAL)
    norm_query = normalize_query(query)

    # Memory cache
    if not force_refresh:
        cached = _memory_cache.get(norm_query)
        if cached:
            return cached["context"], cached["results"]

    # RAG cache
    if not force_refresh and RAG_ADAPTER:
        try:
            cached = RAG_ADAPTER.get_cached_web_search(norm_query, max_age_days=7)
            if cached and len(cached) >= 2:
                context = _format_context(norm_query, cached)
                results = [(t, u) for t, u, _ in cached]
                _memory_cache[norm_query] = {
                    "context": context,
                    "results": results,
                }
                return context, results
        except Exception:
            pass

    # Fast HTTP search (ONE request)
    raw = fetch_duckduckgo_results(
        norm_query,
        max_results=max_results,
        fetch_previews=False,
    )

    if not raw:
        return "", []

    # Optional deep mode
    if mode == "deep":
        enriched = []
        for title, url, snippet in raw[:4]:
            content = fetch_page_preview(url, max_length=800) or snippet
            enriched.append((title, url, content))
    else:
        enriched = raw

    # Store in RAG
    if RAG_ADAPTER:
        try:
            RAG_ADAPTER.add_web_search_results(
                norm_query,
                enriched,
                confidence=0.7 if mode == "deep" else 0.65,
            )
        except Exception:
            pass

    context = _format_context(norm_query, enriched)
    results = [(t, u) for t, u, _ in enriched]

    _memory_cache[norm_query] = {
        "context": context,
        "results": results,
    }

    return context, results

def _format_context(query: str, results):
    if not results:
        return ""

    out = [
        f"Web Search Results for: '{query}'",
        "=" * 60,
        "",
    ]

    for i, (title, url, content) in enumerate(results, 1):
        out.append(f"[{i}] {title}")
        out.append(f"URL: {url}")
        if content:
            preview = content[:500] + "..." if len(content) > 500 else content
            out.append(f"Content: {preview}")
        out.append("")

    out.append("=" * 60)
    out.append("Answer using the results above.")

    return "\n".join(out)

# ============================================================
# FILENAME QUERY HANDLER - Add to yan.py
# ============================================================

import re
from typing import Optional, Tuple

def detect_filename_query(query: str) -> Optional[str]:
    """
    Detect if user is asking about a specific PDF file by name
    
    Returns:
        Filename if detected, None otherwise
    """
    query_lower = query.lower()
    
    # Pattern 1: "what does [filename.pdf] talk about"
    pattern1 = r"what (?:does|is in|about) ([a-zA-Z0-9_\-]+\.pdf)"
    match = re.search(pattern1, query_lower, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: "tell me about [filename.pdf]"
    pattern2 = r"tell me about ([a-zA-Z0-9_\-]+\.pdf)"
    match = re.search(pattern2, query_lower, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 3: "summarize [filename.pdf]"
    pattern3 = r"summarize ([a-zA-Z0-9_\-]+\.pdf)"
    match = re.search(pattern3, query_lower, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 4: Direct mention of .pdf file
    pattern4 = r"([a-zA-Z0-9_\-]+\.pdf)"
    match = re.search(pattern4, query_lower)
    if match and any(keyword in query_lower for keyword in ['what', 'about', 'talk', 'contain', 'cover', 'discuss']):
        return match.group(1)
    
    return None


def retrieve_by_filename(filename: str) -> Tuple[str, float, list]:
    """
    Retrieve all chunks from a specific PDF file
    
    Args:
        filename: Name of the PDF file
    
    Returns:
        (context_string, confidence_score, sources_list)
    """
    if not RAG_ADAPTER:
        debug_log("⚠️ RAG_ADAPTER not available")
        return "", 0.0, []
    
    try:
        cursor = RAG_ADAPTER.rag_db.conn.cursor()
        
        # Search for exact filename match (case-insensitive)
        cursor.execute("""
            SELECT chunk_id, content, source_name, confidence
            FROM knowledge_chunks
            WHERE source_type = 'pdf_document'
            AND LOWER(source_name) = LOWER(?)
            ORDER BY chunk_id
            LIMIT 10
        """, (filename,))
        
        rows = cursor.fetchall()
        
        if not rows:
            debug_log(f"❌ No chunks found for PDF: {filename}")
            
            # Try fuzzy match
            cursor.execute("""
                SELECT DISTINCT source_name
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
                AND LOWER(source_name) LIKE LOWER(?)
            """, (f"%{filename}%",))
            
            similar = cursor.fetchall()
            
            if similar:
                similar_names = [row['source_name'] for row in similar]
                debug_log(f"💡 Found similar PDFs: {similar_names}")
                return "", 0.0, []
            
            return "", 0.0, []
        
        debug_log(f"✅ Found {len(rows)} chunks for PDF: {filename}")
        
        # Build context from chunks
        context_parts = []
        sources = []
        
        for i, row in enumerate(rows, 1):
            context_parts.append(f"[Section {i}]\n{row['content']}")
            sources.append((row['source_name'], row['chunk_id']))
        
        context = f"Content from {filename}:\n\n" + "\n\n---\n\n".join(context_parts)
        
        # High confidence since we found exact file match
        confidence = 0.9
        
        return context, confidence, sources
        
    except Exception as e:
        debug_log(f"❌ Error retrieving by filename: {e}")
        import traceback
        traceback.print_exc()
        return "", 0.0, []

def quick_filename_patch():
    """
    Add this code block at the START of your pdf_tutor_enhanced function,
    right after the query validation checks
    """
    
    # 🔥 Quick filename detection
    detected_filename = detect_filename_query(query)
    
    if detected_filename and RAG_ADAPTER:
        context, confidence, file_sources = retrieve_by_filename(detected_filename)
        
        if context:
            # Generate answer using file content
            prompt = build_prompt(
                persona + f"\n\nSummarize what {detected_filename} covers:",
                query,
                context,
                use_gemini=use_gemini
            )
            
            # Detect Gemini request
            use_gemini_for_query, cleaned_query = detect_gemini_request(query)
            if use_gemini_for_query:
                query = cleaned_query  # Use cleaned query
                debug_log("🔷 User requested Gemini")
            
            llm_out = query_llm_smart(
                prompt, 
                max_tokens=500, 
                temperature=0.4,
                use_gemini=use_gemini_for_query
            )
            answer = process_llm_answer(llm_out)
            answer = humanize_response_sentient(answer, query, chat_history)
            
            final_answer = answer + f"\n\n📄 **Source:** {detected_filename}"
            save_message_to_history(query, final_answer)
            
            return final_answer, confidence, file_sources, False


def detect_permission_response(query: str) -> str:
    """
    Detect if user is granting/denying permission
    
    Returns: 'grant', 'deny', or 'unclear'
    """
    query_lower = query.lower().strip()
    
    # Grant keywords
    grant_phrases = [
        'yes', 'yeah', 'yep', 'sure', 'ok', 'okay',
        'permission granted', 'go ahead', 'proceed',
        'allow', 'approved', 'granted', 'do it'
    ]
    
    # Deny keywords
    deny_phrases = [
        'no', 'nope', 'nah', 'deny', 'denied',
        'permission denied', 'skip', 'cancel',
        "don't", 'stop', 'refuse'
    ]
    
    # Check for exact matches or phrases
    for phrase in grant_phrases:
        if query_lower == phrase or query_lower.startswith(phrase + ' '):
            return 'grant'
    
    for phrase in deny_phrases:
        if query_lower == phrase or query_lower.startswith(phrase + ' '):
            return 'deny'
    
    return 'unclear'

"""
STEP 4: Add this function to detect "use gemini:" prefix (add around line 5000-5100)
"""
def detect_gemini_request(query: str) -> tuple[bool, str]:
    """
    Detect if user wants to use Gemini for this query
    
    Args:
        query: User's query
        
    Returns:
        (use_gemini, cleaned_query)
    """
    query_lower = query.lower().strip()
    
    # Check for "use gemini:" prefix
    gemini_triggers = [
        "use gemini:",
        "gemini:",
        "ask gemini:",
        "with gemini:",
    ]
    
    for trigger in gemini_triggers:
        if query_lower.startswith(trigger):
            # Remove the trigger and return cleaned query
            cleaned_query = query[len(trigger):].strip()
            return True, cleaned_query
    
    return False, query

"""
STEP 5: Add this wrapper function to route between local LLM and Gemini
"""

def query_llm_smart(
    prompt: str,
    max_tokens: int = 8000,
    temperature: float = 0.4,
    use_gemini: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Smart LLM router - uses Gemini or local LLM based on preference
    
    Args:
        prompt: Input prompt
        max_tokens: Max tokens to generate (default 8000 to prevent cutoffs)
        temperature: Sampling temperature
        use_gemini: Force use of Gemini
        **kwargs: Additional parameters
        
    Returns:
        Response dict with 'choices' field
    """
    global gemini_llm, llm, USE_GEMINI
    
    # Determine which LLM to use
    should_use_gemini = use_gemini or USE_GEMINI
    
    if should_use_gemini:
        if gemini_llm is None:
            debug_log("⚠️ Gemini requested but not initialized, falling back to local LLM")
        else:
            try:
                debug_log("🔷 Using Gemini LLM")
                # ✅ NEW: Pass force_web_search parameter for auto-detection
                return gemini_llm(
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    force_web_search=kwargs.pop('force_web_search', None),  # Auto-detect current info needs
                    **kwargs
                )
            except Exception as e:
                debug_log(f"❌ Gemini error: {e}, falling back to local LLM")
    
    # Fall back to local LLM
    debug_log("🔹 Passing results to local LLM...")
    # Remove personality-only fields before calling LLM
    safe_kwargs = kwargs.copy()
    safe_kwargs.pop("personality_result", None)
    safe_kwargs.pop("original_query", None)

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **safe_kwargs
    )

    # ====== ENHANCE LLM RESULT WITH PERSONALITY ======
    try:
        if "personality_result" in kwargs:
            personality_result = kwargs["personality_result"]
            original_query = kwargs.get("original_query", "")

            if "choices" in result and len(result["choices"]) > 0:
                text = result["choices"][0].get("text", "")

                if personality_result.get("opener"):
                    text = personality_result["opener"] + text

                if personality_result.get("suggestion"):
                    text = text + "\n\n" + personality_result["suggestion"]

                #text = enhance_response(text, original_query)
                result["choices"][0]["text"] = text



    except Exception as e:
        debug_log(f"⚠️ Personality enhancement error: {e}")

    # ====== END PERSONALITY ENHANCEMENT ======

    return result

"""
STEP 7: Add command to enable/disable Gemini globally (optional)

Add this to your command processing (if you have a /command system)
"""

def process_gemini_commands(query: str) -> tuple[bool, str]:
    """
    Process Gemini-related commands
    
    Returns:
        (handled, response)
    """
    global USE_GEMINI, gemini_llm
    
    query_lower = query.lower().strip()
    
    if query_lower in ["/gemini on", "/enable gemini", "/use gemini"]:
        if gemini_llm is None:
            return True, "⚠️ Gemini not initialized. Set GEMINI_API_KEY environment variable first."
        USE_GEMINI = True
        return True, "✅ Gemini enabled for all queries. Use '/gemini off' to disable."
    
    elif query_lower in ["/gemini off", "/disable gemini", "/local llm"]:
        USE_GEMINI = False
        return True, "✅ Switched back to local LLM."
    
    elif query_lower in ["/gemini status", "/which model"]:
        if USE_GEMINI:
            status = "🔷 Currently using: **Gemini** (globally enabled)"
        else:
            status = "🔹 Currently using: **Local LLM**"
        
        gemini_status = "✅ Available" if gemini_llm else "❌ Not initialized"
        status += f"\n\nGemini status: {gemini_status}"
        return True, status
    
    return False, ""
# ============================================================
# LIGHT vs DEEP SEARCH IMPLEMENTATION
# ============================================================

def pdf_tutor_with_explicit_routing(
    query,
    chunks,
    sources,
    chunk_emb,
    emb_model,
    llm,
    allow_web=True,
    max_code_tokens=8000,
    search_mode="auto",
    thought_stream_callback=None,
    use_gemini=None,
):
    """
    Main query handler with explicit routing and universal follow-up support.
    
    REWRITTEN VERSION - Cleaned up, no duplicate follow-up logic.
    Follow-ups are handled by build_prompt() automatically in all modes.
    
    Args:
        query: User's question
        chunks: PDF text chunks (legacy, kept for compatibility)
        sources: PDF sources (legacy, kept for compatibility)
        chunk_emb: Chunk embeddings (legacy, kept for compatibility)
        emb_model: Embedding model
        llm: Language model instance
        allow_web: Whether web search is enabled
        max_code_tokens: Max tokens for code generation
        search_mode: "auto", "llm_only", "pdf_only", "web_only", "pdf_web_llm"
    
    Returns:
        Tuple: (answer, confidence, result_sources, search_used)
    """

    if use_gemini is None:
        use_gemini = USE_GEMINI
    # ============================================================
    # THOUGHT PROCESS TRACKING - REAL-TIME STREAMING
    # ============================================================
    def add_thought(step: str, icon: str = ""):
        """Stream reasoning steps in real-time to frontend"""
        formatted_step = f"{icon} {step}"
        debug_log(f"💭 THOUGHT: {step}")
        
        # Stream immediately if callback provided
        if thought_stream_callback:
            try:
                thought_stream_callback(formatted_step)
            except Exception as e:
                debug_log(f"⚠️ Thought streaming error: {e}")
    
    # Initial thought
    add_thought(f"Received query: '{query[:60]}{'...' if len(query) > 60 else ''}'", "🔍")
    # ============================================================
    # STEP 1: INITIALIZATION - Load chat history from chat_history.json
    # ============================================================
    try:
        chat_history, chat_count = load_chat_history()  # Load from chat_history.json
        persistent_data["chat_history"] = chat_history
        
        debug_log(f"📚 FUNCTION START: Loaded {len(chat_history)} messages from chat_history.json")
        
        if chat_history:
            last_msg = chat_history[-1]
            debug_log(f"   Last message: '{last_msg.get('user', '')[:50]}'")
    except Exception as e:
        debug_log(f"❌ Failed to load chat history: {e}")
        chat_history = []
        debug_log(f"   Using empty history")

    # Track history status
    if chat_history:
        add_thought(f"Loaded conversation history ({len(chat_history)} previous messages)", "📚")
    else:
        add_thought("Starting fresh conversation", "🆕")
    # ============================================================
    # STEP 2: VALIDATION
    # ============================================================
    cleaned_query = (query or "").strip()
    
    if not cleaned_query:
        return "Hey — what's on your mind?", 0.0, [], False

    
    if llm is None:
        return "❌ AI model not loaded.", 0.0, [], False

    # New personality system integrates automatically
    comp_response = None
    is_compliment = False

    # Generate suggestions
    #suggestion = get_suggestion(cleaned_query)
    #if suggestion:
    #    add_thought("Preparing helpful suggestion", "💡")


    query_lower = cleaned_query.lower()
    
    add_thought("Analyzing query", "🔍")
    # ============================================================
    # STEP 3: USERNAME EXTRACTION (Priority 1)
    # ============================================================
    extracted_name = extract_and_save_username(query)
    if extracted_name:
        add_thought(f"Detected username: {extracted_name}", "👤")
        time.sleep(1.0)
        add_thought("Response ready", "✅")
        time.sleep(2.0)
        response = f"Nice to meet you, {extracted_name}. I'll remember that. 👋"
        # Save to chat_history.json
        chat_history.append({"user": query, "ai": response, "timestamp": datetime.now().isoformat()})
        save_chat_history(chat_history, len(chat_history))
        # Update user stats
        user_manager.add_to_chat_history(query, response)
        return response, 1.0, [], False
    
    # ============================================================
    # STEP 4: GREETING CHECK (Priority 2)
    # ============================================================
    greeting_patterns = [
        "hi", "hello", "hey", "sup", "good morning", "good afternoon",
        "good evening", "how are you", "how r u",
        "what's up", "whats up", "wassup"
    ]
    
    query_words = query_lower.replace("?", "").replace("!", "").split()
    
    if (
        query_lower in greeting_patterns or
        (len(query_words) <= 3 and any(g in query_lower for g in greeting_patterns))
    ):
        user_name = user_manager.get_user_data("name")
        name_part = f" {user_name}" if user_name else ""
        
        # Derive the real time-of-day period from the clock
        _greet_hour = datetime.now().hour
        if 5 <= _greet_hour < 12:
            _tod_label = "morning"
        elif 12 <= _greet_hour < 18:
            _tod_label = "afternoon"
        elif 18 <= _greet_hour < 22:
            _tod_label = "evening"
        else:
            _tod_label = "night"

        # Time-aware responses for each period
        _tod_responses = {
            "morning":   f"Morning{name_part}. What are we getting into today?",
            "afternoon": f"Hey{name_part}. What's up?",
            "evening":   f"Evening{name_part}. What's on your mind?",
            "night":     f"Still up{name_part}? What are we doing?",
        }

        greetings_map = {
            "how are you": f"Doing well{name_part}, thanks for asking. What's going on?",
            "good morning":   _tod_responses.get("morning",   _tod_responses[_tod_label]),
            "good afternoon": _tod_responses.get("afternoon", _tod_responses[_tod_label]),
            "good evening":   _tod_responses.get("evening",   _tod_responses[_tod_label]),
            "hi":    _tod_responses[_tod_label],
            "hello": _tod_responses[_tod_label],
            "hey":   _tod_responses[_tod_label],
        }
        greeting_response = None
        for pattern, response in greetings_map.items():
            if pattern in query_lower:
                greeting_response = response
                break

        if not greeting_response:
            greeting_response = _tod_responses[_tod_label]
        add_thought("Detected greeting - responding warmly", "👋")
        time.sleep(1.0)
        add_thought("Response ready", "✅")
        time.sleep(1.0)
        save_message_to_history(query, greeting_response)
        return greeting_response, 1.0, [], False
    
    # ============================================================
    # STEP 4.5: GEMINI COMMAND DETECTION (Priority 2.5)
    # ============================================================
    
    # Initialize Gemini flag
    use_gemini_for_query = USE_GEMINI  # Start with global setting

    
    # Check for Gemini slash commands first (/gemini on, /gemini off, etc.)
    gemini_command_handled, gemini_response = process_gemini_commands(cleaned_query)
    if gemini_command_handled:
        add_thought("Response ready", "✅")
        time.sleep(1.0)
        save_message_to_history(query, gemini_response)
        return gemini_response, 1.0, [], False

    
    # Check for "use gemini:" prefix in query
    detected_gemini, cleaned_query_no_prefix = detect_gemini_request(cleaned_query)
    
    if detected_gemini:
        debug_log(f"🔷 GEMINI REQUEST DETECTED: '{cleaned_query_no_prefix}'")
        add_thought("Routing to Gemini API for processing", "🔷")
        use_gemini_for_query = True
        
        # Update the query to the cleaned version (without "use gemini:" prefix)
        cleaned_query = cleaned_query_no_prefix
        query_lower = cleaned_query.lower()
        
        # Check if Gemini is initialized
        if gemini_llm is None:
            debug_log("⚠️ Gemini requested but not initialized")
            use_gemini_for_query = False
        else:
            debug_log("✅ Gemini ready - will use Gemini API")
    # ============================================================
    # STEP 5: EXPLICIT WEB SEARCH COMMANDS (Priority 3)
    # ============================================================
    web_search_triggers = [
        "search:", "web search:", "look up:",
        "deep search:", "detailed search:", "comprehensive search:"
    ]
    
    deep_search_triggers = [
        "deep search:", "detailed search:", "comprehensive search:"
    ]
    
    news_rag_triggers = [
        "news:", "tech news:", "latest:", "recent:", "current events:",
        "what's new in", "what is new in", "what happened with",
        "latest news", "recent news", "current news",
        "this week in", "today in tech", "trending in",
    ]

    is_news_query = any(query_lower.startswith(t) for t in news_rag_triggers) or \
                    any(t in query_lower for t in [
                        "latest news", "recent news", "tech news",
                        "what's happening in", "current developments",
                        "new release", "just announced", "newly released",
                        "this week in tech", "recently announced",
                        "any news", "news on", "news about",
                        "updates on", "update on", "what's going on with",
                        "anything on", "anything about", "anything new",
                        "fill me in on", "catch me up on",
                        "news briefing", "news summary", "news roundup",
                        "what's in the news", "give me the news",
                    ])

    is_web_search = any(query_lower.startswith(t) for t in web_search_triggers)
    is_deep_search = any(query_lower.startswith(t) for t in deep_search_triggers)

    if is_news_query and gemini_llm:
        # Strip trigger prefix if present
        news_query = cleaned_query
        for prefix in news_rag_triggers:
            if query_lower.startswith(prefix):
                news_query = cleaned_query[len(prefix):].strip()
                break

        # If no topic extracted, use broad tech query
        if not news_query or news_query == cleaned_query:
            news_query = "latest tech news today"

        add_thought(f"Fetching live news via Gemini: '{news_query[:50]}'", "📰")
        debug_log(f"📰 NEWS via Gemini triggered: '{news_query}'")

        try:
            news_prompt = (
                f"Give me a current tech news briefing on: {news_query}\n\n"
                f"Format:\n"
                f"- Group by topic (AI, Security, Gadgets, Business etc.)\n"
                f"- Each story: one sentence + source name\n"
                f"- Only real current news — no filler\n"
                f"- Do not open with a greeting\n"
                f"- Start directly with the first topic heading"
            )

            response = gemini_llm(
                news_prompt,
                max_tokens=2000,
                temperature=0.3,
                top_p=0.9,
                force_web_search=True,  # Force Gemini to search the web
            )

            answer = response["choices"][0]["text"].strip()

            if answer:
                answer = f"📰 **Live Tech News** (via Gemini)\n\n{answer}"
                save_message_to_history(query, answer)
                add_thought("Live news briefing delivered via Gemini", "📰")
                return answer, 0.90, [], False
            else:
                debug_log("📰 Gemini returned empty news response — falling through")

        except Exception as e:
            debug_log(f"❌ Gemini news error: {e} — falling through")

    if is_web_search:
        if not allow_web:
            error_response = (
                "🚫 Web search is disabled.\n\n"
                "To use web search:\n"
                "1. Open sidebar (☰ button)\n"
                "2. Enable 'Allow Web Search'\n"
                "3. Try again"
            )
            save_message_to_history(query, error_response)
            return error_response, 0.0, [], False
        
        allowed, firewall_error = WEB_FIREWALL.check_permission()
        if not allowed:
            save_message_to_history(query, firewall_error)
            return firewall_error, 0.0, [], False

        
        # Strip trigger prefix
        search_query = cleaned_query
        for prefix in web_search_triggers:
            if query_lower.startswith(prefix):
                search_query = cleaned_query[len(prefix):].strip()
                break
        
        search_type = "Deep web search" if is_deep_search else "Web search"
        add_thought(f"{search_type} initiated for: '{search_query[:50]}'", "🌐")
        
        debug_log(
            f"{'🔬 DEEP' if is_deep_search else '🔎'} SEARCH triggered: '{search_query}'"
        )
        
        try:
            if is_deep_search:
                context, results = run_deep_web_search(
                    search_query,
                    max_results=5,
                    read_full_pages=True
                )
                confidence = 0.85
                label = "Deep Search"
                max_tokens = 800
            else:
                context, results = run_web_search_protected(
                    search_query,
                    debug=True
                )
                confidence = 0.75
                label = "Web Search"
                max_tokens = 500
            
            if not context or not results:
                add_thought("No web results found", "⚠️")
                msg = f"No results found for '{search_query}'."
                save_message_to_history(query, msg)
                return msg, 0.0, [], True
            
            add_thought(f"Retrieved {len(results)} web results", "✅")
            time.sleep(2.0)
            add_thought("Processing web results...")
            
            # Cache results in RAG
            if RAG_ADAPTER and results:
                normalized = []
                for r in results:
                    if len(r) == 3:
                        normalized.append(r)
                    elif len(r) == 2:
                        normalized.append((r[0], r[1], ""))
                
                if normalized:
                    RAG_ADAPTER.add_web_search_results(
                        search_query,
                        normalized,
                        confidence=confidence
                    )
            
            # Build prompt with web context
            prompt = build_prompt(
                persona + f"\n\nUse {label.lower()} results to answer accurately.",
                search_query,
                context,
                #use_gemini=use_gemini
            )
            
            llm_out = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.85,
                repeat_penalty=1.15,
                stop=["User:", "Question:"]
            )
            
            answer = process_llm_answer(llm_out, is_code=False)
            answer = humanize_response_sentient(answer, search_query, chat_history)
            
            # Add sources
            """
            sources_text = f"\n\n🌐 **Web Sources ({len(results)} results):**\n"
            for result in results[:5]:
                if len(result) >= 2:
                    title, url = result[0], result[1]
                    sources_text += f"• [{title}]({url})\n"
            """
            final_response = answer 
            add_thought("Generated answer from web sources", "✨")
            time.sleep(2.0)

            save_message_to_history(query, final_response)
            
            display_results = [(r[0], r[1]) for r in results[:8 if is_deep_search else 5]]
            
            return final_response, confidence, display_results, True
            
        except Exception as e:
            debug_log(f"❌ Web search error: {e}")
            import traceback
            traceback.print_exc()
            msg = f"Web search encountered an error: {str(e)}"
            save_message_to_history(query, msg)
            return msg, 0.0, [], True

    
    # ============================================================
    # STEP 6: PERMISSION MANAGER CHECK (Priority 4)
    # ============================================================
    #add_thought("Processing user's permission response", "🔐")
    if WEB_PERMISSION_MANAGER.has_pending_request():
        # User granted permission
        if any(phrase in query_lower for phrase in [
            'yes', 'yeah', 'yep', 'sure', 'ok', 'okay',
            'permission granted', 'go ahead', 'proceed',
            'allow', 'approved', 'granted'
        ]):
            permission_data = WEB_PERMISSION_MANAGER.grant_permission()
            original_query = permission_data["query"]
            
            debug_log(f"✅ Permission granted for: {original_query}")
            
            try:
                searching_msg = "🔍 **Searched the web for current information...**\n\n"
                
                context, results = run_web_search_protected(original_query, debug=True)
                
                if not context or not results:
                    add_thought("Web search returned no results - using general knowledge", "⚠️")
                    fallback_answer = (
                        searching_msg + 
                        "I couldn't find current information. Using my general knowledge instead."
                    )
                    save_message_to_history(original_query, fallback_answer)
                    return fallback_answer, 0.5, [], False
                
                add_thought(f"Retrieved {len(results)} web sources", "✅")
                
                # Build prompt with search results
                prompt = build_prompt(
                    persona + "\n\nYou have current web search results. Synthesize them to answer accurately.",
                    original_query,
                    context,
                    use_gemini=use_gemini
                )
                
                llm_out = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.85,
                    repeat_penalty=1.15,
                    stop=["User:", "Question:"]
                )
                
                answer = process_llm_answer(llm_out, is_code=False)
                answer = humanize_response_sentient(answer, original_query, chat_history)
                
                # Add sources
                """
                sources_text = "\n\n🌐 **Web Sources:**\n"
                for result in results[:5]:
                    if len(result) >= 2:
                        title, url = result[0], result[1]
                        sources_text += f"• [{title}]({url})\n"
                """
                full_response = searching_msg + answer 
                save_message_to_history(original_query, full_response)
                
                display_results = [(title, url) for title, url, *_ in results[:5]]
                

                add_thought("Generated web-enhanced response", "✨")
                #time.sleep(0.1)  # Small delay to ensure thought is queued
                return full_response, 0.8, display_results, True
                
            except Exception as e:
                debug_log(f"❌ Web search error: {e}")
                error_msg = f"Search error: {str(e)}"
                save_message_to_history(original_query, error_msg)
                return error_msg, 0.0, [], False

        
        # User denied permission
        elif any(phrase in query_lower for phrase in [
            'no', 'nope', 'nah', 'deny', 'denied',
            'permission denied', 'skip', 'cancel', "don't"
        ]):
            permission_data = WEB_PERMISSION_MANAGER.deny_permission()
            original_query = permission_data["query"]
            
            debug_log(f"⛔ Permission denied for: {original_query}")
            
            # Answer using LLM knowledge only
            response_msg = "👍 **Understood.** Using my general knowledge instead.\n\n"
            
            memory_context = build_memory_context_with_name(original_query)
            prompt = build_prompt(persona, original_query, memory_context, use_gemini=use_gemini)
            
            llm_out = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.85,
                repeat_penalty=1.15,
                stop=["User:", "Question:"]
            )
            
            answer = process_llm_answer(llm_out, is_code=False)
            answer = humanize_response_sentient(answer, original_query, chat_history)
            
            full_response = response_msg + answer + "\n\n⚠️ *Note: This information may not be current.*"
            
            save_message_to_history(original_query, full_response)
            
            add_thought("Generated response using general knowledge only", "🤖")
            #time.sleep(0.1)  # Small delay to ensure thought is queued
            return full_response, 0.5, [], False
        # User said something else - clear pending and continue
        else:
            debug_log("ℹ️ User response unclear - clearing pending permission")
            WEB_PERMISSION_MANAGER.clear_pending()
    
    # ============================================================
    # STEP 7: UPDATE STATS (Priority 5)
    # ============================================================
    try:
        current_count = user_manager.get_user_data("interaction_stats.total_questions", 0)
        user_manager.update_user_stat("total_questions", current_count + 1)
        user_manager.update_user_stat("last_active", datetime.now().isoformat())
    except Exception as e:
        debug_log(f"⚠️ Stats update failed: {e}")
    
    # ============================================================
    # STEP 8: SPECIAL CASE - Filename Query (Priority 6)
    # ============================================================
    detected_filename = detect_filename_query(query)
    
    if detected_filename and RAG_ADAPTER:
        add_thought(f"Searching for specific file: {detected_filename}", "📄")
        debug_log(f"🎯 Detected filename query: {detected_filename}")
        
        context, confidence, file_sources = retrieve_by_filename(detected_filename)
        
        if context:
            add_thought(f"Found and retrieved content from {detected_filename}", "✅")
            debug_log(f"✅ Retrieved content from {detected_filename}")
            
            memory_context = build_memory_context_with_name(query)
            full_context = f"{memory_context}\n\n{context}" if memory_context else context
            
            prompt = build_prompt(
                persona + f"\n\nYou have the complete content of {detected_filename}. Summarize what it covers.",
                query,
                full_context,
                use_gemini=use_gemini
            )
            
            llm_out = llm(
                prompt, 
                max_tokens=500, 
                temperature=0.4,
                top_p=0.9, 
                stop=["User:", "Question:"]
            )
            
            answer = process_llm_answer(llm_out, is_code=False)
            answer = humanize_response_sentient(answer, query, chat_history)
            
            sources_text = f"\n\n📄 **Source:** {detected_filename}\n"
            sources_text += f"   Retrieved {len(file_sources)} sections from this document"
            
            final_answer = answer + sources_text
            save_message_to_history(query, final_answer)
            
            save_message_to_history(query, final_answer)
            add_thought("Generated file-specific response", "✨")
            time.sleep(0.2)
            return final_answer, confidence, file_sources, False
        else:
            # File not found
            available_pdfs = get_pdf_list()
            
            if available_pdfs:
                error_response = f"I couldn't find '{detected_filename}' in the database.\n\n"
                error_response += f"📚 **Available PDFs:**\n"
                for pdf in available_pdfs[:5]:
                    error_response += f"  • {pdf['filename']}\n"
                
                if len(available_pdfs) > 5:
                    error_response += f"  ... and {len(available_pdfs) - 5} more\n"
            else:
                error_response = f"I couldn't find '{detected_filename}' and there are no PDFs loaded.\n\n"
                error_response += "Upload PDFs using the sidebar to enable document search."
            
            save_message_to_history(query, error_response)

            add_thought(f"File '{detected_filename}' not found in database", "❌")
            #time.sleep(0.1)
            return error_response, 0.0, [], False
    # ============================================================
    # STEP 9: SPECIAL CASE - LLM Only Mode (Priority 7)
    # ============================================================
    if search_mode == "llm_only":
        add_thought("Passing results to local LLM...", "🤖")
        debug_log("🤖 LLM ONLY mode - pure AI knowledge")

        # build_prompt() handles follow-ups automatically!
        answer, confidence = _handle_llm_only(
            cleaned_query,
            llm,
            use_gemini=use_gemini_for_query
        )

        if use_gemini_for_query:
            add_thought("Generated response from Gemini", "✨")
            time.sleep(2.0)
        else:
            add_thought("Generated response from local LLM", "🧠")
            time.sleep(2.0)

        # Save with error handling
        try:
            if not user_manager.current_user:
                debug_log("⚠️ LLM ONLY: No user set! Creating default...")
                user_manager.set_current_user("default_user")
            
            save_success = save_message_to_history(query, answer)
            if save_success:
                debug_log(f"💾 LLM ONLY: Saved successfully")
            else:
                debug_log(f"⚠️ LLM ONLY: Save returned False")
        except Exception as e:
            debug_log(f"❌ LLM ONLY: Save exception: {e}")
            import traceback
            debug_log(traceback.format_exc())
        
        return answer, confidence, [], False
    # ============================================================
    # STEP 10: AUTO MODE ROUTING (Priority 8)
    # ============================================================
    if search_mode == "auto":
        router = AutoModeRouter(RAG_ADAPTER, llm, emb_model)
        
        debug_log(f"📚 Router analyzing with {len(chat_history)} messages")

        add_thought("Analyzing best source for this query", "🎯")
        decision = router.decide_source(cleaned_query, allow_web)
        
        # Check if permission is needed
        if decision.reason.startswith("REQUEST_PERMISSION:"):
            permission_reason = decision.reason.replace("REQUEST_PERMISSION:", "")
            
            debug_log(f"🔐 Requesting web search permission: {permission_reason}")
            
            permission_request = WEB_PERMISSION_MANAGER.request_permission(
                cleaned_query,
                permission_reason
            )
            
            save_message_to_history(cleaned_query, permission_request["message"])
            
            return permission_request["message"], 0.0, [], False
        
        add_thought(f"Selected source: {decision.primary_source.value}", "🎯")
        add_thought(f"Reasoning: {decision.reason}", "💡")
        
        debug_log(f"🎯 DECISION: {decision.primary_source.value}")
        debug_log(f"   Reason: {decision.reason}")
        debug_log(f"   Confidence: {decision.confidence:.2f}")
    else:
        decision = None
    
    # ============================================================
    # STEP 11: EXECUTE BASED ON DECISION (Priority 9)
    # ============================================================
    answer = None
    confidence = decision.confidence if decision else 0.6
    search_used = False
    result_sources = []
    
    try:
        # Route to appropriate handler
        # Each handler calls build_prompt() which handles follow-ups automatically!
        
        if search_mode == "pdf_only" or (decision and decision.primary_source == KnowledgeSource.PDF_RAG):
            add_thought("Searching through uploaded PDF documents", "📚")
            debug_log("📚 Using PDF RAG")
            answer, confidence, result_sources = _handle_pdf_rag(cleaned_query, llm)
            if result_sources:
                add_thought(f"Found {len(result_sources)} relevant PDF sections", "✅")
            
        elif search_mode == "web_only" or (decision and decision.primary_source == KnowledgeSource.WEB_SEARCH):
            add_thought("Initiating web search", "🌐")
            debug_log("🌐 Using web search")
            
            if not allow_web:
                error_response = (
                    "🚫 Web search is disabled.\n\n"
                    "Enable 'Allow Web Search' in settings to use this feature."
                )
                save_message_to_history(query, error_response)
                add_thought("Web search disabled - cannot proceed", "🚫")
                return error_response, 0.0, [], False          
            allowed, firewall_error = WEB_FIREWALL.check_permission()
            if not allowed:
                save_message_to_history(query, firewall_error)
                add_thought("Web search blocked by firewall", "🚫")
                return firewall_error, 0.0, [], False          
            answer, confidence, result_sources, search_used = _handle_web_search(cleaned_query, llm)
        
        elif search_mode == "pdf_web_llm" or (decision and decision.primary_source == KnowledgeSource.HYBRID):
            add_thought("Using hybrid mode (combining multiple sources)", "🔀")
            debug_log("🔀 Using HYBRID mode (PDF + LLM)")
            answer, confidence, result_sources = _handle_hybrid(cleaned_query, llm)
            
            if answer is None:
                debug_log("⚠️ Hybrid mode returned None, falling back to LLM only")
                answer, confidence = _handle_llm_only(cleaned_query, llm, use_gemini=use_gemini_for_query)
                result_sources = []
        
        else:
            # Always try DB before falling back to pure LLM
            add_thought("Checking knowledge base...", "🔍")
            debug_log("🔍 Always-check: trying DB before pure LLM")
            if RAG_ADAPTER:
                answer, confidence, result_sources = _handle_hybrid(cleaned_query, llm)
            if not answer or len(answer.strip()) < 10:
                add_thought("No DB results — using LLM knowledge", "🤖")
                debug_log("🤖 Falling back to pure LLM (DB had nothing)")
                answer, confidence = _handle_llm_only(cleaned_query, llm, use_gemini=use_gemini_for_query)
                result_sources = []
        
        # Check if primary source failed and we have a fallback
        if decision and decision.fallback_source and (not answer or len(answer.strip()) < 10):
            add_thought(f"Primary source insufficient - trying fallback: {decision.fallback_source.value}", "⚠️")
            debug_log(f"⚠️ Primary source failed, trying fallback: {decision.fallback_source.value}")
            
            if decision.fallback_source == KnowledgeSource.LLM_KNOWLEDGE:
                answer, confidence = _handle_llm_only(cleaned_query, llm, use_gemini=use_gemini_for_query)
                if answer:
                    answer = "⚠️ *Using general knowledge (primary source unavailable)*\n\n" + answer
            
            elif decision.fallback_source == KnowledgeSource.WEB_SEARCH and allow_web:
                allowed, _ = WEB_FIREWALL.check_permission()
                if allowed:
                    answer, confidence, result_sources, search_used = _handle_web_search(cleaned_query, llm)
                    if answer:
                        answer = "⚠️ *Searched web as fallback*\n\n" + answer
        
    except Exception as e:
        debug_log(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        answer = None
    
    # ============================================================
    # STEP 12: FINAL FALLBACK (Priority 10)
    # ============================================================
    # Track completion
    if answer and len(answer.strip()) >= 10:
        add_thought("Successfully generated response", "✅")
    if not answer or len(answer.strip()) < 10:
        answer = (
            "I'm having trouble finding a good answer to that. "
            "Could you rephrase your question or provide more context?"
        )
        confidence = 0.3

    # ============================================================
    # HALLUCINATION GUARD — verify and annotate before returning
    # ============================================================
    if answer and len(answer.strip()) >= 10:
        # Only run grounding check when we have actual text to verify against
        # Web results store context in _last_response_source["context"], not in result_sources tuples
        source_texts = []

        # Pull real text content — not URLs
        if _last_response_source.get("context"):
            source_texts.append(_last_response_source["context"])

        for src in result_sources:
            if isinstance(src, str) and not src.startswith("http"):
                source_texts.append(src)
            # Skip bare URLs — they are not text content

        guard_metadata = {
            "searchUsed": search_used,
            "sources": result_sources,
        }

        # Only run guard when we have real source text to check against
        if source_texts:
            answer = apply_hallucination_guard(
                answer,
                source_texts,
                guard_metadata,
                query_source=decision.primary_source.value if decision else "llm"
            )
        # else: no source text available — skip guard, don't strip a valid answer
    
    # ============================================================
    # STEP 13: ADD SOURCE TRANSPARENCY (Priority 11)
    # ============================================================
    if decision and decision.show_user and answer:
        source_note = _get_source_transparency_note(decision)
        if source_note and not answer.startswith("⚠️"):
            answer = source_note + "\n\n" + answer
    
    # ============================================================
    # STEP 14: HUMANIZATION (Priority 12)
    # ============================================================
    add_thought("Enhancing response for natural conversation", "✨")
    answer = humanize_response_sentient(answer, query, chat_history)
    
    # ============================================================
    # STEP 15: SAVE TO DATABASE (Priority 13)
    # ============================================================
    try:
        if not user_manager.current_user:
            debug_log("⚠️ AUTO: No user set! Creating default...")
            user_manager.set_current_user("default_user")
        
        save_success = save_message_to_history(query, answer)
        if save_success:
            debug_log(f"💾 AUTO: Saved successfully")
        else:
            debug_log(f"⚠️ AUTO: Save returned False")
        
        # Reload from chat_history.json for immediate use
        chat_history, _ = load_chat_history()  
        
        # Sync to persistent_data for backward compatibility
        persistent_data["chat_history"] = chat_history
        debug_log(f"💾 Synced: {len(chat_history)} total messages from chat_history.json")
        
    except Exception as e:
        debug_log(f"❌ Save failed: {e}")
        # Emergency fallback
        if 'chat_history' not in persistent_data:
            persistent_data['chat_history'] = []
        persistent_data["chat_history"].append({"user": query, "ai": answer})
    
    # Save to RAG database if available
    if RAG_ADAPTER:
        try:
            RAG_ADAPTER.add_chat_exchange(
                query,
                answer,
                sources=[s[0] if isinstance(s, tuple) else str(s) for s in result_sources[:5]],
                confidence=min(confidence, 0.90)
            )
        except Exception as e:
            debug_log(f"⚠️ Failed to save to RAG: {e}")
    

    # Add personality to response
    # ===== PERSONALITY V2 ENHANCEMENT =====
    try:
        if PERSONALITY_V2_AVAILABLE:
            # STEP 1: Process with personality BEFORE returning
            # Apply personality composition to final answer
            user_id = "default"  # Single-user mode
            answer = process(user_id, cleaned_query, answer)
            
            # STEP 2: Handle compliments (already handled earlier, this is fallback)
            if personality_result.get("is_compliment"):
                return personality_result["compliment_response"], 1.0, [], False
            
            # STEP 3: Build enhanced response
            response_parts = []
            
            # Add memory reference if available
            if personality_result.get("memory_reference"):
                response_parts.append(personality_result["memory_reference"] + " ")
            
            # Add conversation opener if available
            if personality_result.get("opener"):
                response_parts.append(personality_result["opener"])
            
            # Add main response (enhanced with personality)
            #enhanced_answer = enhance_response(answer, cleaned_query)
            response_parts.append(enhanced_answer)
            
            # STEP 4: Add suggestions/questions if available
            if personality_result.get("suggestion"):
                response_parts.append(f"\n\n💡 {personality_result['suggestion']}")
            
            if personality_result.get("curious_question"):
                response_parts.append(f"\n\n{personality_result['curious_question']}")
            
            # STEP 5: Combine all parts
            answer = "".join(response_parts)
            
            # Update mood
            # Mood tracked automatically via emotional valence
            pass
        else:
            debug_log("⚠️ Personality V2 not available, skipping enhancement")
            
    except Exception as e:
        debug_log(f"⚠️ Personality enhancement failed: {e}")
        import traceback
        debug_log(traceback.format_exc())

    add_thought("Response complete and ready", "🎉")
    time.sleep(2.0)
    return answer, confidence, result_sources, search_used

# ============================================================
# Maintain compatibility aliases
# ============================================================
pdf_tutor_enhanced = pdf_tutor_with_explicit_routing
pdf_tutor = pdf_tutor_enhanced


# ============================================================
# PERSONALITY V2 COMMANDS - Add to your main chat loop
# ============================================================

# ============================================================
# Helper Functions
# ============================================================

def _get_source_transparency_note(decision: SourceDecision) -> str:
    """Generate a user-friendly note about which source we used"""
    
    notes = {
        KnowledgeSource.WEB_SEARCH: "🌐 *Searched the web for current information*",
        KnowledgeSource.PDF_RAG: f"📄 *Found in your documents (confidence: {decision.confidence:.0%})*",
        KnowledgeSource.LLM_KNOWLEDGE: "💭 *Using general knowledge* (no specific sources)",
    }
    
    return notes.get(decision.primary_source, "")

# Tracks the last response source so follow-ups can inherit context
_last_response_source = {
    "source": None,       # "web", "pdf", "llm", "hybrid"
    "query": None,        # original query that produced the result
    "context": None,      # the context/results that were used
}

def _handle_web_search(query: str, llm, use_gemini: bool = False) -> Tuple[str, float, list, bool]:
    """Handle web search with caching and proper error handling"""
    try:
        debug_log(f"🌐 Performing web search: '{query}'")
        
        # Reuse cached web context for follow-ups instead of re-searching
        if _last_response_source["source"] == "web" and _last_response_source["context"]:
            debug_log("🔁 Using cached web context for follow-up (no new search)")
            context = _last_response_source["context"]
            results = []  # no new results to display
        else:
            context, results = run_web_search_protected(query, debug=True)
        
        # Check if firewall blocked it
        if isinstance(context, str) and context.startswith("🚫"):
            return context, 0.0, [], False
        
        if not context or not results:
            debug_log("❌ No web search results")
            return None, 0.0, [], False
        
        # Store results in RAG database for future caching
        if RAG_ADAPTER and results:
            try:
                normalized_results = []
                for result in results:
                    if len(result) == 3:
                        normalized_results.append(result)
                    elif len(result) == 2:
                        title, url = result
                        normalized_results.append((title, url, ""))
                
                if normalized_results:
                    added_count = RAG_ADAPTER.add_web_search_results(
                        query,
                        normalized_results,
                        confidence=0.75
                    )
                    debug_log(f"💾 Cached {added_count} search results")
            except Exception as store_error:
                debug_log(f"⚠️ Failed to cache results: {store_error}")
        
        # Build prompt with search results
        prompt = build_prompt(
            persona + "\n\nSynthesize information from web search results to answer accurately.",
            query,
            context,
            use_gemini=use_gemini

        )
        
        llm_out = query_llm_smart(
            prompt,
            max_tokens=500,
            temperature=0.5,
            top_p=0.9,
            stop=["User:", "Question:"],
            use_gemini=use_gemini
        )
        
        answer = process_llm_answer(llm_out, is_code=False)
        
        # Add sources
        """
        sources_text = "\n\n🌐 **Web Sources:**\n"
        for result in results[:5]:
            if len(result) >= 2:
                title, url = result[0], result[1]
                sources_text += f"• [{title}]({url})\n"
        """
        full_response = answer

        display_results = [(title, url) for title, url, *_ in results[:5]]

        # Track that last response came from web search
        _last_response_source["source"] = "web"
        _last_response_source["query"] = query
        _last_response_source["context"] = context

        return full_response, 0.75, display_results, True
        
    except Exception as e:
        debug_log(f"Web search error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, [], False


def _handle_pdf_rag(query: str, llm, use_gemini: bool = False) -> Tuple[str, float, List]:
    """
    Handle query using PDF RAG (Retrieval-Augmented Generation).
    
    REWRITTEN VERSION - Simplified to let build_prompt() handle follow-ups.
    
    This function now focuses solely on:
    1. Retrieving relevant PDF content
    2. Calling build_prompt() which automatically handles:
       - Follow-up detection
       - Conversation context building
       - Combining PDF context with conversation context when needed
    3. Generating and processing the response
    
    Args:
        query: User's question
        llm: Language model instance
    
    Returns:
        Tuple of (answer, confidence, result_sources)
    
    How follow-ups work with PDFs:
        Scenario 1: Follow-up about PDF content
            User: "What does page 5 say about climate change?"
            AI: [quotes PDF content]
            User: "What are the implications of this?"
            → build_prompt() detects follow-up
            → Retrieves previous PDF answer from chat history
            → NEW PDF search for "implications of climate change"
            → Combines both contexts intelligently
        
        Scenario 2: Follow-up about code in PDF
            User: "Show me the algorithm from the paper"
            AI: [provides algorithm/code from PDF]
            User: "Can you explain how it works?"
            → build_prompt() detects this is about previous code
            → Retrieves the code from chat history
            → Also searches PDF for related context
            → LLM can reference both
    """
    
    try:
        debug_log("📚 PDF RAG mode activated")
        
        # ============================================================
        # STEP 1: Check if RAG is available
        # ============================================================
        if not RAG_ADAPTER:
            debug_log("   ⚠️ RAG not available, falling back to LLM only")
            answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
            return answer, confidence, []
        
        # ============================================================
        # STEP 2: Retrieve relevant PDF content
        # ============================================================
        try:
            # Get relevant chunks from PDFs
            pdf_context, retrieval_results = RAG_ADAPTER.retrieve_context(
                query, 
                top_k=5,
                min_score=0.3  # Minimum relevance threshold
            )
            
            if not pdf_context or not retrieval_results:
                debug_log("   ⚠️ No relevant PDF content found")
                
                # Check if any PDFs are loaded
                available_pdfs = get_pdf_list()
                
                if not available_pdfs:
                    # No PDFs loaded at all
                    answer = (
                        "I don't have any PDF documents loaded to search.\n\n"
                        "📤 Upload PDFs using the sidebar to enable document search."
                    )
                    return answer, 0.0, []
                else:
                    # PDFs exist but nothing relevant found
                    debug_log("   ℹ️ PDFs available but no relevant content, using LLM")
                    answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
                    
                    # Add note about no relevant content
                    answer = (
                        "ℹ️ *No highly relevant content found in loaded PDFs. "
                        "Using general knowledge instead.*\n\n" + answer
                    )
                    return answer, confidence, []
            
            debug_log(f"   ✅ Retrieved PDF context: {len(pdf_context)} chars")
            debug_log(f"   📄 Sources: {len(retrieval_results)} chunks")
            
            # Log source diversity
            sources = set(r.chunk.source_name for r in retrieval_results if hasattr(r, 'chunk'))
            if sources:
                debug_log(f"   📚 From {len(sources)} document(s): {', '.join(list(sources)[:3])}")
            
        except Exception as e:
            debug_log(f"   ❌ PDF retrieval failed: {e}")
            import traceback
            debug_log(f"   Stack trace:\n{traceback.format_exc()}")
            
            # Fallback to LLM only
            answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
            return answer, confidence, []
        
        # ============================================================
        # STEP 3: Build prompt with PDF context
        # ============================================================
        # build_prompt() will:
        # 1. Detect if this is a follow-up question
        # 2. If follow-up: load previous conversation
        # 3. Combine PDF context (passed here) with conversation context
        # 4. Format everything appropriately
        
        # Enhanced persona for PDF-based answers
        pdf_persona = persona + (
            "\n\n**INSTRUCTIONS**: "
            "You have relevant content from PDF documents below. "
            "Use this content to answer the question accurately. "
            "Cite specific information from the PDFs when relevant. "
            "If the PDFs don't fully answer the question, combine PDF info with your knowledge."
        )
        
        # build_prompt() handles follow-ups AND combines with PDF context
        prompt = build_prompt(
            pdf_persona,
            query,
            pdf_context,  # PDF context provided here
            use_gemini=use_gemini
        )
        
        debug_log(f"   📨 Built prompt: {len(prompt)} chars")
        
        # ============================================================
        # STEP 4: Generate response
        # ============================================================
        llm_out = query_llm_smart(
            prompt,
            max_tokens=500,
            temperature=0.2,
            top_p=0.8,
            repeat_penalty=1.2,
            stop=["User:", "Question:", "Human:", "Assistant:"],
            use_gemini=use_gemini
        )
        
        if not llm_out:
            debug_log("   ⚠️ LLM returned empty response")
            return "I couldn't generate a response from the PDF content.", 0.3, retrieval_results
        
        # ============================================================
        # STEP 5: Process answer
        # ============================================================
        answer = process_llm_answer(llm_out, is_code=False)
        
        if not answer or len(answer.strip()) < 10:
            debug_log("   ⚠️ Processed answer is too short")
            return "I couldn't generate a proper answer from the PDFs.", 0.3, retrieval_results
        
        debug_log(f"   ✅ Generated answer: {len(answer)} chars")
        
        # ============================================================
        # STEP 6: Add source citations
        # ============================================================
        # Build source list from retrieval results
        source_list = []
        seen_sources = set()
        
        for result in retrieval_results[:5]:  # Top 5 sources
            try:
                if hasattr(result, 'chunk'):
                    source_name = result.chunk.source_name
                    if source_name and source_name not in seen_sources:
                        seen_sources.add(source_name)
                        
                        # Get page number if available
                        page = getattr(result.chunk, 'page_number', None)
                        if page:
                            source_list.append((source_name, f"Page {page}"))
                        else:
                            source_list.append((source_name, ""))
            except AttributeError:
                continue
        
        # Add sources to answer
        if source_list:
            sources_text = "\n\n📄 **Sources:**\n"
            for source_name, page_info in source_list:
                if page_info:
                    sources_text += f"• {source_name} ({page_info})\n"
                else:
                    sources_text += f"• {source_name}\n"
            
            answer = answer + sources_text
            debug_log(f"   📌 Added {len(source_list)} source citations")
        
        # ============================================================
        # STEP 7: Calculate confidence
        # ============================================================
        # Higher confidence for PDF RAG since we have retrieved content
        # Base confidence on retrieval quality
        if retrieval_results:
            avg_score = sum(r.score for r in retrieval_results if hasattr(r, 'score')) / len(retrieval_results)
            confidence = min(0.85, 0.6 + (avg_score * 0.3))  # 0.6 to 0.85 range
        else:
            confidence = 0.7
        
        debug_log(f"   📊 Confidence: {confidence:.2f}")
        
        return answer, confidence, retrieval_results
        
    except Exception as e:
        debug_log(f"❌ Error in _handle_pdf_rag: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        
        # Fallback to LLM only
        debug_log("   🔄 Falling back to LLM only")
        answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
        return answer, confidence, []


def _handle_pdf_rag_with_filters(
    query: str, 
    llm,
    filename_filter: Optional[str] = None,
    min_score: float = 0.3,
    top_k: int = 5
) -> Tuple[str, float, List]:
    """
    Enhanced PDF RAG handler with filtering options.
    
    Allows searching specific files or applying stricter relevance filters.
    Still uses build_prompt() for universal follow-up handling.
    
    Args:
        query: User's question
        llm: Language model instance
        filename_filter: Optional filename to restrict search to
        min_score: Minimum relevance score (0.0 to 1.0)
        top_k: Number of chunks to retrieve
    
    Returns:
        Tuple of (answer, confidence, result_sources)
    
    Examples:
        # Search specific file
        answer, conf, sources = _handle_pdf_rag_with_filters(
            "What is the methodology?",
            llm,
            filename_filter="research_paper.pdf"
        )
        
        # Stricter relevance
        answer, conf, sources = _handle_pdf_rag_with_filters(
            "Define quantum entanglement",
            llm,
            min_score=0.5  # Only highly relevant chunks
        )
    """
    
    try:
        debug_log(f"📚 PDF RAG (filtered) mode activated")
        if filename_filter:
            debug_log(f"   🎯 Filtering by file: {filename_filter}")
        debug_log(f"   📊 Min score: {min_score}, Top K: {top_k}")
        
        if not RAG_ADAPTER:
            debug_log("   ⚠️ RAG not available")
            answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
            return answer, confidence, []
        
        # Retrieve with filters
        try:
            if filename_filter:
                # Search specific file
                pdf_context, retrieval_results = RAG_ADAPTER.retrieve_from_file(
                    query,
                    filename_filter,
                    top_k=top_k,
                    min_score=min_score
                )
            else:
                # Search all files with score filter
                pdf_context, retrieval_results = RAG_ADAPTER.retrieve_context(
                    query,
                    top_k=top_k,
                    min_score=min_score
                )
            
            if not pdf_context:
                debug_log("   ⚠️ No content passed filter")
                
                # Provide helpful feedback
                if filename_filter:
                    answer = f"I couldn't find relevant content in '{filename_filter}' for your question."
                else:
                    answer = "I couldn't find content that meets the relevance threshold."
                
                # Fallback to LLM
                llm_answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
                answer = answer + "\n\nUsing general knowledge instead:\n\n" + llm_answer
                
                return answer, confidence, []
            
            debug_log(f"   ✅ Retrieved {len(pdf_context)} chars from {len(retrieval_results)} chunks")
            
        except Exception as e:
            debug_log(f"   ❌ Filtered retrieval failed: {e}")
            answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
            return answer, confidence, []
        
        # Build prompt (handles follow-ups automatically)
        pdf_persona = persona + (
            "\n\n**INSTRUCTIONS**: "
            f"You have highly relevant content (min score: {min_score}) from PDF documents. "
            "Use this content to answer accurately. "
            "The content has been filtered for relevance, so trust it."
        )
        
        prompt = build_prompt(pdf_persona, query, pdf_context, use_gemini=use_gemini)
        
        # Generate response
        llm_out = llm(
            prompt,
            max_tokens=500,
            temperature=0.4,
            top_p=0.9,
            stop=["User:", "Question:"]
        )
        
        if not llm_out:
            return "I couldn't generate a response.", 0.3, retrieval_results
        
        answer = process_llm_answer(llm_out, is_code=False)
        
        if not answer or len(answer.strip()) < 10:
            return "I couldn't generate a proper answer.", 0.3, retrieval_results
        
        # Add source citations
        source_list = []
        seen_sources = set()
        
        for result in retrieval_results[:5]:
            try:
                if hasattr(result, 'chunk'):
                    source_name = result.chunk.source_name
                    score = getattr(result, 'score', 0.0)
                    
                    if source_name and source_name not in seen_sources:
                        seen_sources.add(source_name)
                        page = getattr(result.chunk, 'page_number', None)
                        
                        if page:
                            source_list.append((source_name, f"Page {page}", score))
                        else:
                            source_list.append((source_name, "", score))
            except AttributeError:
                continue
        
        if source_list:
            sources_text = "\n\n📄 **Sources (Relevance Scores):**\n"
            for source_name, page_info, score in source_list:
                if page_info:
                    sources_text += f"• {source_name} ({page_info}) - {score:.2f}\n"
                else:
                    sources_text += f"• {source_name} - {score:.2f}\n"
            
            answer = answer + sources_text
        
        # Higher confidence for filtered results
        if retrieval_results:
            avg_score = sum(r.score for r in retrieval_results if hasattr(r, 'score')) / len(retrieval_results)
            confidence = min(0.90, 0.65 + (avg_score * 0.3))
        else:
            confidence = 0.75
        
        return answer, confidence, retrieval_results
        
    except Exception as e:
        debug_log(f"❌ Error in _handle_pdf_rag_with_filters: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        
        answer, confidence = _handle_llm_only(query, llm, use_gemini=use_gemini_for_query)
        return answer, confidence, []

def _handle_llm_only(query: str, llm, use_gemini: bool = False) -> Tuple[str, float]:
    """
    Handle query using pure LLM knowledge.
    
    REWRITTEN VERSION - Simplified to let build_prompt() handle all context.
    
    This function now focuses solely on:
    1. Building memory/preference context
    2. Calling build_prompt() which automatically handles:
       - Follow-up detection
       - Conversation context building
       - Pronoun resolution
       - Code awareness
    3. Generating and processing the response
    
    Args:
        query: User's question
        llm: Language model instance
    
    Returns:
        Tuple of (answer, confidence)
    
    How follow-ups work:
        - User: "Write a Python function to sort a list"
        - AI: [provides code]
        - User: "Can you explain it?"
        - build_prompt() automatically:
            * Detects this is a follow-up
            * Retrieves previous code from chat history
            * Builds context showing the code
            * Enhances system message for code awareness
        - Result: LLM can reference the code directly
    """
    
    try:
        debug_log("🤖 LLM ONLY mode activated")
        
        # ============================================================
        # STEP 1: Build memory/preference context
        # ============================================================
        # This adds user preferences, name, etc.
        # Note: We don't manually check for follow-ups here!
        # build_prompt() will detect and handle them automatically
        
        memory_context = build_memory_context_with_name(query)
        
        if memory_context:
            debug_log(f"   📝 Added memory context ({len(memory_context)} chars)")
        
        # ============================================================
        # STEP 2: Build prompt with universal follow-up handling
        # ============================================================
        # build_prompt() does ALL the magic:
        # - Loads chat history from database
        # - Detects if this is a follow-up question
        # - If follow-up: builds conversation context with previous exchanges
        # - If follow-up about code: preserves code blocks and enhances system msg
        # - If new question: just adds memory context
        # - Formats everything for the current model type
        
        prompt = build_prompt(
            persona,  # System message/persona
            query,    # User's current question
            memory_context,  # User preferences (optional, can be empty)
            use_gemini=use_gemini
        )
        
        debug_log(f"   📨 Built prompt: {len(prompt)} chars")
        
        # ============================================================
        # STEP 3: Generate response
        # ============================================================
        llm_out = query_llm_smart(
            prompt,
            max_tokens=8000,
            temperature=0.5,
            top_p=0.9,
            stop=["User:", "Question:", "Human:", "Assistant:"],
            use_gemini=use_gemini
        )
        
        if not llm_out:
            debug_log("   ⚠️ LLM returned empty response")
            return "I apologize, but I couldn't generate a response. Please try again.", 0.3
        
        # ============================================================
        # STEP 4: Process and clean the answer
        # ============================================================
        answer = process_llm_answer(llm_out, is_code=False)
        
        if not answer or len(answer.strip()) < 5:
            debug_log("   ⚠️ Processed answer is too short")
            return "I couldn't generate a proper answer. Could you rephrase your question?", 0.3
        
        debug_log(f"   ✅ Generated answer: {len(answer)} chars")
        
        # Confidence for LLM-only is moderate since we have no external validation
        confidence = 0.6
        
        return answer, confidence
        
    except Exception as e:
        debug_log(f"❌ Error in _handle_llm_only: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        
        # Return a safe fallback
        return (
            "I encountered an error while processing your question. "
            "Please try rephrasing it or try again.",
            0.3
        )


def _handle_llm_with_code(query: str, llm, max_tokens: int = 800, use_gemini: bool = False) -> Tuple[str, float]:
    """
    Handle code generation queries using LLM.
    
    This is a specialized version of _handle_llm_only for code generation.
    Still uses build_prompt() for universal follow-up handling.
    
    Args:
        query: User's code-related question
        llm: Language model instance
        max_tokens: Maximum tokens for code generation (default 800)
    
    Returns:
        Tuple of (answer, confidence)
    
    Examples of follow-up handling:
        User: "Write a binary search function"
        AI: [provides code]
        User: "Add error handling to it"
        → build_prompt() retrieves previous code and builds context
        User: "What's the time complexity?"
        → build_prompt() retrieves code and adds it to context
    """
    
    try:
        debug_log("💻 LLM CODE mode activated")
        
        # Build memory context
        memory_context = build_memory_context_with_name(query)
        
        # Enhanced persona for code generation
        code_persona = persona + (
            "\n\nYou are an expert programmer. "
            "Provide clean, well-commented code with explanations. "
            "Include docstrings and handle edge cases."
        )
        
        # build_prompt() handles follow-ups automatically
        # If user asks "add error handling to it", it will retrieve previous code
        prompt = build_prompt(
            code_persona,
            query,
            memory_context,
            use_gemini=use_gemini
        )
        
        debug_log(f"   📨 Built code prompt: {len(prompt)} chars")
        
        # Generate with more tokens for code
        llm_out = query_llm_smart(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temp for code (more deterministic)
            top_p=0.9,
            stop=["User:", "Question:", "Human:", "```\n\nUser:"],
            use_gemini=use_gemini
        )
        
        if not llm_out:
            return "I couldn't generate code. Please try again.", 0.3
        
        # Process answer (preserves code blocks)
        answer = process_llm_answer(llm_out, is_code=True)
        
        if not answer or len(answer.strip()) < 10:
            return "I couldn't generate a proper code response. Please rephrase.", 0.3
        
        debug_log(f"   ✅ Generated code response: {len(answer)} chars")
        
        # Higher confidence for code since we use specialized prompt
        confidence = 0.7
        
        return answer, confidence
        
    except Exception as e:
        debug_log(f"❌ Error in _handle_llm_with_code: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        
        return (
            "I encountered an error while generating code. "
            "Please try again or rephrase your request.",
            0.3
        )
       
def _handle_hybrid(query: str, llm, use_gemini: bool = False) -> Tuple[str, float, list]:
    """
    ✅ FIXED: Hybrid (PDF + LLM) with proper error handling and response flow
    """
    if not RAG_ADAPTER:
        debug_log("🛡️ SAFETY: RAG_ADAPTER not available for hybrid mode")
        return None, 0.0, []

    try:
        debug_log("🔀 Executing HYBRID mode (PDF + LLM)")

        # -------------------------------------------------
        # Step 1: PDF retrieval
        # -------------------------------------------------
        search_query = query
        pdf_context = ""
        rag_sources = []
        pdf_confidence = 0.0

        try:
            all_results = []

            # Try multiple retrieval methods
            hybrid_results = RAG_ADAPTER.rag_db.retrieve(
                search_query, top_k=15, method="hybrid"
            )
            if hybrid_results:
                all_results.extend(hybrid_results)

            vector_results = RAG_ADAPTER.rag_db.retrieve(
                search_query, top_k=10, method="vector"
            )
            if vector_results:
                existing_ids = {r.chunk.chunk_id for r in all_results}
                for r in vector_results:
                    if r.chunk.chunk_id not in existing_ids:
                        all_results.append(r)

            # Filter and sort with LOWER threshold for better recall
            min_score = 0.08  # ✅ LOWERED from 0.15 for better semantic matching
            filtered = [r for r in all_results if r.score >= min_score]

            # ✅ FALLBACK: If still no results, use top results regardless of score
            if not filtered and all_results:
                debug_log(f"⚠️ No results above {min_score}, using top {min(5, len(all_results))} by score")
                filtered = sorted(all_results, key=lambda x: x.score, reverse=True)[:5]

            if not filtered and all_results:
                filtered = sorted(all_results, key=lambda x: x.score, reverse=True)[:3]

            if filtered:
                filtered = sorted(filtered, key=lambda x: x.score, reverse=True)[:3]

                # Build context
                parts = []
                for i, r in enumerate(filtered, 1):
                    text = r.chunk.content
                    if len(text) > 500:
                        text = text[:500] + "..."
                    parts.append(
                        f"[Source {i}: {r.chunk.source_name}]\n{text}"
                    )

                pdf_context = "\n\n---\n\n".join(parts)

                if len(pdf_context) > 2000:
                    pdf_context = pdf_context[:2000] + "\n\n[Content truncated...]"

                pdf_confidence = sum(r.score for r in filtered) / len(filtered)
                rag_sources = [(r.chunk.source_name, r.chunk.chunk_id) for r in filtered]

                debug_log(
                    f"   ✅ PDF context: {len(filtered)} chunks "
                    f"(confidence {pdf_confidence:.2f})"
                )

        except Exception as e:
            debug_log(f"   ⚠️ PDF retrieval failed: {e}")
            pdf_context = ""
            pdf_confidence = 0.0

        # -------------------------------------------------
        # Step 2: Build context
        # -------------------------------------------------
        memory_context = build_memory_context_with_name(query)
        if memory_context and len(memory_context) > 500:
            memory_context = memory_context[:500] + "..."

        context_parts = []

        if memory_context:
            context_parts.append(memory_context)

        if pdf_context:
            context_parts.append(
                "**Reference Material:**\n"
                "Use when relevant, supplement with general knowledge.\n\n"
                + pdf_context
            )

        full_context = "\n\n".join(context_parts)

        MAX_CONTEXT = 3000
        if len(full_context) > MAX_CONTEXT:
            full_context = full_context[:MAX_CONTEXT] + "\n\n[Context truncated]"

        # -------------------------------------------------
        # Step 3: Generate response
        # -------------------------------------------------
        enhanced_persona = (
            persona
            + "\n\nYou have access to PDF documents. "
            "Use them to provide accurate answers. "
            "If the PDFs contain relevant information, cite them."
        )

        prompt = build_prompt(enhanced_persona, query, full_context, use_gemini=use_gemini)

        # 🔥 CRITICAL: Add debug logging before LLM call
        debug_log(f"📝 Calling LLM with prompt length: {len(prompt)} chars")
        
        llm_out = query_llm_smart(
            prompt,
            max_tokens=8000,
            temperature=0.5,
            top_p=0.9,
            stop=["User:", "Question:"],
            use_gemini=use_gemini
        )

        # 🔥 CRITICAL: Debug the LLM output structure
        debug_log(f"📤 LLM output type: {type(llm_out)}")
        if isinstance(llm_out, dict):
            debug_log(f"📤 LLM output keys: {llm_out.keys()}")

        # Process answer
        answer = process_llm_answer(llm_out, is_code=False)
        
        # 🔥 CRITICAL: Check if answer is empty
        if not answer or len(answer.strip()) < 5:
            debug_log(f"❌ Empty or invalid answer generated: '{answer}'")
            return None, 0.0, []
        
        debug_log(f"✅ Generated answer length: {len(answer)} chars")


        # -------------------------------------------------
        # Step 4: Add source attribution
        # -------------------------------------------------
        if rag_sources:
            
            sources_used = {src for src, _ in rag_sources}
           
            
            
            
            final_answer = answer
        else:
            final_answer = answer

        # -------------------------------------------------
        # Step 5: Calculate confidence
        # -------------------------------------------------
        hybrid_confidence = (
            (pdf_confidence * 0.6) + (0.65 * 0.4)
            if pdf_confidence > 0 else 0.65
        )

        debug_log(f"📊 Final confidence: {hybrid_confidence:.2f}")
        debug_log(f"✅ Returning: answer ({len(final_answer)} chars), confidence ({hybrid_confidence:.2f}), sources ({len(rag_sources)})")

        return final_answer, hybrid_confidence, rag_sources

    except Exception as e:
        debug_log(f"❌ Hybrid mode error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, []

# ----------------------------
# ENHANCED PDF MANAGEMENT
# ----------------------------
def process_pdfs(uploaded_files):
    """
    Process PDFs using new RAG system with proper error handling
    """
    if not uploaded_files:
        return {"error": "No files uploaded — drop some PDFs in and I'll get reading! 📖"}
    
    if RAG_ADAPTER is None:
        return {"error": "RAG system not available. PDF search requires the embedding model."}
    
    try:
        debug_log(f"📄 Processing {len(uploaded_files)} PDF(s)...")
        
        # ✅ FIX 1: Extract text from PDFs with error handling
        pdf_texts = {}
        failed_files = []
        
        for filepath in uploaded_files:
            filename = os.path.basename(filepath)
            try:
                text = extract_pdf_text(filepath)
                if text and len(text.strip()) > 50:  # Minimum viable text
                    pdf_texts[filename] = text
                    debug_log(f"  ✅ {filename}: {len(text)} chars")
                else:
                    failed_files.append(filename)
                    debug_log(f"  ⚠️ {filename}: No text extracted")
            except Exception as e:
                failed_files.append(filename)
                debug_log(f"  ❌ {filename}: {e}")
        
        if not pdf_texts:
            return {
                "error": f"Could not extract text from any PDFs. Failed: {', '.join(failed_files)}"
            }
        
        # ✅ FIX 2: Process using RAG system
        debug_log("🔄 Processing with RAG system...")
        
        chunks, sources, embeddings, topics = RAG_ADAPTER.process_pdfs_legacy(
            pdf_texts,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if not chunks or not sources:
            return {
                "error": "PDF processing failed - no chunks created"
            }
        
        debug_log(f"✅ Created {len(chunks)} chunks from {len(set(sources))} PDFs")
        
        # ✅ FIX 3: Update persistent data (for backward compatibility)
        persistent_data.update({
            "chunks": chunks,
            "sources": sources,
            "chunk_emb": embeddings,
            "topics": topics
        })
        
        # ✅ FIX 4: Save to disk (legacy format)
        try:
            save_pdf_index(chunks, sources, embeddings, topics)
            debug_log("💾 Saved to legacy storage")
        except Exception as e:
            debug_log(f"⚠️ Legacy save failed: {e}")
        
        # ✅ FIX 5: Get updated stats
        stats = {}
        total_chunks = len(chunks)
        total_pdfs = len(set(sources))
        
        if RAG_ADAPTER:
            try:
                stats = RAG_ADAPTER.get_stats()
                total_chunks = stats.get("total_chunks", len(chunks))
                total_pdfs = stats.get("total_sources", len(set(sources)))
            except Exception as e:
                debug_log(f"⚠️ Stats retrieval failed: {e}")
        
        # ✅ FIX 6: Build success message
        msg = f"✅ Processed {len(pdf_texts)} PDF(s)! "
        msg += f"Total: {total_pdfs} PDFs with {total_chunks} chunks."
        
        if failed_files:
            msg += f"\n⚠️ Failed to process: {', '.join(failed_files)}"
        
        msg += f"\n💾 Stored in RAG database — ready for intelligent search! 🎉"
        
        # ✅ FIX 7: Return structured response
        return {
            "message": msg,
            "total_chunks": total_chunks,
            "total_documents": total_pdfs,
            "processed_files": list(pdf_texts.keys()),
            "failed_files": failed_files,
            "stats": stats
        }
        
    except Exception as e:
        debug_log(f"❌ PDF processing error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error processing PDFs: {str(e)}"
        }
def clear_pdfs():
    """Clear all PDFs from hybrid system"""
    if HYBRID_RAG is None:
        return "⚠️ Hybrid RAG system not initialized."

    try:
        # Get list of all PDFs
        pdf_list = HYBRID_RAG.get_pdf_list()

        if not pdf_list:
            return "📭 No PDFs to clear."

        # Delete each PDF
        deleted_count = 0
        for pdf in pdf_list:
            success, msg = HYBRID_RAG.delete_pdf(pdf["pdf_id"])
            if success:
                deleted_count += 1

        # Also clear legacy persistent data
        persistent_data.update(
            {"chunks": None, "sources": None, "chunk_emb": None, "topics": None}
        )

        debug_log(f"🗑️ Cleared {deleted_count} PDFs from hybrid system")

        return (
            f"🗑️ Cleared {deleted_count} PDF(s)!\n\n💡 Upload new ones to start fresh."
        )

    except Exception as e:
        logging.error(f"Error clearing PDFs: {e}")
        return f"❌ Error clearing PDFs: {str(e)}"



def get_pdf_status():
    """Get detailed status of currently loaded PDFs with proper formatting"""
    try:
        if RAG_ADAPTER is None:
            return "RAG system not initialized. PDF search is unavailable."
        
        # Get fresh stats from database
        stats = RAG_ADAPTER.get_stats()
        pdf_list = get_pdf_list()
        
        # Build formatted status message
        status_msg = "📊 **PDF Status Report**\n"
        status_msg += "=" * 50 + "\n\n"
        
        if not pdf_list or len(pdf_list) == 0:
            status_msg += "📭 **No PDFs loaded**\n\n"
            status_msg += "Upload PDFs using the sidebar to enable document search.\n"
            return status_msg
        
        # Summary stats
        total_chunks = stats.get("total_chunks", 0)
        total_docs = len(pdf_list)
        
        status_msg += f"✅ **Loaded PDFs:** {total_docs}\n"
        status_msg += f"📦 **Total Chunks:** {total_chunks}\n"
        status_msg += f"🔍 **Search Status:** Active\n\n"
        
        # List each PDF with details
        status_msg += "📄 **Document List:**\n\n"
        
        for i, pdf in enumerate(pdf_list, 1):
            filename = pdf.get("filename", "Unknown")
            chunk_count = pdf.get("chunk_count", 0)
            last_updated = pdf.get("last_updated", "Unknown")
            
            status_msg += f"{i}. **{filename}**\n"
            status_msg += f"   • Chunks: {chunk_count}\n"
            status_msg += f"   • Updated: {last_updated}\n\n"
        
        status_msg += "=" * 50 + "\n"
        status_msg += "💡 Use 'use pdf: [question]' to search these documents\n"
        
        return status_msg
        
    except Exception as e:
        debug_log(f"Error getting PDF status: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ Error retrieving PDF status: {str(e)}"


def get_pdf_list_for_deletion():
    """Get list of currently loaded PDFs for the dropdown selector."""
    sources = persistent_data.get("sources")
    if not sources:
        return []
    return sorted(list(set(sources)))



def load_previous_chat():
    """Load previous chat history into the UI."""
    if not persistent_data["chat_history"]:
        return None, "No previous chat history found."
    chat_display = [
        (entry["user"], entry["ai"]) for entry in persistent_data["chat_history"]
    ]
    return chat_display, f"Loaded {len(chat_display)} previous messages!"


def clear_history():
    """Clear chat history from memory and disk."""
    persistent_data["chat_history"] = []
    persistent_data["interaction_count"] = 0
    save_chat_history([], 0)
    return "Chat history cleared! Starting fresh.", None


def view_cache_stats():
    """View cache statistics."""
    chat_size = len(persistent_data["chat_history"])
    search_size = len(search_cache)

    stats = f"""📊 **Cache Statistics:**
    
💬 **Chat History:** {chat_size} messages
🔍 **Search Cache:** {search_size} queries cached
📁 **Storage Location:** `{CACHE_DIR.absolute()}`
"""
    recent_searches = sorted(
        [(k, v) for k, v in search_cache.items()],
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True,
    )[:5]
    stats += "\n**Recent Searches:**\n"
    for _, data in recent_searches:
        query = data.get("query", "Unknown")
        timestamp = data.get("timestamp", "Unknown")
        stats += f"\n• {query} (cached at {timestamp[:19]})"
    if not recent_searches:
        stats += "\n(No searches cached yet)"
    return stats


def export_chat_pdf():
    if not persistent_data["chat_history"]:
        return "Nothing to export yet — start chatting first!"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Chat History with Project Elixer (AI Assistant)", ln=True, align="C")
    pdf.ln(5)
    for entry in persistent_data["chat_history"]:
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, f"You: {entry['user']}")
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, f"Project Elixer: {entry['ai']}")
        pdf.ln(3)
    filename = f"chat_with_alex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return f"Saved! Your chat is in '{filename}' 📄"


# ============================================================
# AUTO-DETECT MODEL PROMPT FORMAT (MUST BE HERE, BEFORE build_prompt)
# ============================================================
def detect_model_format(model_path):
    """Detect which prompt format to use based on model name"""
    model_name = model_path.lower()

    # CodeLlama uses Llama 2 format
    if "codellama" in model_name or "code-llama" in model_name:
        return "llama2"
    elif any(x in model_name for x in ["mistral", "mixtral"]):
        return "mistral"
    elif any(x in model_name for x in ["llama-3", "llama3"]):
        return "llama3"
    elif any(x in model_name for x in ["llama-2", "llama2", "llama"]):
        return "llama2"
    elif any(x in model_name for x in ["phi", "phi-3"]):
        return "phi"
    elif any(x in model_name for x in ["alpaca", "vicuna"]):
        return "alpaca"
    elif any(x in model_name for x in ["chatgpt", "gpt"]):
        return "chatml"
    else:
        return "universal"


# ----------------------------
# Model Initialization
# ----------------------------
from config import MODEL_PATH as model_path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    emb_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
    print("✅ Embedding model loaded from local cache")
except Exception as e:
    print(f"⚠️ Could not load embedding model offline: {e}")
    print("📥 Attempting to download embedding model (this may require internet)...")
    try:
        emb_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
        print("✅ Embedding model downloaded and cached")
    except Exception as e2:
        print(f"❌ Failed to download embedding model: {e2}")
        print("⚠️ PDF search will not work without embedding model")
        emb_model = None


# Load the LLM model
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=32768,
        n_threads=os.cpu_count(),
        verbose=False,
        n_gpu_layers=0,
    )
    print("✅ Llama model loaded successfully")
except Exception as e:
    debug_log(f"Failed to initialize Llama model: {e}")
    print(f"❌ Failed to load Llama model: {e}")
    llm = None

# ➤ FIX: detect model format so build_prompt() works
MODEL_FORMAT = detect_model_format(model_path)
print(f"📌 Detected model format: {MODEL_FORMAT}")

def initialize_gemini(api_key: str = None):
    """
    Initialize Gemini LLM
    
    Args:
        api_key: Google API key (optional if GEMINI_API_KEY env var is set)
    
    Returns:
        bool: True if successful
    """
    global gemini_llm
    
    if not GEMINI_AVAILABLE:
        debug_log("❌ Gemini integration not available")
        return False
    
    try:
        gemini_llm = create_gemini_llm(
            api_key=api_key, 
            model="gemini-2.5-flash",
            enable_web_search=True
        )
        debug_log("✅ Gemini LLM initialized successfully")
        return True
    except Exception as e:
        debug_log(f"❌ Failed to initialize Gemini: {e}")
        return False
    
# ============================================================
# MODEL MANAGER - Dynamic Model Switching
# ============================================================

import os
from pathlib import Path
from typing import Dict, Optional, List
import threading

class ModelManager:
    """
    Manages multiple LLM models with hot-swapping capability
    """
    
    def __init__(self, models_dir: str = r"C:\Users\salam\Downloads"):
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.current_model_path = None
        self.current_format = None
        self.available_models = {}
        self._lock = threading.RLock()
        
        # Scan for available models
        self._scan_models()
    
    def _scan_models(self):
        """Scan directory for available .gguf models"""
        try:
            if not self.models_dir.exists():
                debug_log(f"⚠️ Models directory not found: {self.models_dir}")
                return
            
            debug_log(f"🔍 Scanning for models in: {self.models_dir}")
            
            for file in self.models_dir.glob("*.gguf"):
                model_info = self._parse_model_name(file.name)
                model_info['path'] = str(file)
                model_info['size_mb'] = file.stat().st_size / (1024 * 1024)
                
                self.available_models[file.stem] = model_info
                debug_log(f"  ✅ Found: {file.name} ({model_info['size_mb']:.1f} MB)")
            
            debug_log(f"📊 Total models found: {len(self.available_models)}")
            
        except Exception as e:
            debug_log(f"❌ Error scanning models: {e}")
    
    def _parse_model_name(self, filename: str) -> Dict:
        """
        Parse model metadata from filename
        Examples:
          - Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
          - mistral-7b-instruct-v0.2.Q4_K_M.gguf
        """
        name_lower = filename.lower()
        
        info = {
            'filename': filename,
            'display_name': filename.replace('.gguf', ''),
            'family': 'unknown',
            'size': 'unknown',
            'quantization': 'unknown',
            'format': 'unknown'
        }
        
        # Detect model family
        if 'llama-3' in name_lower or 'llama3' in name_lower:
            info['family'] = 'Llama 3'
            info['format'] = 'llama3'
        elif 'llama-2' in name_lower or 'llama2' in name_lower:
            info['family'] = 'Llama 2'
            info['format'] = 'llama2'
        elif 'codellama' in name_lower:
            info['family'] = 'CodeLlama'
            info['format'] = 'llama2'
        elif 'mistral' in name_lower:
            info['family'] = 'Mistral'
            info['format'] = 'mistral-instruct' if 'instruct' in name_lower else 'mistral-base'
        elif 'phi' in name_lower:
            info['family'] = 'Phi'
            info['format'] = 'phi'
        
        # Detect size
        import re
        size_match = re.search(r'(\d+)B', filename, re.IGNORECASE)
        if size_match:
            info['size'] = f"{size_match.group(1)}B"
        
        # Detect quantization
        quant_match = re.search(r'Q\d+_[KM_]+', filename, re.IGNORECASE)
        if quant_match:
            info['quantization'] = quant_match.group(0)
        
        return info
    
    def load_model(self, model_key: str, n_ctx: int = 8000, n_gpu_layers: int = 0) -> bool:
        """
        Load a specific model by key
        
        Args:
            model_key: Model identifier (filename stem)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
        
        Returns:
            Success status
        """
        with self._lock:
            if model_key not in self.available_models:
                debug_log(f"❌ Model not found: {model_key}")
                return False
            
            model_info = self.available_models[model_key]
            model_path = model_info['path']
            
            try:
                debug_log(f"🔄 Loading model: {model_info['display_name']}")
                debug_log(f"   Path: {model_path}")
                debug_log(f"   Format: {model_info['format']}")
                
                # Unload current model if exists
                if self.current_model is not None:
                    debug_log("   Unloading previous model...")
                    del self.current_model
                    self.current_model = None
                
                # Load new model
                from llama_cpp import Llama
                
                self.current_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=os.cpu_count(),
                    verbose=False,
                    n_gpu_layers=n_gpu_layers
                )
                
                self.current_model_path = model_path
                self.current_format = model_info['format']
                
                # Update global MODEL_FORMAT
                global MODEL_FORMAT, llm
                MODEL_FORMAT = self.current_format
                llm = self.current_model
                
                debug_log(f"✅ Model loaded successfully!")
                debug_log(f"   Format set to: {MODEL_FORMAT}")
                
                return True
                
            except Exception as e:
                debug_log(f"❌ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models with metadata"""
        return [
            {
                'key': key,
                'display_name': info['display_name'],
                'family': info['family'],
                'size': info['size'],
                'quantization': info['quantization'],
                'size_mb': info['size_mb'],
                'is_current': self.current_model_path == info['path']
            }
            for key, info in self.available_models.items()
        ]
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get info about currently loaded model"""
        if not self.current_model_path:
            return None
        
        for key, info in self.available_models.items():
            if info['path'] == self.current_model_path:
                return {
                    'key': key,
                    'display_name': info['display_name'],
                    'family': info['family'],
                    'size': info['size'],
                    'quantization': info['quantization'],
                    'format': self.current_format
                }
        
        return None


# ============================================================
# INITIALIZE MODEL MANAGER
# ============================================================

MODEL_MANAGER = ModelManager()

# Load default model (current model_path)
if llm and model_path:
    # Set current model in manager
    default_key = Path(model_path).stem
    if default_key in MODEL_MANAGER.available_models:
        MODEL_MANAGER.current_model = llm
        MODEL_MANAGER.current_model_path = model_path
        MODEL_MANAGER.current_format = MODEL_FORMAT
        debug_log(f"✅ Current model registered: {default_key}")






# 1. Initialize RAG_ENGINE with embedding model AND the cache directory
try:
    # We use CACHE_DIR / "rag_index" to give the RAG engine its own subfolder
    rag_cache_path = CACHE_DIR / "rag_index"
    rag_cache_path.mkdir(exist_ok=True)
    
    # Pass both required arguments
    RAG_ENGINE = create_rag_engine(
        embedding_model=emb_model, 
        cache_dir=str(rag_cache_path)  # Pass as string if the function expects a path string
    )
    print(f"✅ RAG_ENGINE initialized successfully at {rag_cache_path}")
    
except Exception as e:
    print(f"⚠️ Failed to initialize RAG_ENGINE: {e}")
    RAG_ENGINE = None

# 2. Proceed with Hybrid RAG Manager
if RAG_ENGINE and emb_model:
    HYBRID_RAG = HybridRAGManager(RAG_ENGINE, emb_model)
    logging.info("✅ Hybrid RAG Manager initialized")
# ============================================================
# INITIALIZE ALL GLOBAL INSTANCES (CORRECTED)
# Place after model loading (after llm and emb_model are initialized)
# ============================================================

# Initialize AFTER checking if models loaded successfully
QUERY_DECOMPOSER = None
CONFIDENCE_CALIBRATOR = None
ANSWER_VERIFIER = None
CONTEXT_COMPRESSOR = None
FEEDBACK_LEARNER = None

"""
# Initialize FeedbackLearner (doesn't need models)
try:
    FEEDBACK_LEARNER = FeedbackLearner(CACHE_DIR)
    debug_log("✅ Feedback learner initialized")
except Exception as e:
    debug_log(f"⚠️ Failed to initialize feedback learner: {e}")
"""

# Initialize LLM-dependent components
if llm is not None:
    try:
        QUERY_DECOMPOSER = QueryDecomposer(llm)
        CONTEXT_COMPRESSOR = ContextCompressor(llm)
        debug_log("✅ Query decomposer and context compressor initialized")
    except Exception as e:
        debug_log(f"⚠️ Failed to initialize query components: {e}")
else:
    debug_log("⚠️ LLM is None - query decomposer disabled")

# Initialize embedding-dependent components
if emb_model is not None:
    try:
        ANSWER_VERIFIER = AnswerVerifier(emb_model)
        CONFIDENCE_CALIBRATOR = ConfidenceCalibrator()
        debug_log("✅ Answer verifier and confidence calibrator initialized")
    except Exception as e:
        debug_log(f"⚠️ Failed to initialize verification components: {e}")
else:
    debug_log("⚠️ Embedding model is None - answer verifier disabled")


# ================= RAG API FUNCTIONS (FIXES) =================

def clear_rag_index():
    """
    Clears the entire RAG index (vector DB), deletes all documents,
    and resets the persistent data file for RAG.

    This is necessary before processing new PDFs to ensure a clean state.
    """
    global RAG_ENGINE, persistent_data

    try:
        if RAG_ENGINE:
            RAG_ENGINE.clear_index()  # Assumes this method exists in AdvancedRAGEngine

        # Clear persistent data related to RAG (chunks, sources, embeddings)
        persistent_data["chunks"] = []
        persistent_data["sources"] = []
        persistent_data["embeddings"] = None

        # Save the cleared persistent data
        save_persistent_data()  # Assumes this function is defined elsewhere in yan.py

        # Clean up the PDFs directory
        PDF_DIR = PDF_STORAGE_DIR  # Assumes this global constant is defined in yan.py
        shutil.rmtree(PDF_DIR, ignore_errors=True)  # Assumes shutil is imported
        Path(PDF_DIR).mkdir(exist_ok=True)  # Assumes Path is imported

        return True

    except Exception as e:
        # debug_log(f"Error during RAG index clear: {e}")
        return False


# ============================================================
# INITIALIZE SEARCH ENGINE (CRITICAL)
# ============================================================
def simple_web_search(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Simple fallback web search using DuckDuckGo lite
    Returns (context_string, results_list)
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        print(f"🔍 Simple search for: {query}")

        # Use DuckDuckGo lite (simple, no JS required)
        response = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # Find result links
        for link in soup.find_all("a", class_="result-link"):
            title = link.get_text(strip=True)
            url = link.get("href")

            if url and url.startswith("http"):
                results.append((title, url))

            if len(results) >= 5:
                break

        if not results:
            print("⚠️ No results found")
            return "", []

        # Format context
        context = f"Web search results for '{query}':\n\n"
        for i, (title, url) in enumerate(results, 1):
            context += f"[{i}] {title}\n"
            context += f"URL: {url}\n\n"

        print(f"✅ Found {len(results)} results")
        return context, results

    except Exception as e:
        print(f"❌ Simple search failed: {e}")
        return "", []

def get_rag_stats():
    """Get RAG system statistics"""
    if RAG_ADAPTER is None:
        return {"error": "RAG system not initialized"}
    return RAG_ADAPTER.get_stats()

def clear_rag_cache():
    """Clear RAG cache"""
    if RAG_ADAPTER is None:
        return {"success": False, "message": "RAG not initialized"}
    deleted = RAG_ADAPTER.clean_expired()
    return {"success": True, "message": f"Cleared {deleted} entries"}

def rebuild_rag_index():
    """Rebuild FAISS index"""
    if RAG_ADAPTER is None:
        return {"success": False, "message": "RAG not initialized"}
    RAG_ADAPTER.rebuild_index()
    return {"success": True, "message": "Index rebuilt"}

def search_rag_knowledge(query: str, top_k: int = 5):
    """Direct search of RAG knowledge"""
    if RAG_ADAPTER is None:
        return {"error": "RAG not initialized"}
    
    context, results = RAG_ADAPTER.get_context_for_llm(query, top_k)
    
    return {
        "query": query,
        "results": [
            {
                "content": r.chunk.content,
                "source": r.chunk.source_name,
                "score": r.score,
                "method": r.retrieval_method
            }
            for r in results
        ],
        "count": len(results)
    }

# ==================== DATABASE CLEANUP ====================
import atexit


def cleanup_database():
    """Clean up database on exit"""
    try:
        db.clean_expired_cache()
        db.close()
        logging.info("✅ Database closed cleanly")
    except Exception as e:
        logging.error(f"⚠️ Database cleanup error: {e}")


atexit.register(cleanup_database)

# yan.py - Add these diagnostic functions

def diagnose_cache_system():
    """
    Complete diagnostic of cache system
    """
    if not RAG_ADAPTER:
        return "❌ RAG_ADAPTER not initialized"
    
    try:
        cursor = RAG_ADAPTER.rag_db.conn.cursor()
        
        report = "🔍 **Cache System Diagnostic Report**\n"
        report += "=" * 60 + "\n\n"
        
        # 1. Check knowledge_chunks table
        cursor.execute("""
            SELECT source_type, COUNT(*) as count
            FROM knowledge_chunks
            GROUP BY source_type
        """)
        
        report += "📦 **Knowledge Chunks by Type:**\n"
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                report += f"  • {row['source_type']}: {row['count']} chunks\n"
        else:
            report += "  (empty)\n"
        
        # 2. Check query_cache table
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN expiry_timestamp > ? THEN 1 ELSE 0 END) as valid
            FROM query_cache
        """, (time.time(),))
        
        row = cursor.fetchone()
        report += f"\n🔑 **Query Cache:**\n"
        report += f"  • Total entries: {row['total']}\n"
        report += f"  • Valid (not expired): {row['valid']}\n"
        
        # 3. Show recent web searches
        cursor.execute("""
            SELECT source_name, timestamp
            FROM knowledge_chunks
            WHERE source_type = 'web_search'
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        
        report += f"\n🌐 **Recent Web Searches (last 5):**\n"
        rows = cursor.fetchall()
        if rows:
            for i, row in enumerate(rows, 1):
                age = time.time() - row['timestamp']
                report += f"  {i}. {row['source_name'][:50]} (age: {format_age(age)})\n"
        else:
            report += "  (no web searches cached)\n"
        
        # 4. Show recent cached queries
        cursor.execute("""
            SELECT query, timestamp
            FROM query_cache
            WHERE expiry_timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 5
        """, (time.time(),))
        
        report += f"\n🔍 **Recent Cached Queries (last 5):**\n"
        rows = cursor.fetchall()
        if rows:
            for i, row in enumerate(rows, 1):
                age = time.time() - row['timestamp']
                report += f"  {i}. '{row['query'][:50]}' (age: {format_age(age)})\n"
        else:
            report += "  (no queries cached)\n"
        
        # 5. Check PDF documents
        cursor.execute("""
            SELECT source_type, COUNT(*) as count
            FROM knowledge_chunks
            WHERE source_type = 'pdf_document'
        """)
        
        row = cursor.fetchone()
        pdf_count = row['count'] if row else 0
        
        report += f"\n📄 **PDF Documents:**\n"
        report += f"  • Total chunks: {pdf_count}\n"
        
        if pdf_count > 0:
            cursor.execute("""
                SELECT DISTINCT source_name
                FROM knowledge_chunks
                WHERE source_type = 'pdf_document'
            """)
            
            pdf_files = cursor.fetchall()
            report += f"  • Unique files: {len(pdf_files)}\n"
            report += "  • Files:\n"
            for pdf in pdf_files[:5]:
                report += f"    - {pdf['source_name']}\n"
        
        # 6. Database file size
        import os
        db_path = RAG_ADAPTER.rag_db.db_path
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            report += f"\n💾 **Database File:**\n"
            report += f"  • Path: {db_path}\n"
            report += f"  • Size: {size_mb:.2f} MB\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
        
    except Exception as e:
        import traceback
        return f"❌ Diagnostic failed: {e}\n\n{traceback.format_exc()}"
        # 1. Run diagnostics
        print(diagnose_cache_system())

# ================= RAG ENGINE =================
# Initialize RAG Integration Adapter
RAG_ADAPTER = None

def initialize_rag_adapter():
    """Initialize the new RAG system"""
    global RAG_ADAPTER
    
    if emb_model is None:
        debug_log("⚠️ Embedding model missing – RAG disabled")
        return None
    
    try:
        RAG_ADAPTER = RAGIntegrationAdapter(
            cache_dir=CACHE_DIR / "rag",
            embedding_model=emb_model
        )
        
        debug_log("✅ RAG Integration Adapter initialized")
        return RAG_ADAPTER
        
    except Exception as e:
        debug_log(f"❌ RAG adapter init failed: {e}")
        return None



def initialize_rag_metadata():
    """
    Initialize or refresh RAG metadata on startup
    Safe to call multiple times
    """
    global rag_metadata
    
    try:
        debug_log("🔄 Initializing RAG metadata...")
        
        # Ensure rag_metadata exists
        if 'rag_metadata' not in globals():
            globals()['rag_metadata'] = {
                "total_chunks": 0,
                "total_documents": 0,
                "last_updated": None,
            }
        
        # Refresh from current data
        refresh_rag_metadata()
        
        debug_log(f"✅ RAG metadata initialized: {rag_metadata}")
        return True
        
    except Exception as e:
        debug_log(f"⚠️ RAG metadata initialization failed: {e}")
        
        # Ensure minimum viable state
        if 'rag_metadata' not in globals():
            globals()['rag_metadata'] = {
                "total_chunks": 0,
                "total_documents": 0,
                "last_updated": None,
            }
        
        return False

# Initialize on startup
RAG_ADAPTER = initialize_rag_adapter()


# Initialize metadata after RAG_ADAPTER is created
if RAG_ADAPTER:
    initialize_rag_metadata()
    debug_log("✅ RAG metadata initialized on startup")
    debug_log("✅ Enhanced RAG retrieval method added")

if __name__ == "__main__":
    print("✅ Core modules loaded successfully!")
    print("🚀 Ready to use with Flask backend")
    print(f"📚 {len(set(persistent_data.get('sources', [])))} PDFs auto-loaded")
    print("🔐 Permission system active - web search requires user approval")
    debug_log("🔐 Permission system active - web search requires user approval")

        # Initialize RAG metadata
    #initialize_rag_metadata()
    
    # Show status
    pdf_sources = persistent_data.get('sources', [])
    if pdf_sources:
        unique_pdfs = len(set(pdf_sources))
        print(f"📚 {unique_pdfs} PDFs auto-loaded")
        print(f"📦 {len(pdf_sources)} chunks available")
        



# ============================================================
# ALSO ADD: Safe getter function for rag_metadata
# ============================================================

def get_rag_metadata():
    """
    Safe getter for rag_metadata with fallback
    """
    global rag_metadata
    
    try:
        if 'rag_metadata' not in globals():
            initialize_rag_metadata()
        
        return rag_metadata
    
    except Exception as e:
        debug_log(f"⚠️ Error getting rag_metadata: {e}")
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "last_updated": None,
        }





# ================= HYBRID RAG =================

HYBRID_RAG = None

if RAG_ENGINE and emb_model:
    HYBRID_RAG = HybridRAGManager(RAG_ENGINE, emb_model)
    logging.info("✅ Hybrid RAG Manager initialized")


# ============================================================
# ENHANCED RAG RETRIEVAL WITH CONVERSATION AWARENESS
# ============================================================

def enhanced_rag_retrieval(
    query: str,
    chat_history: list = None,
    top_k: int = 5,
    min_score: float = 0.20,  # ✅ LOWERED from 0.30
    use_conversation_context: bool = True
) -> Tuple[str, List, float]:
    """
    Enhanced RAG retrieval with aggressive relevance boosting
    """
    if not RAG_ADAPTER:
        debug_log("⚠️ RAG_ADAPTER not available")
        return "", [], 0.0
    
    try:
        # ============================================================
        # STEP 1: Smart query preprocessing with EXPANSION
        # ============================================================
        expanded_query = query
        query_variations = [query]  # Start with original
        
        # Context expansion for anaphora
        if use_conversation_context and chat_history:
            recent_context = []
            for entry in chat_history[-1:]:
                user_msg = entry.get("user", "")
                if user_msg:
                    recent_context.append(user_msg)
            
            anaphora_words = ["it", "that", "this", "them", "these", "those"]
            query_words = query.lower().split()
            has_anaphora = any(word in query_words for word in anaphora_words)
            
            if has_anaphora and recent_context and len(query_words) < 8:
                prev_query = recent_context[-1]
                expanded_query = f"{prev_query} {query}"
                debug_log(f"🔍 Context expanded: '{expanded_query}'")
        
        # ✅ CRITICAL FIX: Generate query variations for better matching
        # This helps match "full meaning of mcp" to "mcp" documents
        try:
            if llm and len(query.split()) <= 15:  # Only for reasonable query lengths
                debug_log(f"🔄 Generating query variations...")
                variations = expand_query_smart(llm, expanded_query, max_queries=2)
                if variations and len(variations) > 1:
                    query_variations = variations
                    debug_log(f"✅ Using {len(query_variations)} query variations:")
                    for i, var in enumerate(query_variations, 1):
                        debug_log(f"   {i}. {var}")
                else:
                    debug_log(f"⚠️ No variations generated, using original only")
            else:
                debug_log(f"⏭️ Skipping expansion (query too long or LLM unavailable)")
        except Exception as e:
            debug_log(f"⚠️ Query expansion failed: {e}, using original")
            query_variations = [expanded_query]
        
        # Extract key terms for better matching
        key_terms = extract_key_terms(expanded_query)
        debug_log(f"🎯 Key terms: {key_terms}")
        
        # ============================================================
        # STEP 2: Multi-method retrieval with QUERY VARIATIONS
        # ============================================================
        all_results = []
        results_by_query = {}  # Track which variation found which results

        # Search using ALL query variations
        for idx, query_var in enumerate(query_variations):
            debug_log(f"\n🔍 Searching with variation {idx + 1}: '{query_var}'")
            
            # Method 1: Hybrid search (primary)
            hybrid_results = RAG_ADAPTER.rag_db.retrieve(
                query_var,
                top_k=top_k * 3,
                method="hybrid"
            )
            
            if hybrid_results:
                debug_log(f"   📊 Hybrid: {len(hybrid_results)} results")
                existing_ids = {r.chunk.chunk_id for r in all_results}
                new_count = 0
                for r in hybrid_results:
                    if r.chunk.chunk_id not in existing_ids:
                        all_results.append(r)
                        existing_ids.add(r.chunk.chunk_id)
                        new_count += 1
                if new_count > 0:
                    debug_log(f"   ✅ Added {new_count} new results from hybrid")
            
            # Method 2: Vector search (semantic similarity)
            vector_results = RAG_ADAPTER.rag_db.retrieve(
                query_var,
                top_k=top_k * 3,
                method="vector"
            )
            
            if vector_results:
                debug_log(f"   📊 Vector: {len(vector_results)} results")
                existing_ids = {r.chunk.chunk_id for r in all_results}
                new_count = 0
                for r in vector_results:
                    if r.chunk.chunk_id not in existing_ids:
                        all_results.append(r)
                        existing_ids.add(r.chunk.chunk_id)
                        new_count += 1
                if new_count > 0:
                    debug_log(f"   ✅ Added {new_count} new results from vector")
            
            # Method 3: vector search (keyword matching)
            vector_results = RAG_ADAPTER.rag_db.retrieve(
                query_var,
                top_k=top_k * 3,
                method="vector"
            )
            
            if vector_results:
                debug_log(f"   📊 vector: {len(vector_results)} results")
                existing_ids = {r.chunk.chunk_id for r in all_results}
                new_count = 0
                for r in vector_results:
                    if r.chunk.chunk_id not in existing_ids:
                        all_results.append(r)
                        existing_ids.add(r.chunk.chunk_id)
                        new_count += 1
                if new_count > 0:
                    debug_log(f"   ✅ Added {new_count} new results from vector")

        debug_log(f"\n📊 Total unique results from all variations: {len(all_results)}")

        if not all_results:
            debug_log(f"⚠️ No RAG results for query variations: {query_variations}")
            return "", [], 0.0

        debug_log(f"✅ Total unique candidates from all methods: {len(all_results)}")

        # ============================================================
        # STEP 2.5: Score-based filtering with FALLBACK
        # ============================================================
        min_score = 0.08  # ✅ LOWERED from 0.15
        debug_log(f"🔍 Filtering {len(all_results)} results (min_score: {min_score})")

        filtered_results = [r for r in all_results if r.score >= min_score]

        if not filtered_results and all_results:
            # FALLBACK: Use top results even if below threshold
            debug_log(f"⚠️ No results above {min_score}, using top 5 by score")
            filtered_results = sorted(all_results, key=lambda x: x.score, reverse=True)[:5]

        if not filtered_results:
            debug_log(f"❌ No results found even after fallback")
            return "", [], 0.0

        debug_log(f"✅ {len(filtered_results)} results after filtering")
        all_results = filtered_results  # Continue with filtered results  
        
        # ============================================================
        # STEP 3: SMART SCORING with key term boosting
        # ============================================================
        for result in all_results:
            content_lower = result.chunk.content.lower()
            
            # Boost score if key terms found
            term_matches = sum(1 for term in key_terms if term in content_lower)
            if term_matches > 0:
                boost = 0.15 * term_matches  # +15% per key term
                result.score = min(result.score + boost, 1.0)
                debug_log(f"   📈 Boosted {result.chunk.chunk_id[:8]} by {boost:.2f} ({term_matches} terms)")
        
        # ============================================================
        # STEP 4: Adaptive filtering (NOT strict)
        # ============================================================
        # Sort by score first
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Filter by minimum score
        filtered_results = [r for r in all_results if r.score >= min_score]
        
        if not filtered_results:
            # ✅ FALLBACK: Use top 3 results regardless of score
            filtered_results = all_results[:3]
            debug_log(f"📌 Using top {len(filtered_results)} results (below threshold)")
        
        # Take top_k
        filtered_results = filtered_results[:top_k]
        
        # ============================================================
        # STEP 5: Reranking (final relevance boost)
        # ============================================================
        filtered_results = rerank_results(expanded_query, filtered_results, key_terms)
        
        # ============================================================
        # STEP 6: Build rich context
        # ============================================================
        sources = set(r.chunk.source_name for r in filtered_results)
        debug_log(f"✅ Final: {len(filtered_results)} chunks from {len(sources)} source(s)")
        
        for source in sources:
            source_results = [r for r in filtered_results if r.chunk.source_name == source]
            avg_score = sum(r.score for r in source_results) / len(source_results)
            debug_log(f"  📄 {source}: {len(source_results)} chunks (avg: {avg_score:.3f})")
        
        # Build context with relevance indicators
        context_parts = []
        
        for i, r in enumerate(filtered_results, 1):
            relevance_label = "⭐ HIGH" if r.score >= 0.7 else "✓ RELEVANT" if r.score >= 0.4 else "• LOW"
            context_parts.append(
                f"[Source {i}: {r.chunk.source_name}] ({relevance_label} - {r.score:.2f})\n{r.chunk.content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate intelligently
        MAX_CONTEXT_LENGTH = 3000  # ✅ INCREASED from 2500
        if len(context) > MAX_CONTEXT_LENGTH:
            # Keep highest-scoring chunks
            context = context[:MAX_CONTEXT_LENGTH] + "\n\n[Additional content available but truncated for brevity...]"
            debug_log(f"✂️ Context truncated to {MAX_CONTEXT_LENGTH} chars")
        
        avg_confidence = sum(r.score for r in filtered_results) / len(filtered_results)
        debug_log(f"📊 Confidence: {avg_confidence:.3f}")
        
        return context, filtered_results, avg_confidence
        
    except Exception as e:
        debug_log(f"❌ RAG retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return "", [], 0.0

# ============================================================
# RAG HELPER FUNCTIONS (NEW)
# ============================================================

def extract_key_terms(query: str) -> List[str]:
    """
    Extract key terms from query for relevance boosting
    """
    import re
    
    # Remove common stop words
    stop_words = {
        'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
        'who', 'which', 'can', 'does', 'do', 'tell', 'me', 'about',
        'explain', 'describe', 'show', 'give', 'find', 'get', 'make'
    }
    
    # Extract words (alphanumeric + underscores)
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Filter out stop words and short words
    key_terms = [w for w in words if len(w) > 3 and w not in stop_words]
    
    return key_terms[:5]  # Top 5 key terms


def rerank_results(query: str, results: List, key_terms: List[str]) -> List:
    """
    Final reranking based on multiple relevance signals
    """
    if not results:
        return results
    
    query_lower = query.lower()
    
    for result in results:
        content_lower = result.chunk.content.lower()
        
        # Signal 1: Query term density
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        density_score = matches / len(query_words) if query_words else 0
        
        # Signal 2: Position of matches (earlier = better)
        first_match_pos = float('inf')
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                first_match_pos = min(first_match_pos, pos)
        
        position_score = 1.0 if first_match_pos < 100 else 0.5 if first_match_pos < 500 else 0.2
        
        # Signal 3: Content length (prefer substantial chunks)
        length_score = min(len(result.chunk.content) / 500, 1.0)
        
        # Combined reranking score
        rerank_boost = (
            0.4 * density_score +
            0.3 * position_score +
            0.3 * length_score
        ) * 0.2  # Max 20% boost
        
        result.score = min(result.score + rerank_boost, 1.0)
    
    # Sort by new scores
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results


# ============================================================
# 🎨 SENTIENT EMOJI SYSTEM
# ============================================================
from collections import deque
from functools import lru_cache


class EmojiPersonality:
    """
    OPTIMIZED: 5-10x faster emoji selection with pre-compiled patterns and caching.
    """

    # Topic patterns with emojis
    TOPIC_PATTERNS = {
        "science_tech": (
            r"\b(science|data|research|tech|technology|digital|ai|artificial intelligence|machine learning|space|math|physics|engine|system|logic|robot|code|software|hardware|algorithm|network)\b",
            ["🔬","💻","🛰️","🧬","🧪","⚙️","🤖","🧠","📡","🔍","🖥️","📊"]
        ),
        "programming": (
            r"\b(code|debug|python|javascript|java|compile|error|exception|syntax|api|backend|frontend|database|server|framework|deploy)\b",
            ["🐍","💻","🛠️","📦","🧩","⚡","🔧","📁","📜","🖥️","🚀"]
        ),
        "gaming": (
            r"\b(game|gaming|player|level|quest|mission|battle|fps|rpg|strategy|console|xbox|playstation|nintendo|zelda|horizon|elden|fortnite|minecraft|pubg|call of duty|gta|grand theft auto)\b",
            ["🎮","🕹️","🏆","🔥","⚔️","🎯","👾","🛡️","🎲","💥"]
        ),
        "nature_env": (
            r"\b(nature|weather|climate|environment|plant|animal|world|earth|ocean|green|energy|forest|wildlife|mountain|river)\b",
            ["🌿","🌍","☀️","🌊","🌲","🐾","🌻","🍃","⛰️","🌦️"]
        ),
        "business_finance": (
            r"\b(money|business|market|finance|growth|career|work|success|company|strategy|investment|profit|startup|economy|sales)\b",
            ["📈","💼","💰","🤝","🚀","📊","🏦","💹","🧾","📉"]
        ),
        "education_learning": (
            r"\b(learn|study|school|university|exam|knowledge|course|lesson|teach|training|skill|academic)\b",
            ["📚","🎓","📝","📖","🧠","✏️","📘","🏫","📑"]
        ),
        "health_wellbeing": (
            r"\b(health|fitness|food|mind|body|rest|wellness|safety|protection|balance|diet|exercise|sleep|medical)\b",
            ["🥗","💪","🧠","🧘","🍎","🛡️","🏥","💊","❤️","🩺"]
        ),
        "humanities_arts": (
            r"\b(art|music|history|culture|society|literature|design|philosophy|creative|poetry|theatre|film|movie)\b",
            ["🎨","🎭","🎻","📜","🏛️","🖋️","🎬","🎼","🖼️"]
        ),
        "storytelling": (
            r"\b(story|narrative|plot|character|tale|emotional|journey|survival|connection|experience)\b",
            ["📖","🎭","💔","✨","🌟","💫","🎬"]
        ),
        "adventure": (
            r"\b(adventure|explore|quest|journey|discover|open.world|vast|beautiful)\b",
            ["🗺️","🌄","⛰️","🧭","🎒","✨","🌍"]
        ),
        "action": (
            r"\b(action|combat|fight|battle|mechanics|gameplay|hunt|rpg|shooter|multiplayer|campaign)\b",
            ["⚔️","🎯","💥","🔥","🛡️","⚡","🏹"]
        ),
        "creative_build": (
            r"\b(build|create|sandbox|blocky|craft|construct|resource|treasure)\b",
            ["🏗️","🧱","⛏️","💎","🎨","🔨"]
        ),
        "survival": (
            r"\b(survive|survival|last one standing|realistic|intense)\b",
            ["🎯","💪","🏆","⚠️"]
        ),
        "communication": (
            r"\b(talk|discuss|media|news|write|speak|question|social|connection|community|message|email|chat|conversation)\b",
            ["🗣️","📱","🌐","📢","💬","✉️","📡","🤝"]
        ),
        "travel": (
            r"\b(travel|trip|journey|flight|hotel|vacation|tour|explore|destination|adventure)\b",
            ["✈️","🌍","🧳","🏖️","🗺️","🚗","🚆","🏕️"]
        ),
        "relationships": (
            r"\b(friend|family|relationship|partner|love|support|trust|together|team)\b",
            ["❤️","🤝","👨‍👩‍👧‍👦","💞","💬","🌟","🫶"]
        )
    }

    EMOTIONS = {
        "positive": (
            r"\b(good|great|awesome|happy|excellent|perfect|love|enjoy|solved|done|success|amazing|fantastic|brilliant|incredible|masterpiece|critically acclaimed|popular|highly rated)\b",
            ["✨","✅","🎉","🌟","👍","💯","🔥","🙌","🥳"]
        ),
        "thoughtful": (
            r"\b(think|understand|consider|why|how|analyze|perspective|concept|idea|reflect|explore|evaluate|thought.provoking)\b",
            ["🤔","💭","💡","🧩","🔍","📌","🧐","📎"]
        ),
        "cautionary": (
            r"\b(but|however|risk|issue|problem|error|careful|warning|difficult|complex|challenge|concern|keep in mind)\b",
            ["⚠️","😬","👀","❗","🛑","🚧","🔎"]
        ),
        "curious": (
            r"\b(curious|interesting|wonder|discover|explore|intriguing|mysterious)\b",
            ["🧐","🔎","✨","👁️","📡","🧠"]
        ),
        "serious": (
            r"\b(important|critical|essential|significant|major|key|vital)\b",
            ["📌","❗","🔑","⚖️","📍"]
        ),
        "engaging": (
            r"\b(engaging|compelling|captivating|immersive|rich|powerful|unique|innovative|addictive|colorful)\b",
            ["🎯","✨","🌟","💎","🔥"]
        ),
        "fun": (
            r"\b(fun|exciting|enjoy|classic|following)\b",
            ["🎉","😄","🎊","🌈"]
        )
    }

    POSITIONAL = {
        "opening": ["👋","✨","🎯","💡","🚀","🌟"],
        "closing": ["✅","🏁","🙌","👍","🎉","💬"],
        "generic": ["🔹","📍","✨","➡️"]
    }

    MODE_SETTINGS = {
        "formal": {"sentences_per_emoji": 5, "max_per_block": 1, "list_emojis": 1},
        "balanced": {"sentences_per_emoji": 3, "max_per_block": 2, "list_emojis": 1},
        "playful": {"sentences_per_emoji": 2, "max_per_block": 3, "list_emojis": 1}
    }

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.rotation_index = {}
        self.used_emojis = deque(maxlen=15)
        
        # ⚡ OPTIMIZATION 1: Pre-compile all regex patterns
        self.compiled_topics = {
            name: (re.compile(pattern, re.IGNORECASE), emojis)
            for name, (pattern, emojis) in self.TOPIC_PATTERNS.items()
        }
        
        self.compiled_emotions = {
            name: (re.compile(pattern, re.IGNORECASE), emojis)
            for name, (pattern, emojis) in self.EMOTIONS.items()
        }
        
        # ⚡ OPTIMIZATION 2: Create quick keyword lookup for fast path
        self._build_keyword_index()
        
        self.reset()

    def _build_keyword_index(self):
        """Build a simple keyword index for ultra-fast matching of common terms."""
        self.keyword_map = {}
        
        # Extract first keyword from each pattern for quick checks
        common_keywords = {
            "code": "programming",
            "game": "gaming",
            "science": "science_tech",
            "money": "business_finance",
            "learn": "education_learning",
            "health": "health_wellbeing",
            "art": "humanities_arts",
        }
        
        self.keyword_map = common_keywords

    def reset(self):
        self.emoji_count = 0
        self.rotation_index = {}
        # Clear the cache when resetting
        self.get_emoji_cached.cache_clear()

    @lru_cache(maxsize=256)  # ⚡ OPTIMIZATION 3: Cache emoji lookups
    def get_emoji_cached(self, text_hash: str, pos: str = "generic") -> str:
        """
        Cached emoji selection - avoids re-processing identical text.
        Note: This uses the hash because you can't cache with unhashable types.
        """
        # This will be called by get_emoji with the actual text
        # We just use this for the caching mechanism
        return None  # Placeholder, actual logic in get_emoji

    def get_emoji(self, text: str, pos: str = "generic") -> str:
        """
        OPTIMIZED: Get contextually appropriate emoji with multiple speedups.
        """
        text_lower = text.lower()
        
        # ⚡ FAST PATH: Check simple keywords first
        for keyword, category in self.keyword_map.items():
            if keyword in text_lower:
                compiled_pattern, pool = self.compiled_topics[category]
                emoji = self._rotate(pool)
                if emoji not in self.used_emojis:
                    self.used_emojis.append(emoji)
                    self.emoji_count += 1
                    return emoji
        
        # ⚡ NORMAL PATH: Use pre-compiled patterns
        candidates = []
        
        # 1. Check topics with compiled patterns
        for _, (compiled_pattern, pool) in self.compiled_topics.items():
            if compiled_pattern.search(text_lower):
                candidates.append((self._rotate(pool), 0.95))
                if len(candidates) >= 3:  # ⚡ EARLY EXIT: Stop after 3 good matches
                    break

        # 2. Check emotions (only if we don't have enough candidates)
        if len(candidates) < 2:
            for _, (compiled_pattern, pool) in self.compiled_emotions.items():
                if compiled_pattern.search(text_lower):
                    candidates.append((self._rotate(pool), 0.85))
                    if len(candidates) >= 3:  # ⚡ EARLY EXIT
                        break

        # 3. Positional fallback
        if not candidates:
            candidates.append((self._rotate(self.POSITIONAL.get(pos, self.POSITIONAL["generic"])), 0.5))

        # Sort by weight and select best unused emoji
        candidates.sort(key=lambda x: x[1], reverse=True)

        for emoji, _ in candidates:
            if emoji not in self.used_emojis:
                self.used_emojis.append(emoji)
                self.emoji_count += 1
                return emoji

        # If all are recent, use best one anyway
        emoji = candidates[0][0]
        self.used_emojis.append(emoji)
        self.emoji_count += 1
        return emoji

    def _rotate(self, pool: List[str]) -> str:
        key = tuple(pool)
        idx = self.rotation_index.get(key, 0)
        self.rotation_index[key] = (idx + 1) % len(pool)
        return pool[idx % len(pool)]


class SmartEmojiFormatter:
    """
    OPTIMIZED: Handles intelligent emoji insertion with better performance.
    """
    
    def __init__(self, handler: EmojiPersonality):
        self.handler = handler
        
        # ⚡ Pre-compile regex patterns used in formatting
        self.list_line_pattern = re.compile(r'^\s*(\d+\.|\*|-|•)\s+')
        self.colon_line_pattern = re.compile(r'^\s*[A-Z][^:]{3,40}:\s+')
        self.traditional_list_pattern = re.compile(r'^(\s*)(\d+\.|\*|-|•)(\s+)(.*)$')
        self.colon_list_pattern = re.compile(r'^(\s*)([A-Z][^:]{3,40}:)(\s*)(.*)$')
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')

    def format(self, text: str, query: str = "") -> str:
        if not text:
            return text
        
        self.handler.reset()
        
        # Split into blocks
        if '\n\n' in text:
            blocks = text.split('\n\n')
        else:
            blocks = self._smart_block_split(text)
        
        formatted_blocks = []
        
        for block in blocks:
            if not block.strip():
                formatted_blocks.append(block)
                continue
            
            # Check if this block is a list
            if self._is_list_block(block):
                formatted_block = self._format_list(block, query)
            else:
                formatted_block = self._format_paragraph(block, query)
            
            formatted_blocks.append(formatted_block)
        
        # Rejoin with the same separator
        if '\n\n' in text:
            return '\n\n'.join(formatted_blocks)
        else:
            return '\n'.join(formatted_blocks)
    
    def _smart_block_split(self, text: str) -> List[str]:
        """Split text into blocks when there are no double newlines."""
        lines = text.split('\n')
        blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            current_block.append(line)
            
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                
                is_current_list = self._is_list_line(line)
                is_next_list = self._is_list_line(next_line)
                
                if is_current_list != is_next_list:
                    blocks.append('\n'.join(current_block))
                    current_block = []
        
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _is_list_line(self, line: str) -> bool:
        """Check if a single line is a list item using pre-compiled patterns."""
        return bool(self.list_line_pattern.match(line)) or \
               bool(self.colon_line_pattern.match(line))
    
    def _is_list_block(self, block: str) -> bool:
        """Check if a block contains list items."""
        lines = block.strip().split('\n')
        
        traditional_list_count = sum(1 for line in lines 
                                     if self.list_line_pattern.match(line))
        
        colon_list_count = sum(1 for line in lines 
                               if self.colon_line_pattern.match(line))
        
        return (traditional_list_count > 0) or (colon_list_count >= 2)
    
    def _format_list(self, block: str, query: str) -> str:
        """Format a list block with emojis."""
        lines = block.split('\n')
        formatted_lines = []
        settings = self.handler.MODE_SETTINGS[self.handler.mode]
        
        for line in lines:
            if not line.strip():
                formatted_lines.append(line)
                continue
            
            # Use pre-compiled patterns
            traditional_match = self.traditional_list_pattern.match(line)
            colon_match = self.colon_list_pattern.match(line)
            
            if traditional_match:
                indent, marker, space, content = traditional_match.groups()
                segments = self._split_list_content(content)
                formatted_content = self._add_emojis_to_segments(segments, query, settings["list_emojis"])
                line = f"{indent}{marker}{space}{formatted_content}"
            
            elif colon_match:
                indent, title_with_colon, space, content = colon_match.groups()
                segments = self._split_list_content(content)
                formatted_content = self._add_emojis_to_segments(segments, query, settings["list_emojis"])
                line = f"{indent}{title_with_colon}{space}{formatted_content}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _add_emojis_to_segments(self, segments: List[str], query: str, emoji_count: int) -> str:
        """Add emojis to list item segments."""
        formatted_segments = []
        
        for i, segment in enumerate(segments):
            if i < emoji_count and segment.strip():
                emoji = self.handler.get_emoji(segment + " " + query, "generic")
                formatted_segments.append(f"{segment.rstrip()} {emoji}")
            else:
                formatted_segments.append(segment)
        
        return " ".join(formatted_segments)
    
    def _split_list_content(self, content: str) -> List[str]:
        """Split list item content into segments."""
        segments = []
        
        if '. ' in content and content.count('. ') <= 2:
            sentence_parts = [s.strip() + '.' for s in content.split('. ') if s.strip()]
            if sentence_parts and not content.endswith('.'):
                sentence_parts[-1] = sentence_parts[-1].rstrip('.')
            segments = sentence_parts
        
        elif '(' in content and ')' in content:
            paren_split = re.split(r'(\([^)]+\))', content)
            segments = [s.strip() for s in paren_split if s.strip()]
        
        elif ',' in content:
            segments = [s.strip() for s in content.split(',') if s.strip()]
        
        else:
            mid = len(content) // 2
            space_idx = content.find(' ', mid)
            if space_idx > 0:
                segments = [content[:space_idx], content[space_idx:]]
            else:
                segments = [content]
        
        return segments
    
    def _format_paragraph(self, paragraph: str, query: str) -> str:
        """Format a paragraph with emojis."""
        settings = self.handler.MODE_SETTINGS[self.handler.mode]
        
        # Use pre-compiled pattern for sentence splitting
        sentences = self.sentence_split_pattern.split(paragraph.strip())
        
        if len(sentences) <= 1:
            return paragraph
        
        formatted_sentences = []
        emojis_added = 0
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 15:
                formatted_sentences.append(sentence)
                continue
            
            pos = "generic"
            if i == 0:
                pos = "opening"
            elif i == len(sentences) - 1 and len(sentences) > 3:
                pos = "closing"
            
            should_add = (
                i % settings["sentences_per_emoji"] == 0 and
                emojis_added < settings["max_per_block"]
            )
            
            if should_add:
                emoji = self.handler.get_emoji(sentence + " " + query, pos)
                sentence = f"{sentence.rstrip()} {emoji}"
                emojis_added += 1
            
            formatted_sentences.append(sentence)
        
        return ' '.join(formatted_sentences)


# ========================================
# USAGE FUNCTION (same API as before)
# ========================================

def add_contextual_emojis(text: str, query: str = "", mode: str = "playful") -> str:
    """
    Add contextual emojis to text with list-aware formatting.
    
    Args:
        text: The text to format
        query: User query for context
        mode: 'formal', 'balanced', or 'playful'
    
    Returns:
        Formatted text with contextually relevant emojis
    """
    handler = EmojiPersonality(mode)
    formatter = SmartEmojiFormatter(handler)
    return formatter.format(text, query)


# ============================================================
# ENHANCED HUMANIZATION SYSTEM - Improved & Hardened
# ============================================================
import re
import random
from typing import Optional, List

ASSISTANT_NAME = "Project Elixer"

# Minimal emoji use – real people don't overuse them
EMOJI_REGEX = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]')

def humanize_response(text: str) -> str:
    """
    Simplified humanization wrapper - calls sentient version
    
    Args:
        text: Raw response text
    
    Returns:
        Humanized response
    """
    if not text or len(text.strip()) < 3:
        return text
    
    # Get chat history safely
    try:
        chat_history = persistent_data.get("chat_history", [])
    except:
        chat_history = []
    
    # Call the full sentient version
    return humanize_response_sentient(text, user_msg=None, chat_history=chat_history)

def humanize_response_sentient(
    text: str, 
    user_msg: Optional[str] = None, 
    chat_history: List = None
) -> str:
    """
    Personality-aware humanization - respects AI personality settings
    """
    if not text or len(text.strip()) < 3:
        return text
    
    # Get personality profile to determine how much to humanize
    try:
        from yan import personality
        stats = personality.get_profile_stats("default")
        warmth = stats.get("warmth", 0.7)
        verbosity = stats.get("verbosity", 0.5)
    except:
        # Fallback if personality not available
        warmth = 0.7
        verbosity = 0.5
    
    # Check user emoji preference
    use_emojis = user_manager.get_user_data("preferences.use_emojis", True)
    
    # ============================================================
    # MINIMAL CLEANUP (always do this)
    # ============================================================
    text = remove_robotic_language(text)
    text = make_conversational(text)  # Basic contractions
    
    # ============================================================
    # WARMTH-BASED FEATURES (only if warmth > 0.5)
    # ============================================================
    if warmth > 0.5:
        # Handle special greetings
        if user_msg and re.match(r'^(hi|hey|hello|sup|yo|howdy)[\s!?]*$', user_msg.lower()):
            user_name = user_manager.get_user_data("name")
            name_part = f" {user_name}" if user_name else ""
            return f"Hey{name_part}! What's up?"
        
        # Thanks responses
        if user_msg and re.search(r'\b(thanks|thank you|thx)\b', user_msg.lower()):
            return "No problem! 😊" if use_emojis else "No problem!"
        
        # Only add emojis if warm AND user wants them
        # First strip any emojis the LLM already put in to avoid doubles
        if use_emojis and warmth > 0.6:
            text = re.sub(r'([\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF])\s*\1', r'\1', text)  # remove consecutive dupes
            text = re.sub(r'([\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF])\s+([\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF])', r'\1', text)  # remove pairs
            text = add_contextual_emojis(text, user_msg or "")
    
    # ============================================================
    # VERBOSITY-BASED FEATURES (only if verbose)
    # ============================================================
    if verbosity > 0.6:
        # Add natural flow/fillers only if being detailed anyway
        text = add_natural_flow(text, user_msg or "")
    
    # ============================================================
    # PRESERVE CODE BLOCKS (always)
    # ============================================================
    if "```" in text:
        text = humanize_code_response(text)
    
    return text.strip()

def remove_robotic_language(text: str) -> str:
    """
    Strip corporate / robotic phrasing – EXPANDED
    """
    replacements = {
        # Ultra-formal phrases
        r'\bI would be happy to\b': "I can",
        r'\bI would be glad to\b': "I'll",
        r'\bI am pleased to\b': "I'll",
        r'\ballow me to\b': "let me",
        r'\bpermit me to\b': "let me",
        r'\bI shall\b': "I'll",

        # Corporate speak
        r'\bIt is important to note that\b': "Just so you know,",
        r'\bIt should be noted that\b': "Heads up –",
        r'\bIt is worth mentioning that\b': "Also,",
        r'\bPlease be aware that\b': "FYI,",
        r'\bKindly note that\b': "FYI,",
        r'\bPlease note that\b': "Note:",

        # Transition overkill
        r'\bFurthermore,\b': "Also,",
        r'\bMoreover,\b': "Plus,",
        r'\bAdditionally,\b': "Also,",
        r'\bIn addition,\b': "Also,",
        r'\bConsequently,\b': "So,",
        r'\bTherefore,\b': "So,",
        r'\bThus,\b': "So,",
        r'\bHence,\b': "So,",
        r'\bAccordingly,\b': "So,",

        # Passive voice
        r'\bcan be done by\b': "you can do it by",
        r'\bmay be used to\b': "you can use it to",
        r'\bshould be implemented\b': "you should implement",
        r'\bmust be considered\b': "you need to consider",

        # AI self-references
        r'\bAs an AI,?\b': "",
        r'\bAs a language model,?\b': "",
        r'\bAs an artificial intelligence,?\b': "",
        r'\bI don\'t have personal (experiences?|opinions?|feelings?)\b': "",
        r'\bI cannot (feel|experience)\b': "I can't",

        # Over-apologizing
        r'\bI apologize for any confusion,?\b': "",
        r'\bI\'m sorry for the confusion,?\b': "",
        r'\bI apologize,?\b': "",
        r'\bI\'m sorry,?\b': "",
        
        # Hedging (sometimes useful, but often overused)
        r'\bIt seems that\b': "",
        r'\bIt appears that\b': "",
        r'\bIt would seem\b': "",
        
        # Unnecessary qualifiers
        r'\bvery much\b': "really",
        r'\bquite simply\b': "basically",
        r'\bbasically speaking\b': "basically",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def make_conversational(text: str) -> str:
    """
    Convert formal writing to natural speech – ENHANCED
    """
    # Remove AI self-references
    text = re.sub(r'\b(The AI|This AI|The assistant)\b', "I", text, flags=re.I)
    text = re.sub(r'\b(The model|This model)\b', "I", text, flags=re.I)

    # Contractions (essential for natural speech)
    contractions = {
        r'\bI am\b': "I'm",
        r'\byou are\b': "you're",
        r'\bit is\b': "it's",
        r'\bthat is\b': "that's",
        r'\bwhat is\b': "what's",
        r'\bwho is\b': "who's",
        r'\bdo not\b': "don't",
        r'\bdoes not\b': "doesn't",
        r'\bcannot\b': "can't",
        r'\bwill not\b': "won't",
        r'\bshould not\b': "shouldn't",
        r'\bwould not\b': "wouldn't",
        r'\bcould not\b': "couldn't",
        r'\bhas not\b': "hasn't",
        r'\bhave not\b': "haven't",
        r'\bhad not\b': "hadn't",
        r'\bwas not\b': "wasn't",
        r'\bwere not\b': "weren't",
    }

    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text, flags=re.I)

    # Casual replacements
    text = re.sub(r'^To answer your question,', "So,", text, flags=re.M | re.I)
    text = re.sub(r'^In order to', "To", text, flags=re.M | re.I)
    text = re.sub(r'^With regards? to', "About", text, flags=re.M | re.I)
    text = re.sub(r'^Concerning\b', "About", text, flags=re.M | re.I)
    text = re.sub(r'^Regarding\b', "About", text, flags=re.M | re.I)

    return text


def add_natural_fillers(text: str, user_query: str) -> str:
    """
    Add occasional filler words for realism (but don't overdo it)
    """
    if random.random() > 0.25:  # Only 25% of the time
        return text
    
    query = user_query.lower() if user_query else ""
    
    # Technical questions get fewer fillers
    if any(word in query for word in ["code", "function", "algorithm", "implement"]):
        return text
    
    # Add fillers to explanations
    fillers = {
        r'\bthis is\b': lambda: random.choice(["this is basically", "this is pretty much", "this is"]),
        r'\bthat means\b': lambda: random.choice(["that basically means", "that means", "so that means"]),
        r'\byou need to\b': lambda: random.choice(["you'll need to", "you gotta", "you need to"]),
        r'\bthe reason is\b': lambda: random.choice(["the reason is basically", "the thing is", "the reason is"]),
    }
    
    for pattern, replacer in fillers.items():
        if random.random() < 0.4:  # 40% chance per pattern
            text = re.sub(pattern, replacer(), text, count=1, flags=re.I)
    
    return text


def vary_sentence_starters(text: str) -> str:
    """
    Vary how sentences begin to avoid repetition
    """
    # Split into sentences
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Track repeated starters
    starters = {}
    result = []
    
    for i, part in enumerate(sentences):
        if i % 2 == 0:  # Actual sentences (not separators)
            # Get first word
            match = re.match(r'^(\w+)', part.strip())
            if match:
                first_word = match.group(1).lower()
                
                # If we've seen this starter 2+ times, vary it
                if first_word in starters and starters[first_word] >= 2:
                    variations = {
                        'this': ['this', 'it', 'that'],
                        'the': ['the', 'this', 'that'],
                        'you': ['you', 'you can', "you'll"],
                        'it': ['it', 'this', 'that'],
                    }
                    
                    if first_word in variations:
                        replacement = random.choice([w for w in variations[first_word] if w != first_word])
                        part = re.sub(r'^\w+', replacement.capitalize() if part[0].isupper() else replacement, part, count=1)
                
                starters[first_word] = starters.get(first_word, 0) + 1
        
        result.append(part)
    
    return ''.join(result)


def add_natural_flow(text: str, user_query: str) -> str:
    """
    Add casual openers – IMPROVED with more variety
    """
    if len(text.split()) < 15 or random.random() > 0.35:
        return text

    # Don't add if already starts casually
    if re.match(r'^(So|Alright|OK|Right|Yeah|Well|Basically|Look|Listen)', text, re.I):
        return text

    query = user_query.lower() if user_query else ""

    # Context-aware openers
    if "how" in query or "why" in query:
        openers = ["Alright, so", "OK so", "Right, so", "So basically", "Well,"]
    elif "what" in query:
        openers = ["So", "Basically", "Alright", "OK so", "Well,"]
    elif any(word in query for word in ["confused", "don't understand", "stuck"]):
        openers = ["Okay, let me explain.", "Alright, here's the deal.", "So here's what's up."]
    elif any(word in query for word in ["code", "program", "function"]):
        openers = ["So", "Alright", "Here's the thing:"]


    opener = random.choice(openers)
    
    # Handle punctuation
    if opener.endswith(('.', ':')):
        return f"{opener} {text[0].upper()}{text[1:]}"
    else:
        return f"{opener}, {text[0].lower()}{text[1:]}"


def maybe_add_closer(text: str, user_query: str = "") -> str:
    """
    Add a casual closer – MORE VARIED
    """
    # Don't add if text is short or already has a question/exclamation
    if len(text.split()) < 20 or text.rstrip().endswith(('?', '!')):
        return text

    # Only add 25% of the time
    if random.random() > 0.25:
        return text

    query = user_query.lower() if user_query else ""

    # Context-aware closers
    if any(word in query for word in ["code", "program", "function", "implement"]):
        closers = [
            " Make sense?",
            " Got it?",
            " Let me know if that's unclear.",
            " Holler if you need help.",
            " Feel free to ask if something's off.",
        ]
    elif any(word in query for word in ["explain", "what", "how", "why"]):
        closers = [
            " Does that clear it up?",
            " Make sense now?",
            " Let me know if you want me to clarify anything.",
            " Shout if that's confusing.",
        ]
    else:
        closers = [
            " Make sense?",
            " Does that help?",
            " Let me know if you need more details.",
            " Hit me up if you have questions.",
            " Feel free to ask more!",
        ]

    return text + random.choice(closers)


def humanize_code_response(text: str) -> str:
    """
    Preserve code blocks, humanize surrounding explanation – ENHANCED
    """
    parts = re.split(r'(```[\s\S]*?```)', text)
    output = []

    for i, part in enumerate(parts):
        if part.startswith("```"):
            output.append(part)
        else:
            if part.strip():
                part = remove_robotic_language(part)
                part = make_conversational(part)
                
                # Vary "Here is" phrases
                part = re.sub(
                    r'\bHere is (the|a|an)\b',
                    lambda _: random.choice([
                        "Here's",
                        "Check out",
                        "Take a look at",
                        "Here's what I got:",
                        "Alright, here's",
                    ]),
                    part,
                    flags=re.I
                )
                
                # Vary "this code" phrases
                part = re.sub(
                    r'\bThis code\b',
                    lambda _: random.choice([
                        "This",
                        "This code",
                        "The code above",
                        "This snippet",
                    ]),
                    part,
                    flags=re.I
                )
            
            output.append(part)

    return ''.join(output)

def _is_repeated_question(query: str, chat_history: list, lookback: int = 5) -> bool:
    """
    Returns True if the user is re-asking a question from recent history.
    Prevents repeat questions from being routed as follow-ups.
    """
    if not chat_history:
        return False
    q_norm = query.lower().strip().rstrip("?").strip()
    for turn in chat_history[-lookback:]:
        prev = turn.get("user", "").lower().strip().rstrip("?").strip()
        # Exact or near-exact match
        if q_norm == prev:
            return True
        # High overlap (>80% of words shared)
        q_words = set(q_norm.split())
        p_words = set(prev.split())
        if q_words and p_words:
            overlap = len(q_words & p_words) / max(len(q_words), len(p_words))
            if overlap > 0.80:
                return True
    return False
# ============================================================
# 🚀 AUTO-ACTIVATION LAYER (SAFE, NON-DESTRUCTIVE)
# ============================================================

# This wrapper guarantees intent + anaphora + memory
# are ALWAYS applied before prompt construction.

def build_prompt(system_msg: str, query: str, context: str = "", use_gemini: bool = False) -> str:
    """
    Master prompt builder with universal follow-up handling.
    
    Handles ALL follow-up types equally: code, concepts, data, PDFs, web results, etc.
    
    Args:
        system_msg: System/persona instructions
        query: User's current question
        context: Optional pre-built context (PDF, web results, etc.)
        use_gemini: Whether using Gemini (skip chat history for privacy)
    
    Returns:
        Formatted prompt string for the LLM
    """
    
    try:
        # ============================================================
        # STEP 1: Load chat history (skip if using Gemini)
        # ============================================================
        if use_gemini:
            chat_history = []
            debug_log(f"🔷 Gemini mode: Skipping chat history for privacy")
        else:
            chat_history, _ = load_chat_history()  
            debug_log(f"📚 Loaded {len(chat_history)} messages from chat_history.json")
        
        # ============================================================
        # STEP 2: Detect query intent
        # ============================================================
        intent_result = INTENT_CLASSIFIER.classify(query, chat_history)
        intent_enum = intent_result.intent
        intent = intent_enum.value
        is_repeated = _is_repeated_question(query, chat_history)
        is_followup = (
            intent_enum in (QueryIntent.FOLLOWUP, QueryIntent.META_QUESTION)
            and not is_repeated  # repeated questions go to PRIORITY 3 for a fresh answer
        )
        if is_repeated:
            debug_log(f"🔁 Repeated question detected — routing as new question")
        is_meta = intent_enum == QueryIntent.META_QUESTION
        
        debug_log(f"🎯 Intent: {intent} | Follow-up: {is_followup}")
        
        # ============================================================
        # STEP 3: Build context based on query type
        # ============================================================
        
        final_context = ""
        
        # PRIORITY 1: Caller provided explicit context (PDF/RAG/Web)
        if context and context.strip():
            debug_log(f"📦 Using caller context ({len(context)} chars)")
            final_context = context
            
            # If it's a follow-up AND caller context exists, append conversation
            if is_followup and chat_history:
                conv_context = build_universal_conversation_context(
                    chat_history,
                    max_chars=1000,
                    max_turns=10
                )
                if conv_context:
                    final_context = f"{context}\n\n---\n\n**Previous Conversation:**\n{conv_context}"
                    debug_log(f"   ➕ Added conversation context")
        
        # PRIORITY 2: Follow-up question → need full conversation
        elif is_followup and intent_result.needs_context:
            debug_log(f"🔄 Building conversation context for follow-up")

            # Always prepend memory/name so it's never lost in follow-ups
            memory_context = build_memory_context_with_name(query)

            if not chat_history:
                debug_log("⚠️ Follow-up detected but no history!")
                final_context = memory_context  # at least inject name
            else:
                conv_context = build_universal_conversation_context(
                    chat_history,
                    max_chars=20000,
                    max_turns=20
                )

                if conv_context:
                    # Combine: memory first, then conversation
                    if memory_context:
                        final_context = f"{memory_context}\n\n---\n\n{conv_context}"
                    else:
                        final_context = conv_context
                    system_msg = enhance_system_for_followup(
                        system_msg,
                        query,
                        final_context
                    )
                    debug_log(f"   ✅ Built context ({len(final_context)} chars)")
                else:
                    final_context = memory_context
                    debug_log("   ⚠️ Context building failed, using memory only")
        
        # PRIORITY 3: New question → add memory/preferences only
        else:
            debug_log(f"🆕 New question - checking memory")
            memory_context = build_memory_context_with_name(query)
            
            if memory_context and memory_context.strip():
                final_context = memory_context
                debug_log(f"   ✅ Added memory ({len(final_context)} chars)")
        # ============================================================
        # STEP 3.5: Identity override — always inject name for "who am i" etc.
        # ============================================================
        if should_include_name_in_context(query):
            identity_memory = build_memory_context_with_name(query)
            if identity_memory:
                if final_context:
                    final_context = f"{identity_memory}\n\n---\n\n{final_context}"
                else:
                    final_context = identity_memory
                debug_log(f"👤 Identity query — name injected into context")
        # ============================================================
        # STEP 4: Format prompt
        # ============================================================
        formatted_prompt = build_prompt_with_format(
            system_msg,
            query,
            final_context,
            MODEL_FORMAT
        )
        
        debug_log(f"✅ Final prompt: {len(formatted_prompt)} chars")
        return formatted_prompt
        
            
    except Exception as e:
        debug_log(f"❌ Error in build_prompt: {e}")
        import traceback
        debug_log(f"Stack trace:\n{traceback.format_exc()}")
        
        # Safe fallback
        return build_prompt_with_format(system_msg, query, context, MODEL_FORMAT)

def build_universal_conversation_context(
    chat_history: List[dict],
    max_chars: int = 30000,
    max_turns: int = 50
) -> str:
    """
    Build conversation context that preserves ALL content types equally.
    
    Handles:
    - Code snippets (preserve formatting)
    - Explanations (full text)
    - Data/tables (preserve structure)
    - PDFs (preserve citations)
    - Web results (preserve sources)
    
    Args:
        chat_history: List of {"user": str, "ai": str} dicts
        max_chars: Maximum context length (default 50000 for ~100 messages)
        max_turns: Maximum conversation turns to include (default 100)
    
    Returns:
        Formatted conversation context string
    """
    if not chat_history:
        return ""
    
    # Start from most recent, work backwards
    recent_turns = list(reversed(chat_history[-max_turns:]))
    
    context_parts = []
    total_chars = 0
    
    for turn in recent_turns:
        user_msg = turn.get("user", "").strip()
        ai_msg = turn.get("ai", "").strip()
        
        if not user_msg or not ai_msg:
            continue
        
        # Format the turn
        turn_text = f"**User:** {user_msg}\n\n**Assistant:** {ai_msg}"
        turn_length = len(turn_text)
        
        # Check if adding this turn exceeds limit
        if total_chars + turn_length > max_chars:
            # Try to fit a truncated version
            remaining = max_chars - total_chars
            if remaining > 200:  # Only truncate if we have meaningful space
                # Truncate AI response, keep user query intact
                truncated_ai = ai_msg[:remaining - len(user_msg) - 100] + "\n[...truncated]"
                turn_text = f"**User:** {user_msg}\n\n**Assistant:** {truncated_ai}"
                context_parts.append(turn_text)
            break
        
        context_parts.append(turn_text)
        total_chars += turn_length
    
    # Reverse to chronological order
    context_parts.reverse()
    
    if not context_parts:
        return ""
    
    # Build final context with clear structure
    conversation = "\n\n---\n\n".join(context_parts)
    
    debug_log(f"📝 Built conversation context: {len(context_parts)} turns, {len(conversation)} chars")
    
    return conversation


def enhance_system_for_followup(
    system_msg: str,
    query: str,
    context: str
) -> str:
    """
    Enhance system message with follow-up awareness.
    
    Detects what type of content the follow-up is about and adds
    appropriate instructions to the system message.
    
    Args:
        system_msg: Original system message
        query: User's follow-up query
        context: Conversation context being provided
    
    Returns:
        Enhanced system message
    """
    query_lower = query.lower()
    
    # Detect what the follow-up is about
    is_about_code = any(kw in query_lower for kw in [
        "code", "function", "program", "script", "line", "syntax",
        "implement", "write", "create", "build"
    ])
    
    has_code_in_context = "```" in context
    
    is_pronoun_heavy = any(pronoun in query_lower for pronoun in [
        " it ", " this ", " that ", " these ", " those ", " they "
    ])
    
    # Build appropriate enhancement
    if is_about_code and has_code_in_context:
        enhancement = (
            "\n\n**FOLLOW-UP CONTEXT**:\n"
            "This is a follow-up about CODE from the previous response.\n"
            "The complete code is provided in the conversation history below.\n"
            "Reference it directly when explaining. Do NOT say 'I don't have the code'."
        )
    elif is_pronoun_heavy:
        enhancement = (
            "\n\n**FOLLOW-UP CONTEXT**:\n"
            "This question uses pronouns (it/this/that) referring to the previous conversation.\n"
            "The full context is provided below. Use it to resolve what the pronouns mean."
        )
    else:
        enhancement = (
            "\n\n**FOLLOW-UP CONTEXT**:\n"
            "This is a follow-up question building on the previous conversation.\n"
            "The relevant conversation history is provided below."
        )
    
    return system_msg + enhancement

def build_followup_context(query: str, chat_history: List, intent: str) -> str:
    """
    Build context for follow-up queries with anaphora resolution
    
    Args:
        query: Current user query
        chat_history: Recent conversation history
        intent: Detected intent ("followup" or other)
    
    Returns:
        Context string explaining what pronouns refer to
    """
    if not chat_history or intent != "followup":
        return ""
    
    # Get last exchange
    last_entry = chat_history[-1]
    last_user_msg = last_entry.get("user", "")
    last_ai_msg = last_entry.get("ai", "")
    
    if not last_user_msg:
        return ""
    
    # Build anaphora resolution context
    context = f"Previous question: {last_user_msg}\n"
    
    # Extract main topic from previous answer
    if last_ai_msg:
        # Get first 100 chars as topic summary
        topic_summary = last_ai_msg[:100].strip()
        if len(last_ai_msg) > 100:
            topic_summary += "..."
        
        context += f"Context: {topic_summary}\n"
    
    return context

# ============================================================
# 🔚 END AUTO-ACTIVATION
# =============================================================

