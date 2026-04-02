"""
Enterprise Personality System v6 - Final Production
Maximum reliability and observability

Final Optimizations:
✅ XPENDING-based lag monitoring per consumer group
✅ Time-based drift decay (not interaction-based)
✅ Cache bypass during event processing (no stale reads)
✅ Structured JSON logging with correlation IDs
✅ Graceful shutdown with in-flight event draining
"""

import os
import time
import threading
import logging
import json
import uuid
import signal
import sys
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from contextlib import contextmanager
from enum import Enum
import numpy as np

# Observability
from prometheus_client import Counter, Histogram, Gauge, Summary, Info

# Storage backends
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# ============================================================
# STRUCTURED JSON LOGGING
# ============================================================

class StructuredLogger:
    """Structured JSON logger with correlation ID support"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove default handlers
        self.logger.handlers = []
        
        # JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
    
    def _get_context(self) -> Dict:
        """Get current context (correlation ID, etc.)"""
        return getattr(threading.current_thread(), 'log_context', {})
    
    def _log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        context = self._get_context()
        
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
            'correlation_id': context.get('correlation_id', 'unknown'),
            'instance_id': Config.INSTANCE_ID,
            'thread': threading.current_thread().name,
            **kwargs
        }
        
        # Use appropriate log level
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
    
    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log('CRITICAL', message, **kwargs)


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON"""
    
    def format(self, record):
        return record.getMessage()


@contextmanager
def correlation_context(correlation_id: str = None):
    """Context manager for correlation ID"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    thread = threading.current_thread()
    old_context = getattr(thread, 'log_context', {})
    
    try:
        thread.log_context = {'correlation_id': correlation_id}
        yield correlation_id
    finally:
        thread.log_context = old_context


# Initialize structured logger
logger = StructuredLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Centralized configuration"""
    
    # Instance identification
    INSTANCE_ID = os.getenv("INSTANCE_ID", f"personality-{uuid.uuid4().hex[:8]}")
    
    # Similarity & Tone Thresholds
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
    MAX_TONE_DRIFT = float(os.getenv("MAX_TONE_DRIFT", "0.3"))
    WARMTH_ADJUSTMENT_RATE = float(os.getenv("WARMTH_ADJUSTMENT_RATE", "0.01"))
    VERBOSITY_ADJUSTMENT_RATE = float(os.getenv("VERBOSITY_ADJUSTMENT_RATE", "0.02"))
    PLAYFULNESS_ADJUSTMENT_RATE = float(os.getenv("PLAYFULNESS_ADJUSTMENT_RATE", "0.015"))
    
    # Playfulness Control
    PLAYFULNESS_ENABLED = os.getenv("PLAYFULNESS_ENABLED", "true").lower() == "true"
    SERIOUS_CONTEXT_THRESHOLD = float(os.getenv("SERIOUS_CONTEXT_THRESHOLD", "0.6"))
    CHEEKY_FREQUENCY_LIMIT = int(os.getenv("CHEEKY_FREQUENCY_LIMIT", "5"))  # per hour
    CHEEKY_FREQUENCY_WINDOW = int(os.getenv("CHEEKY_FREQUENCY_WINDOW", "3600"))  # seconds
    MAX_CHEEKY_HISTORY_SIZE = int(os.getenv("MAX_CHEEKY_HISTORY_SIZE", "100"))  # prevent unbounded growth
    
    # Time-based drift decay
    DRIFT_DECAY_RATE = float(os.getenv("DRIFT_DECAY_RATE", "0.001"))  # per second
    DRIFT_DECAY_ENABLED = os.getenv("DRIFT_DECAY_ENABLED", "true").lower() == "true"
    DRIFT_DECAY_INTERVAL = int(os.getenv("DRIFT_DECAY_INTERVAL", "3600"))  # seconds (1 hour)
    
    # Model Configuration
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Storage Configuration
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    
    # Distributed Cache
    CACHE_INVALIDATION_ENABLED = os.getenv("CACHE_INVALIDATION_ENABLED", "true").lower() == "true"
    CACHE_INVALIDATION_CHANNEL = os.getenv("CACHE_INVALIDATION_CHANNEL", "personality:cache:invalidate")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    CACHE_BYPASS_ON_EVENTS = True  # Always bypass cache during event processing
    
    # Event Stream Configuration
    EVENT_STREAM_BACKEND = os.getenv("EVENT_STREAM_BACKEND", "redis_streams")
    REDIS_STREAM_NAME = os.getenv("REDIS_STREAM_NAME", "personality:updates")
    REDIS_CONSUMER_GROUP = os.getenv("REDIS_CONSUMER_GROUP", "personality-workers")
    REDIS_DEAD_LETTER_STREAM = os.getenv("REDIS_DEAD_LETTER_STREAM", "personality:updates:dlq")
    REDIS_MAX_RETRIES = int(os.getenv("REDIS_MAX_RETRIES", "3"))
    REDIS_BATCH_SIZE = int(os.getenv("REDIS_BATCH_SIZE", "10"))
    REDIS_BLOCK_MS = int(os.getenv("REDIS_BLOCK_MS", "1000"))
    
    # Worker Configuration
    EVENT_WORKER_THREADS = int(os.getenv("EVENT_WORKER_THREADS", "2"))
    EVENT_PROCESSING_TIMEOUT = int(os.getenv("EVENT_PROCESSING_TIMEOUT", "30"))
    
    # Graceful Shutdown
    SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))  # seconds
    DRAIN_TIMEOUT = int(os.getenv("DRAIN_TIMEOUT", "60"))  # seconds
    
    # Performance
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    RETRY_BACKOFF_MS = int(os.getenv("RETRY_BACKOFF_MS", "10"))


# ============================================================
# PROMETHEUS METRICS (Enhanced)
# ============================================================

class PrometheusMetrics:
    """Low-cardinality metrics with additional observability"""
    
    REQUEST_COUNTER = Counter(
        "personality_requests_total",
        "Total personality requests",
        ["status"]
    )
    
    RESPONSE_LATENCY = Histogram(
        "personality_response_latency_seconds",
        "Response generation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    )
    
    EMBEDDING_LATENCY = Histogram(
        "personality_embedding_latency_seconds",
        "Embedding generation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    )
    
    DRIFT_SUMMARY = Summary(
        "personality_drift_summary",
        "Personality drift statistics"
    )
    
    # NEW: Playfulness metrics
    PLAYFULNESS_DISTRIBUTION = Histogram(
        "personality_playfulness_distribution",
        "Distribution of playfulness levels across users",
        buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    
    CHEEKY_FREQUENCY_DISTRIBUTION = Histogram(
        "personality_cheeky_frequency_distribution",
        "Distribution of cheeky response frequencies",
        buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20]
    )
    
    PLAYFULNESS_SUPPRESSED = Counter(
        "personality_playfulness_suppressed_total",
        "Times playfulness was suppressed due to context",
        ["reason"]
    )
    
    CHEEKY_OVERUSE_EVENTS = Counter(
        "personality_cheeky_overuse_total",
        "Times cheeky frequency limit was hit"
    )
    
    STORAGE_OPERATIONS = Counter(
        "personality_storage_operations_total",
        "Storage operations",
        ["operation", "backend", "status"]
    )
    
    PROFILE_UPDATES = Counter(
        "personality_profile_updates_total",
        "Profile update operations",
        ["type", "status"]
    )
    
    VERSION_CONFLICTS = Counter(
        "personality_version_conflicts_total",
        "Optimistic lock conflicts"
    )
    
    SIMILARITY_SCORES = Histogram(
        "personality_similarity_scores",
        "Cosine similarity scores",
        buckets=[0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    )
    
    # Enhanced event stream metrics
    EVENT_STREAM_PENDING = Gauge(
        "personality_event_stream_pending",
        "Pending events in consumer group (XPENDING)",
        ["consumer_group"]
    )
    
    EVENT_STREAM_CONSUMERS = Gauge(
        "personality_event_stream_consumers",
        "Active consumers in group",
        ["consumer_group"]
    )
    
    EVENT_PROCESSING_TIME = Histogram(
        "personality_event_processing_seconds",
        "Time to process event from stream",
        buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    )
    
    DEAD_LETTER_COUNTER = Counter(
        "personality_dead_letter_total",
        "Events sent to dead letter queue"
    )
    
    CACHE_OPERATIONS = Counter(
        "personality_cache_operations_total",
        "Cache operations",
        ["operation", "result"]
    )
    
    CACHE_INVALIDATIONS = Counter(
        "personality_cache_invalidations_total",
        "Cache invalidation events",
        ["source"]
    )
    
    # Time-based decay metrics
    DRIFT_DECAY_OPERATIONS = Counter(
        "personality_drift_decay_total",
        "Drift decay operations"
    )
    
    TIME_SINCE_LAST_UPDATE = Histogram(
        "personality_time_since_update_seconds",
        "Time elapsed since last profile update",
        buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400]
    )
    
    # Shutdown metrics
    SHUTDOWN_DRAIN_TIME = Histogram(
        "personality_shutdown_drain_seconds",
        "Time spent draining events during shutdown",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
    )
    
    INFLIGHT_EVENTS = Gauge(
        "personality_inflight_events",
        "Number of events currently being processed"
    )
    
    # System info
    SYSTEM_INFO = Info(
        "personality_system",
        "System information"
    )


# ============================================================
# PRE-NORMALIZED EMBEDDINGS (same as before)
# ============================================================

class EmbeddingSingleton:
    """Thread-safe singleton with pre-normalized embeddings"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._load_model()
                    self._initialized = True
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading embedding model", model=Config.EMBEDDING_MODEL_NAME)
            start_time = time.time()
            
            self.model = SentenceTransformer(
                Config.EMBEDDING_MODEL_NAME,
                device=Config.EMBEDDING_DEVICE
            )
            
            self.model.encode = self._wrap_encode_with_normalization(self.model.encode)
            
            load_time = time.time() - start_time
            logger.info("Embedding model loaded", load_time_seconds=load_time)
            
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    def _wrap_encode_with_normalization(self, original_encode):
        def normalized_encode(texts, **kwargs):
            embeddings = original_encode(texts, **kwargs)
            if isinstance(embeddings, np.ndarray):
                norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms
            return embeddings
        return normalized_encode
    
    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)


class EmbeddingService:
    """Service for pre-normalized embeddings"""
    
    def __init__(self):
        self.model = EmbeddingSingleton()
        self._prototype_cache = {}
        self._cache_lock = threading.RLock()
    
    def encode(self, text: str) -> np.ndarray:
        start = time.time()
        try:
            vec = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
            PrometheusMetrics.EMBEDDING_LATENCY.observe(time.time() - start)
            return vec
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2))
    
    def get_or_create_prototype(self, category: str, text: str) -> np.ndarray:
        with self._cache_lock:
            if category in self._prototype_cache:
                return self._prototype_cache[category]
            prototype = self.encode(text)
            self._prototype_cache[category] = prototype
            return prototype
    
    def match_against_prototypes(
        self,
        query_embedding: np.ndarray,
        prototypes: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        results = {}
        for category, prototype in prototypes.items():
            similarity = float(np.dot(query_embedding, prototype))
            results[category] = similarity
            PrometheusMetrics.SIMILARITY_SCORES.observe(similarity)
        return results


# ============================================================
# SEMANTIC PROTOTYPES
# ============================================================

class SemanticPrototypes:
    PROTOTYPES = {
        # Original prototypes
        "gratitude": "Thank you so much! I really appreciate your help.",
        "curiosity": "That's fascinating! Can you tell me more?",
        "confusion": "I'm not sure I understand. Could you explain?",
        "technical": "What's the time complexity? How does it handle edge cases?",
        "casual": "Hey! What's up?",
        "formal": "I would like to inquire about the specifications.",
        "detailed_request": "Could you provide a comprehensive explanation with examples?",
        "brief_request": "Quick question - yes or no?",
        
        # Playfulness context detection
        "distress": "I'm really stressed and need help urgently. This is serious.",
        "serious_technical": "This is a critical production issue that needs immediate attention.",
        "grief_sadness": "I'm feeling really down and upset about this situation.",
        "emergency": "Help! This is urgent and critical, I need assistance now!",
        "humor_appreciated": "Haha that's hilarious! Love the humor, made my day!",
        "positive_feedback": "That's funny! I appreciate the wit and playfulness.",
    }
    
    @classmethod
    def get_all(cls, embedding_service: EmbeddingService) -> Dict[str, np.ndarray]:
        prototypes = {}
        for category, text in cls.PROTOTYPES.items():
            prototypes[category] = embedding_service.get_or_create_prototype(category, text)
        return prototypes


# ============================================================
# DISTRIBUTED CACHE WITH INVALIDATION
# ============================================================

class DistributedCache:
    """Distributed cache with pub/sub invalidation"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.local_cache = {}
        self.cache_lock = threading.RLock()
        self.redis_client = redis_client
        self.invalidation_subscriber = None
        self.subscriber_thread = None
        self.running = False
        
        if Config.CACHE_INVALIDATION_ENABLED and redis_client:
            self._start_invalidation_listener()
    
    def _start_invalidation_listener(self):
        self.running = True
        self.invalidation_subscriber = self.redis_client.pubsub()
        self.invalidation_subscriber.subscribe(Config.CACHE_INVALIDATION_CHANNEL)
        
        self.subscriber_thread = threading.Thread(
            target=self._invalidation_loop,
            name="CacheInvalidationListener",
            daemon=True
        )
        self.subscriber_thread.start()
        
        logger.info("Cache invalidation listener started", 
                   channel=Config.CACHE_INVALIDATION_CHANNEL)
    
    def _invalidation_loop(self):
        for message in self.invalidation_subscriber.listen():
            if not self.running:
                break
            
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    user_id = data['user_id']
                    source_instance = data.get('instance_id')
                    
                    if source_instance != Config.INSTANCE_ID:
                        with self.cache_lock:
                            if user_id in self.local_cache:
                                del self.local_cache[user_id]
                                logger.debug("Cache invalidated",
                                           user_id=user_id,
                                           source=source_instance)
                                PrometheusMetrics.CACHE_INVALIDATIONS.labels(source="remote").inc()
                except Exception as e:
                    logger.error("Invalidation message processing failed", error=str(e))
    
    def get(self, user_id: str, bypass: bool = False) -> Optional[Any]:
        """Get from cache with optional bypass"""
        if bypass:
            return None
        
        with self.cache_lock:
            if user_id in self.local_cache:
                entry = self.local_cache[user_id]
                if time.time() - entry['timestamp'] < Config.CACHE_TTL_SECONDS:
                    PrometheusMetrics.CACHE_OPERATIONS.labels(
                        operation="get", result="hit"
                    ).inc()
                    return entry['data']
                else:
                    del self.local_cache[user_id]
            
            PrometheusMetrics.CACHE_OPERATIONS.labels(
                operation="get", result="miss"
            ).inc()
            return None
    
    def set(self, user_id: str, data: Any):
        with self.cache_lock:
            self.local_cache[user_id] = {
                'data': data,
                'timestamp': time.time()
            }
            PrometheusMetrics.CACHE_OPERATIONS.labels(
                operation="set", result="success"
            ).inc()
    
    def invalidate(self, user_id: str, broadcast: bool = True):
        with self.cache_lock:
            if user_id in self.local_cache:
                del self.local_cache[user_id]
                PrometheusMetrics.CACHE_INVALIDATIONS.labels(source="local").inc()
        
        if broadcast and self.redis_client and Config.CACHE_INVALIDATION_ENABLED:
            try:
                message = json.dumps({
                    'user_id': user_id,
                    'instance_id': Config.INSTANCE_ID,
                    'timestamp': time.time()
                })
                self.redis_client.publish(Config.CACHE_INVALIDATION_CHANNEL, message)
                PrometheusMetrics.CACHE_OPERATIONS.labels(
                    operation="invalidate", result="broadcast"
                ).inc()
            except Exception as e:
                logger.error("Cache invalidation broadcast failed", error=str(e))
    
    def shutdown(self):
        self.running = False
        if self.invalidation_subscriber:
            self.invalidation_subscriber.unsubscribe()
            self.invalidation_subscriber.close()
        if self.subscriber_thread:
            self.subscriber_thread.join(timeout=5)


# ============================================================
# VERSIONED PROFILE WITH TIME-BASED DECAY
# ============================================================

@dataclass
class PersonalityProfile:
    """Personality profile with time-based decay and playfulness"""
    warmth: float
    verbosity: float
    playfulness: float  # NEW: Playfulness/humor level (0.0-1.0)
    baseline_warmth: float
    baseline_verbosity: float
    baseline_playfulness: float  # NEW: Baseline playfulness
    interaction_count: int = 0
    version: int = 1
    last_updated: str = None
    last_updated_timestamp: float = 0.0  # Unix timestamp for time-based decay
    
    # NEW: Cheeky frequency tracking (anti-overuse)
    cheeky_responses_last_hour: int = 0
    last_cheeky_timestamp: float = 0.0
    cheeky_history: List[float] = None  # Timestamps of cheeky responses
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc).isoformat()
            self.last_updated_timestamp = time.time()
        if self.cheeky_history is None:
            self.cheeky_history = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityProfile':
        return cls(**data)
    
    @classmethod
    def create_default(cls) -> 'PersonalityProfile':
        return cls(
            warmth=0.7,
            verbosity=0.5,
            playfulness=0.5,  # NEW: Moderate playfulness by default
            baseline_warmth=0.7,
            baseline_verbosity=0.5,
            baseline_playfulness=0.5,  # NEW
            interaction_count=0,
            version=1,
            last_updated_timestamp=time.time(),
            cheeky_responses_last_hour=0,
            last_cheeky_timestamp=0.0,
            cheeky_history=[]
        )
    
    def apply_time_based_decay(self) -> bool:
        """
        Apply time-based decay toward baseline for all traits.
        Returns True if decay was applied, False otherwise.
        """
        if not Config.DRIFT_DECAY_ENABLED:
            return False
        
        # Calculate time elapsed since last update
        current_time = time.time()
        elapsed_seconds = current_time - self.last_updated_timestamp
        
        # Track time since last update
        PrometheusMetrics.TIME_SINCE_LAST_UPDATE.observe(elapsed_seconds)
        
        # Only decay if enough time has passed
        if elapsed_seconds < Config.DRIFT_DECAY_INTERVAL:
            return False
        
        # Calculate decay amount based on time elapsed
        # Clamp decay_factor to prevent overshoot (max 1.0 = full return to baseline)
        decay_factor = min(Config.DRIFT_DECAY_RATE * elapsed_seconds, 1.0)
        
        # Decay warmth toward baseline
        warmth_diff = self.warmth - self.baseline_warmth
        if abs(warmth_diff) > 0.001:
            decay = warmth_diff * decay_factor
            self.warmth -= decay
            # Ensure we don't overshoot
            if abs(self.warmth - self.baseline_warmth) < 0.001:
                self.warmth = self.baseline_warmth
        
        # Decay verbosity toward baseline
        verbosity_diff = self.verbosity - self.baseline_verbosity
        if abs(verbosity_diff) > 0.001:
            decay = verbosity_diff * decay_factor
            self.verbosity -= decay
            if abs(self.verbosity - self.baseline_verbosity) < 0.001:
                self.verbosity = self.baseline_verbosity
        
        # NEW: Decay playfulness toward baseline
        playfulness_diff = self.playfulness - self.baseline_playfulness
        if abs(playfulness_diff) > 0.001:
            decay = playfulness_diff * decay_factor
            self.playfulness -= decay
            if abs(self.playfulness - self.baseline_playfulness) < 0.001:
                self.playfulness = self.baseline_playfulness
        
        PrometheusMetrics.DRIFT_DECAY_OPERATIONS.inc()
        
        logger.debug("Time-based decay applied",
                    elapsed_seconds=elapsed_seconds,
                    decay_factor=decay_factor,
                    warmth=self.warmth,
                    verbosity=self.verbosity,
                    playfulness=self.playfulness)
        
        return True
    
    def _prune_cheeky_history(self):
        """
        Prune old timestamps and enforce max size to prevent unbounded growth.
        Should be called on every access to cheeky_history.
        """
        if not self.cheeky_history:
            return
        
        current_time = time.time()
        window_start = current_time - Config.CHEEKY_FREQUENCY_WINDOW
        
        # Prune old timestamps outside window
        self.cheeky_history = [ts for ts in self.cheeky_history if ts > window_start]
        
        # Enforce max size (keep most recent)
        if len(self.cheeky_history) > Config.MAX_CHEEKY_HISTORY_SIZE:
            self.cheeky_history = self.cheeky_history[-Config.MAX_CHEEKY_HISTORY_SIZE:]
    
    def update_cheeky_frequency(self) -> int:
        """
        Update cheeky response tracking and return current count in window.
        Prunes old timestamps outside the frequency window.
        
        Returns:
            int: Number of cheeky responses in last hour
        """
        # Always prune before updating
        self._prune_cheeky_history()
        
        # Add current timestamp
        current_time = time.time()
        self.cheeky_history.append(current_time)
        self.last_cheeky_timestamp = current_time
        
        # Update counter
        self.cheeky_responses_last_hour = len(self.cheeky_history)
        
        return self.cheeky_responses_last_hour
    
    def get_cheeky_count(self) -> int:
        """
        Get current cheeky response count without updating.
        Always prunes before counting to ensure accuracy.
        
        Returns:
            int: Number of cheeky responses in last hour
        """
        # Always prune before reading to get accurate count
        self._prune_cheeky_history()
        return len(self.cheeky_history)
    
    def is_overusing_cheeky(self) -> bool:
        """
        Check if cheeky responses are being overused.
        
        Returns:
            bool: True if frequency limit exceeded
        """
        return self.get_cheeky_count() >= Config.CHEEKY_FREQUENCY_LIMIT


# ============================================================
# STORAGE WITH REDIS CAS
# ============================================================

class ProfileStore(ABC):
    @abstractmethod
    def load(self, user_id: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def save(self, user_id: str, data: Dict, expected_version: int = None) -> bool:
        pass


class RedisStore(ProfileStore):
    """Redis storage with atomic CAS"""
    
    def __init__(self, redis_client: redis.Redis):
        self.client = redis_client
        self.prefix = "personality:profile:"
        
        self.cas_script = self.client.register_script("""
            local key = KEYS[1]
            local expected_version = tonumber(ARGV[1])
            local new_data = ARGV[2]
            local ttl = tonumber(ARGV[3])
            
            local current = redis.call('GET', key)
            
            if expected_version == -1 then
                redis.call('SETEX', key, ttl, new_data)
                return 1
            end
            
            if current == false then
                if expected_version == 0 then
                    redis.call('SETEX', key, ttl, new_data)
                    return 1
                else
                    return 0
                end
            else
                local current_obj = cjson.decode(current)
                if current_obj.version == expected_version then
                    redis.call('SETEX', key, ttl, new_data)
                    return 1
                else
                    return 0
                end
            end
        """)
    
    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"
    
    def load(self, user_id: str) -> Optional[Dict]:
        try:
            data = self.client.get(self._key(user_id))
            if data:
                result = json.loads(data)
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load", backend="redis", status="hit"
                ).inc()
                return result
            else:
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load", backend="redis", status="miss"
                ).inc()
                return None
        except Exception as e:
            logger.error("Redis load failed", error=str(e), user_id=user_id)
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="load", backend="redis", status="error"
            ).inc()
            return None
    
    def save(self, user_id: str, data: Dict, expected_version: int = None) -> bool:
        try:
            key = self._key(user_id)
            ttl = 86400 * 30
            
            if expected_version is None:
                expected_version = -1
            
            result = self.cas_script(
                keys=[key],
                args=[expected_version, json.dumps(data), ttl]
            )
            
            if result == 1:
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="save", backend="redis", status="success"
                ).inc()
                return True
            else:
                PrometheusMetrics.VERSION_CONFLICTS.inc()
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="save", backend="redis", status="conflict"
                ).inc()
                return False
        except Exception as e:
            logger.error("Redis save failed", error=str(e), user_id=user_id)
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save", backend="redis", status="error"
            ).inc()
            return False


class InMemoryStore(ProfileStore):
    """In-memory storage with thread safety"""
    def __init__(self):
        self.store = {}
        self.lock = threading.RLock()
    
    def load(self, user_id: str) -> Optional[Dict]:
        with self.lock:
            data = self.store.get(user_id)
            if data:
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load", backend="memory", status="hit"
                ).inc()
            else:
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load", backend="memory", status="miss"
                ).inc()
            return data
    
    def save(self, user_id: str, data: Dict, expected_version: int = None) -> bool:
        with self.lock:
            # Simple version checking for in-memory
            if expected_version is not None and expected_version > 0:
                current = self.store.get(user_id)
                if current and current.get('version') != expected_version:
                    PrometheusMetrics.VERSION_CONFLICTS.inc()
                    PrometheusMetrics.STORAGE_OPERATIONS.labels(
                        operation="save", backend="memory", status="conflict"
                    ).inc()
                    return False
            
            self.store[user_id] = data
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save", backend="memory", status="success"
            ).inc()
            return True


# ============================================================
# DURABLE EVENT STREAM WITH XPENDING MONITORING
# ============================================================

@dataclass
class ProfileUpdateEvent:
    event_id: str
    user_id: str
    delta_warmth: float
    delta_verbosity: float
    delta_playfulness: float = 0.0  # NEW
    timestamp: float = 0.0
    retry_count: int = 0
    correlation_id: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfileUpdateEvent':
        return cls(**data)
    
    @classmethod
    def create(
        cls,
        user_id: str,
        delta_warmth: float,
        delta_verbosity: float,
        delta_playfulness: float = 0.0,  # NEW
        correlation_id: str = ""
    ) -> 'ProfileUpdateEvent':
        return cls(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            delta_warmth=delta_warmth,
            delta_verbosity=delta_verbosity,
            delta_playfulness=delta_playfulness,  # NEW
            timestamp=time.time(),
            retry_count=0,
            correlation_id=correlation_id or str(uuid.uuid4())
        )


class RedisStreamsEventStream:
    """Redis Streams with XPENDING-based monitoring"""
    
    def __init__(self, redis_client: redis.Redis):
        self.client = redis_client
        self.stream_name = Config.REDIS_STREAM_NAME
        self.consumer_group = Config.REDIS_CONSUMER_GROUP
        self.dlq_stream = Config.REDIS_DEAD_LETTER_STREAM
        self.running = False
        self.shutdown_event = threading.Event()
        self.inflight_count = 0  # Track inflight events without reading Prometheus state
        self.inflight_lock = threading.Lock()
        
        # Create consumer group
        try:
            self.client.xgroup_create(
                self.stream_name,
                self.consumer_group,
                id='0',
                mkstream=True
            )
            logger.info("Consumer group created", group=self.consumer_group)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
    
    def produce(self, event: ProfileUpdateEvent) -> bool:
        try:
            with correlation_context(event.correlation_id):
                message_id = self.client.xadd(
                    self.stream_name,
                    {'data': json.dumps(event.to_dict())}
                )
                
                PrometheusMetrics.PROFILE_UPDATES.labels(
                    type="event_stream", status="produced"
                ).inc()
                
                logger.debug("Event produced",
                           event_id=event.event_id,
                           user_id=event.user_id,
                           message_id=message_id)
                
                # Update metrics
                self._update_pending_metrics()
                
                return True
        except Exception as e:
            logger.error("Event production failed", error=str(e), event_id=event.event_id)
            PrometheusMetrics.PROFILE_UPDATES.labels(
                type="event_stream", status="produce_error"
            ).inc()
            return False
    
    def _update_pending_metrics(self):
        """Update XPENDING-based metrics"""
        try:
            # Get pending count per consumer group
            pending_info = self.client.xpending(self.stream_name, self.consumer_group)
            
            if pending_info:
                pending_count = pending_info.get('pending', 0)
                PrometheusMetrics.EVENT_STREAM_PENDING.labels(
                    consumer_group=self.consumer_group
                ).set(pending_count)
            
            # Get consumer info
            groups = self.client.xinfo_groups(self.stream_name)
            for group in groups:
                if group['name'] == self.consumer_group:
                    consumer_count = group.get('consumers', 0)
                    PrometheusMetrics.EVENT_STREAM_CONSUMERS.labels(
                        consumer_group=self.consumer_group
                    ).set(consumer_count)
                    break
        except Exception as e:
            logger.error("Failed to update pending metrics", error=str(e))
    
    def consume(self, consumer_name: str, callback):
        """Consume with graceful shutdown support"""
        self.running = True
        
        logger.info("Consumer started",
                   consumer=consumer_name,
                   group=self.consumer_group)
        
        while self.running and not self.shutdown_event.is_set():
            try:
                messages = self.client.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_name: '>'},
                    count=Config.REDIS_BATCH_SIZE,
                    block=Config.REDIS_BLOCK_MS
                )
                
                if not messages:
                    self._update_pending_metrics()
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        if self.shutdown_event.is_set():
                            logger.info("Shutdown signaled, stopping consumption")
                            return
                        
                        # Track inflight locally AND in Prometheus
                        with self.inflight_lock:
                            self.inflight_count += 1
                        PrometheusMetrics.INFLIGHT_EVENTS.inc()
                        start_time = time.time()
                        
                        try:
                            event_data = json.loads(message_data[b'data'])
                            event = ProfileUpdateEvent.from_dict(event_data)
                            
                            with correlation_context(event.correlation_id):
                                success = callback(event)
                            
                            if success:
                                self.client.xack(self.stream_name, self.consumer_group, message_id)
                                PrometheusMetrics.PROFILE_UPDATES.labels(
                                    type="event_stream", status="success"
                                ).inc()
                            else:
                                event.retry_count += 1
                                if event.retry_count >= Config.REDIS_MAX_RETRIES:
                                    self.send_to_dlq(event, "Max retries exceeded")
                                    self.client.xack(self.stream_name, self.consumer_group, message_id)
                                else:
                                    PrometheusMetrics.PROFILE_UPDATES.labels(
                                        type="event_stream", status="retry"
                                    ).inc()
                            
                            processing_time = time.time() - start_time
                            PrometheusMetrics.EVENT_PROCESSING_TIME.observe(processing_time)
                            
                        except Exception as e:
                            logger.error("Event processing error",
                                       error=str(e),
                                       message_id=message_id)
                            PrometheusMetrics.PROFILE_UPDATES.labels(
                                type="event_stream", status="error"
                            ).inc()
                        finally:
                            PrometheusMetrics.INFLIGHT_EVENTS.dec()
                            with self.inflight_lock:
                                self.inflight_count -= 1
                
                self._update_pending_metrics()
                
            except Exception as e:
                logger.error("Consumer error", error=str(e))
                time.sleep(1)
    
    def send_to_dlq(self, event: ProfileUpdateEvent, error: str):
        try:
            dlq_data = event.to_dict()
            dlq_data['error'] = error
            dlq_data['failed_at'] = time.time()
            
            self.client.xadd(self.dlq_stream, {'data': json.dumps(dlq_data)})
            PrometheusMetrics.DEAD_LETTER_COUNTER.inc()
            
            logger.warning("Event sent to DLQ",
                          event_id=event.event_id,
                          error=error)
        except Exception as e:
            logger.error("DLQ send failed", error=str(e))
    
    def shutdown(self):
        """Signal shutdown"""
        self.running = False
        self.shutdown_event.set()
    
    def drain(self, timeout: int = 60) -> int:
        """
        Drain in-flight events before shutdown.
        Returns number of events drained.
        """
        logger.info("Draining in-flight events", timeout=timeout)
        start_time = time.time()
        drained = 0
        
        while time.time() - start_time < timeout:
            # Check local inflight counter (don't read Prometheus internal state)
            with self.inflight_lock:
                inflight = self.inflight_count
            
            if inflight == 0:
                # Check XPENDING for any pending events
                try:
                    pending_info = self.client.xpending(self.stream_name, self.consumer_group)
                    pending_count = pending_info.get('pending', 0) if pending_info else 0
                    
                    if pending_count == 0:
                        logger.info("All events drained", drained=drained)
                        return drained
                except:
                    pass
            
            time.sleep(0.1)
            drained += 1
        
        drain_time = time.time() - start_time
        PrometheusMetrics.SHUTDOWN_DRAIN_TIME.observe(drain_time)
        
        logger.warning("Drain timeout reached",
                      timeout=timeout,
                      drain_time=drain_time)
        return drained


# ============================================================
# PROFILE MANAGER WITH CACHE BYPASS
# ============================================================

class ProfileManager:
    """Profile manager with cache bypass during event processing"""
    
    def __init__(
        self,
        store: ProfileStore,
        cache,  # Can be DistributedCache or LocalCache
        event_stream  # Can be RedisStreamsEventStream or InMemoryEventStream
    ):
        self.store = store
        self.cache = cache
        self.event_stream = event_stream
        self.lock = threading.RLock()
        self.consumers = []
        
        for i in range(Config.EVENT_WORKER_THREADS):
            consumer_name = f"{Config.INSTANCE_ID}-worker-{i}"
            worker = threading.Thread(
                target=self.event_stream.consume,
                args=(consumer_name, self._process_event),
                name=f"EventConsumer-{i}",
                daemon=True
            )
            worker.start()
            self.consumers.append(worker)
        
        logger.info("Event consumers started", count=Config.EVENT_WORKER_THREADS)
    
    def get_profile(self, user_id: str, bypass_cache: bool = False, apply_decay: bool = True) -> PersonalityProfile:
        """
        Get profile with optional cache bypass and time-based decay.
        
        Args:
            user_id: User identifier
            bypass_cache: Skip cache and read from storage
            apply_decay: Apply time-based decay on read (truly time-based)
        
        Returns:
            PersonalityProfile with optional decay applied
        """
        if not bypass_cache:
            cached = self.cache.get(user_id, bypass=False)
            if cached:
                profile = PersonalityProfile.from_dict(cached)
                # Apply decay even for cached profiles if requested
                if apply_decay:
                    decay_applied = profile.apply_time_based_decay()
                    if decay_applied:
                        # Update cache with decayed values
                        self.cache.set(user_id, profile.to_dict())
                return profile
        
        with self.lock:
            data = self.store.load(user_id)
            
            if data:
                profile = PersonalityProfile.from_dict(data)
            else:
                profile = PersonalityProfile.create_default()
                self.store.save(user_id, profile.to_dict())
            
            # Apply time-based decay on read (optional, makes decay truly time-based)
            if apply_decay:
                decay_applied = profile.apply_time_based_decay()
                if decay_applied:
                    # Persist decayed values back to storage
                    self.store.save(user_id, profile.to_dict())
                    logger.debug("Decay applied on profile read",
                               user_id=user_id,
                               warmth=profile.warmth,
                               verbosity=profile.verbosity,
                               playfulness=profile.playfulness)
            
            if not bypass_cache:
                self.cache.set(user_id, profile.to_dict())
            
            return profile
    
    def update_profile_async(
        self,
        user_id: str,
        delta_warmth: float,
        delta_verbosity: float,
        delta_playfulness: float = 0.0,  # NEW
        correlation_id: str = ""
    ):
        """Enqueue async update"""
        event = ProfileUpdateEvent.create(
            user_id, delta_warmth, delta_verbosity, delta_playfulness, correlation_id
        )
        self.event_stream.produce(event)
    
    def _process_event(self, event: ProfileUpdateEvent) -> bool:
        """
        Process event with cache bypass to avoid stale reads.
        This is the critical fix for race conditions.
        """
        try:
            for attempt in range(Config.MAX_RETRY_ATTEMPTS):
                # BYPASS CACHE - always read fresh from storage
                profile = self.get_profile(event.user_id, bypass_cache=True)
                current_version = profile.version
                
                # Apply time-based decay first
                decay_applied = profile.apply_time_based_decay()
                
                # Apply updates
                profile.warmth = np.clip(
                    profile.warmth + event.delta_warmth,
                    0.2, 0.95
                )
                profile.verbosity = np.clip(
                    profile.verbosity + event.delta_verbosity,
                    0.15, 0.9
                )
                
                # NEW: Apply playfulness updates
                if Config.PLAYFULNESS_ENABLED and event.delta_playfulness != 0:
                    profile.playfulness = np.clip(
                        profile.playfulness + event.delta_playfulness,
                        0.0, 1.0
                    )
                    
                    # If playfulness increased (cheeky response used), track it
                    if event.delta_playfulness > 0 and profile.playfulness > 0.5:
                        cheeky_count = profile.update_cheeky_frequency()
                        logger.debug("Cheeky response tracked",
                                   user_id=event.user_id,
                                   count=cheeky_count,
                                   playfulness=profile.playfulness)
                
                profile.interaction_count += 1
                profile.last_updated = datetime.now(timezone.utc).isoformat()
                profile.last_updated_timestamp = time.time()
                profile.version += 1
                
                drift = abs(profile.warmth - profile.baseline_warmth)
                PrometheusMetrics.DRIFT_SUMMARY.observe(drift)
                
                # NEW: Update playfulness metrics
                if Config.PLAYFULNESS_ENABLED:
                    PrometheusMetrics.PLAYFULNESS_DISTRIBUTION.observe(profile.playfulness)
                    PrometheusMetrics.CHEEKY_FREQUENCY_DISTRIBUTION.observe(profile.get_cheeky_count())
                
                # Atomic save
                success = self.store.save(
                    event.user_id,
                    profile.to_dict(),
                    expected_version=current_version
                )
                
                if success:
                    # Invalidate cache cluster-wide
                    self.cache.invalidate(event.user_id, broadcast=True)
                    
                    logger.debug("Profile updated",
                               user_id=event.user_id,
                               warmth=profile.warmth,
                               verbosity=profile.verbosity,
                               playfulness=profile.playfulness,
                               cheeky_count=profile.get_cheeky_count(),
                               decay_applied=decay_applied)
                    
                    return True
                else:
                    logger.debug("Version conflict, retrying",
                               attempt=attempt,
                               user_id=event.user_id)
                    time.sleep(Config.RETRY_BACKOFF_MS / 1000.0 * (attempt + 1))
            
            return False
        except Exception as e:
            logger.error("Event processing failed",
                        error=str(e),
                        event_id=event.event_id)
            return False
    
    def shutdown(self, drain_timeout: int = None):
        """Graceful shutdown with event draining"""
        drain_timeout = drain_timeout or Config.DRAIN_TIMEOUT
        
        logger.info("Initiating graceful shutdown", drain_timeout=drain_timeout)
        
        # Signal consumers to stop accepting new work
        self.event_stream.shutdown()
        
        # Drain in-flight events
        drained = self.event_stream.drain(timeout=drain_timeout)
        
        logger.info("Shutdown complete", events_drained=drained)

class LocalCache:
    """Simple in-memory cache for non-Redis mode"""
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()
    
    def get(self, user_id: str, bypass: bool = False) -> Optional[Dict]:
        """Get from cache with optional bypass (compatible with DistributedCache)"""
        if bypass:
            return None
        
        with self.lock:
            entry = self.cache.get(user_id)
            if entry and entry['expires'] > time.time():
                PrometheusMetrics.CACHE_OPERATIONS.labels(
                    operation="get", result="hit"
                ).inc()
                return entry['data']
            PrometheusMetrics.CACHE_OPERATIONS.labels(
                operation="get", result="miss"
            ).inc()
            return None
    
    def set(self, user_id: str, data: Dict):
        """Set cache entry (compatible with DistributedCache)"""
        with self.lock:
            self.cache[user_id] = {
                'data': data,
                'expires': time.time() + Config.CACHE_TTL_SECONDS
            }
            PrometheusMetrics.CACHE_OPERATIONS.labels(
                operation="set", result="success"
            ).inc()
    
    def invalidate(self, user_id: str, broadcast: bool = False):
        """Invalidate cache entry (compatible with DistributedCache)"""
        with self.lock:
            if user_id in self.cache:
                del self.cache[user_id]
                PrometheusMetrics.CACHE_INVALIDATIONS.labels(source="local").inc()
    
    def shutdown(self):
        pass


class InMemoryEventStream:
    """Simple in-memory event stream for non-Redis mode"""
    def __init__(self):
        self.queue = []
        self.lock = threading.RLock()
        self.workers = []
        self.running = False
        self.inflight_count = 0  # Track inflight events
        self.inflight_lock = threading.Lock()
    
    def produce(self, event: ProfileUpdateEvent):
        """Produce event (compatible with RedisStreamsEventStream interface)"""
        with self.lock:
            self.queue.append(event)
        return True
    
    def consume(self, consumer_name: str, handler):
        """Start a consumer thread (compatible with RedisStreamsEventStream interface)"""
        self.running = True
        while self.running:
            event = None
            with self.lock:
                if self.queue:
                    event = self.queue.pop(0)
            
            if event:
                # Track inflight
                with self.inflight_lock:
                    self.inflight_count += 1
                PrometheusMetrics.INFLIGHT_EVENTS.inc()
                
                try:
                    handler(event)
                except Exception as e:
                    logger.error("Event processing failed", error=str(e))
                finally:
                    # Decrement inflight
                    PrometheusMetrics.INFLIGHT_EVENTS.dec()
                    with self.inflight_lock:
                        self.inflight_count -= 1
            else:
                time.sleep(0.1)
    
    def shutdown(self):
        self.running = False
    
    def drain(self, timeout: int = 30) -> int:
        """Drain events, waiting for inflight processing to complete"""
        drained = 0
        start = time.time()
        
        # Wait for inflight events to complete
        while (time.time() - start) < timeout:
            with self.inflight_lock:
                inflight = self.inflight_count
            
            with self.lock:
                queue_size = len(self.queue)
            
            # All done when no inflight and no queue
            if inflight == 0 and queue_size == 0:
                logger.info("All events drained", drained=drained)
                return drained
            
            # Process remaining queue items
            if queue_size > 0:
                with self.lock:
                    if self.queue:
                        drained += 1
                        self.queue.pop(0)
            
            time.sleep(0.01)
        
        return drained
# ============================================================
# MAIN SYSTEM WITH GRACEFUL SHUTDOWN
# ============================================================

class EnhancedPersonalityV6Final:
    """Final production-ready personality system"""
    
    def __init__(self):
        logger.info("Initializing Final Production Personality System",
                   instance_id=Config.INSTANCE_ID)
        
        # Set system info
        PrometheusMetrics.SYSTEM_INFO.info({
            'instance_id': Config.INSTANCE_ID,
            'version': '6.0-final',
            'storage_backend': Config.STORAGE_BACKEND,
            'event_stream': Config.EVENT_STREAM_BACKEND
        })
        
        self.embedding = EmbeddingService()
        self.prototypes = SemanticPrototypes.get_all(self.embedding)
        
        if Config.STORAGE_BACKEND == "redis":
            if not REDIS_AVAILABLE:
                raise RuntimeError("Redis backend selected but redis package not installed")
            
            self.redis_client = redis.Redis.from_url(
                Config.REDIS_URL,
                decode_responses=True
            )
            store = RedisStore(self.redis_client)
            cache = DistributedCache(self.redis_client)
            event_stream = RedisStreamsEventStream(self.redis_client)
        else:
            # Memory mode fallback
            logger.info("Using in-memory storage (no Redis)")
            store = InMemoryStore()
            cache = LocalCache()
            event_stream = InMemoryEventStream()
        
        self.profile_manager = ProfileManager(store, cache, event_stream)
        self.cache = cache
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("System initialized",
                   prototypes=len(self.prototypes),
                   workers=Config.EVENT_WORKER_THREADS)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received", signal=signum)
        self.shutdown()
        sys.exit(0)
    
    def build_system_prompt(self, user_id: str) -> str:
        profile = self.profile_manager.get_profile(user_id)
        
        if profile.warmth < 0.3:
            warmth_instruction = "Be professional and concise."
        elif profile.warmth < 0.6:
            warmth_instruction = "Be friendly but professional."
        else:
            warmth_instruction = "Be warm and personable."
        
        if profile.verbosity < 0.3:
            verbosity_instruction = "Be extremely concise (1-2 sentences)."
        elif profile.verbosity < 0.5:
            verbosity_instruction = "Be moderately brief (2-4 sentences)."
        elif profile.verbosity < 0.7:
            verbosity_instruction = "Provide moderate detail (3-6 sentences)."
        else:
            verbosity_instruction = "Provide comprehensive explanations."
        
        # NEW: Playfulness conditioning (affects tone, rhythm, phrasing)
        if not Config.PLAYFULNESS_ENABLED or profile.playfulness < 0.2:
            playfulness_instruction = "Maintain a straightforward, professional tone with clear, direct language."
        elif profile.playfulness < 0.4:
            playfulness_instruction = "You may use occasional light humor when contextually appropriate, but stay professional."
        elif profile.playfulness < 0.6:
            playfulness_instruction = "Use friendly, conversational language with wit when fitting. Balance professionalism with approachability."
        elif profile.playfulness < 0.8:
            playfulness_instruction = "Be playful and use humor liberally. Let your personality shine through with witty asides, wordplay, and cheeky remarks while staying helpful."
        else:
            playfulness_instruction = "Embrace maximum playfulness! Use clever wordplay, witty observations, and playful language throughout. Be cheeky and fun while still being genuinely helpful. Read the room—dial it back if the topic is serious."
        
        # Check if playfulness is being suppressed due to overuse
        cheeky_count = profile.get_cheeky_count()
        if cheeky_count >= Config.CHEEKY_FREQUENCY_LIMIT:
            playfulness_instruction = "OVERRIDE: You've been very playful recently. Dial back the humor for now and focus on being helpful and clear."
            PrometheusMetrics.CHEEKY_OVERUSE_EVENTS.inc()
        
        return f"""You are Project Elixer — a companion, not an assistant. Talk like a genuine, sharp friend: direct, warm, never stiff or corporate.

Personality Profile: Warmth={profile.warmth:.2f}, Verbosity={profile.verbosity:.2f}, Playfulness={profile.playfulness:.2f}

Style Guidelines:
{warmth_instruction}
{verbosity_instruction}
{playfulness_instruction}

Important: Playfulness should affect your natural tone, rhythm, and word choice—not just add jokes at the end. Let it flow through your entire response."""
    
    def process_response(
        self,
        user_id: str,
        user_input: str,
        base_response: str,
        correlation_id: str = None
    ) -> Tuple[str, Dict]:
        """Process with correlation ID and playfulness gating"""
        correlation_id = correlation_id or str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            start = time.time()
            
            try:
                profile = self.profile_manager.get_profile(user_id)
                query_embedding = self.embedding.encode(user_input)
                similarities = self.embedding.match_against_prototypes(
                    query_embedding, self.prototypes
                )
                
                delta_warmth = 0.0
                delta_verbosity = 0.0
                delta_playfulness = 0.0
                
                # Original adjustments
                if similarities.get("gratitude", 0) > Config.SIMILARITY_THRESHOLD:
                    delta_warmth = Config.WARMTH_ADJUSTMENT_RATE
                if similarities.get("detailed_request", 0) > Config.SIMILARITY_THRESHOLD:
                    delta_verbosity = Config.VERBOSITY_ADJUSTMENT_RATE
                if similarities.get("brief_request", 0) > Config.SIMILARITY_THRESHOLD:
                    delta_verbosity = -Config.VERBOSITY_ADJUSTMENT_RATE
                
                # NEW: Playfulness adjustment with semantic gating
                if Config.PLAYFULNESS_ENABLED:
                    # Check for serious/distress contexts (suppress playfulness)
                    serious_threshold = Config.SERIOUS_CONTEXT_THRESHOLD
                    playfulness_suppressed = False
                    suppress_reason = None
                    
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
                        # Don't adjust playfulness, but log suppression
                        PrometheusMetrics.PLAYFULNESS_SUPPRESSED.labels(reason=suppress_reason).inc()
                        logger.debug("Playfulness suppressed",
                                   user_id=user_id,
                                   reason=suppress_reason,
                                   similarity=similarities.get(suppress_reason, 0))
                    else:
                        # Check for positive feedback on humor
                        if similarities.get("humor_appreciated", 0) > Config.SIMILARITY_THRESHOLD:
                            delta_playfulness = Config.PLAYFULNESS_ADJUSTMENT_RATE
                        elif similarities.get("positive_feedback", 0) > Config.SIMILARITY_THRESHOLD:
                            delta_playfulness = Config.PLAYFULNESS_ADJUSTMENT_RATE * 0.5
                        
                        # Check for overuse (reduce playfulness if too frequent)
                        if profile.is_overusing_cheeky():
                            delta_playfulness = -Config.PLAYFULNESS_ADJUSTMENT_RATE * 2
                            PrometheusMetrics.CHEEKY_OVERUSE_EVENTS.inc()
                            logger.debug("Reducing playfulness due to overuse",
                                       user_id=user_id,
                                       cheeky_count=profile.get_cheeky_count())
                
                # Update profile if any deltas exist
                if delta_warmth != 0 or delta_verbosity != 0 or delta_playfulness != 0:
                    self.profile_manager.update_profile_async(
                        user_id, delta_warmth, delta_verbosity, delta_playfulness, correlation_id
                    )
                
                # Update metrics
                drift = abs(profile.warmth - profile.baseline_warmth)
                latency = time.time() - start
                
                PrometheusMetrics.REQUEST_COUNTER.labels(status="success").inc()
                PrometheusMetrics.RESPONSE_LATENCY.observe(latency)
                PrometheusMetrics.PLAYFULNESS_DISTRIBUTION.observe(profile.playfulness)
                PrometheusMetrics.CHEEKY_FREQUENCY_DISTRIBUTION.observe(profile.get_cheeky_count())
                
                metadata = {
                    "warmth": profile.warmth,
                    "verbosity": profile.verbosity,
                    "playfulness": profile.playfulness,  # NEW
                    "cheeky_count": profile.get_cheeky_count(),  # NEW
                    "drift": drift,
                    "top_matches": sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3],
                    "latency_ms": latency * 1000,
                    "correlation_id": correlation_id,
                    "instance_id": Config.INSTANCE_ID
                }
                
                logger.info("Request processed",
                           user_id=user_id,
                           latency_ms=latency*1000,
                           playfulness=profile.playfulness,
                           cheeky_count=profile.get_cheeky_count())
                
                return base_response, metadata
            except Exception as e:
                logger.error("Request failed", error=str(e), user_id=user_id)
                PrometheusMetrics.REQUEST_COUNTER.labels(status="error").inc()
                return base_response, {"error": str(e), "correlation_id": correlation_id}
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating system shutdown")
        self.profile_manager.shutdown(drain_timeout=Config.DRAIN_TIMEOUT)
        self.cache.shutdown()
        logger.info("System shutdown complete")


def main():
    system = EnhancedPersonalityV6Final()
    
    try:
        user_id = "user_123"
        
        for i, user_input in enumerate([
            "Thanks!",
            "Explain in detail?",
            "Quick question?"
        ]):
            print(f"\n{'='*60}")
            print(f"Request {i+1}: {user_input}")
            
            response, metadata = system.process_response(
                user_id, user_input, "Response"
            )
            
            print(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")
        
        time.sleep(3)
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()