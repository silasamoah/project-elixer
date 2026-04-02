"""
Enterprise Personality System v6 - Enhanced Edition
Production-Ready Infrastructure

Enhancements:
- Robust shared embedding singleton with error handling
- Comprehensive environment configuration
- Real moderation model integrations (OpenAI, Perspective API)
- Advanced verbosity conditioning with dynamic prompts
- Thread-safe profile updates with connection pooling
- Redis + PostgreSQL persistent storage implementations
- Dual observability: Prometheus + OpenTelemetry
- Comprehensive load testing with latency analysis
- Health checks and circuit breakers
- Structured logging
"""

import os
import time
import threading
import hashlib
import logging
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime
from contextlib import contextmanager
from enum import Enum
import numpy as np

# Observability
from prometheus_client import Counter, Histogram, Gauge, Summary
# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Storage backends (optional)
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

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (Environment Driven)
# ============================================================

class Config:
    """Centralized configuration from environment variables"""
    
    # Similarity & Tone Thresholds
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.28"))
    MAX_TONE_DRIFT = float(os.getenv("MAX_TONE_DRIFT", "0.4"))
    WARMTH_ADJUSTMENT_RATE = float(os.getenv("WARMTH_ADJUSTMENT_RATE", "0.01"))
    VERBOSITY_ADJUSTMENT_RATE = float(os.getenv("VERBOSITY_ADJUSTMENT_RATE", "0.02"))
    
    # Model Configuration
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    MODERATION_PROVIDER = os.getenv("MODERATION_PROVIDER", "openai")  # openai, perspective, internal
    MODERATION_API_KEY = os.getenv("MODERATION_API_KEY", "")
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, local
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Storage Configuration
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")  # memory, redis, postgres
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost/personality")
    POSTGRES_MIN_CONNECTIONS = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "2"))
    POSTGRES_MAX_CONNECTIONS = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10"))
    
    # Observability
    ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    ENABLE_OTEL = os.getenv("ENABLE_OTEL", "false").lower() == "true"
    OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "localhost:4317")
    
    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
    
    # Performance
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))


# ============================================================
# PROMETHEUS METRICS
# ============================================================

class PrometheusMetrics:
    """Prometheus metrics collection"""
    
    REQUEST_COUNTER = Counter(
        "personality_requests_total", 
        "Total personality requests",
        ["user_id", "status"]
    )
    
    EMBEDDING_LATENCY = Histogram(
        "embedding_latency_seconds", 
        "Embedding generation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    
    RESPONSE_LATENCY = Histogram(
        "response_latency_seconds", 
        "Full response generation latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    TONE_DRIFT_GAUGE = Gauge(
        "tone_drift", 
        "Tone drift from baseline",
        ["user_id"]
    )
    
    MODERATION_COUNTER = Counter(
        "moderation_checks_total",
        "Total moderation checks",
        ["result"]
    )
    
    STORAGE_OPERATIONS = Counter(
        "storage_operations_total",
        "Storage operations",
        ["operation", "backend", "status"]
    )
    
    CIRCUIT_BREAKER_STATE = Gauge(
        "circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open)",
        ["service"]
    )


# ============================================================
# OPENTELEMETRY INTEGRATION
# ============================================================

class OpenTelemetryProvider:
    """OpenTelemetry tracing and metrics"""
    
    def __init__(self):
        if not OTEL_AVAILABLE or not Config.ENABLE_OTEL:
            self.enabled = False
            self.tracer = None
            self.meter = None
            return
        
        self.enabled = True
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        trace.get_tracer_provider().add_span_processor(
            trace.export.SimpleSpanProcessor(
                OTLPSpanExporter(endpoint=Config.OTEL_ENDPOINT)
            )
        )
        self.tracer = trace.get_tracer(__name__)
        
        # Setup metrics
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(__name__)
        
        # Create metrics
        self.request_counter = self.meter.create_counter(
            "personality.requests",
            description="Number of personality requests"
        )
        
        self.latency_histogram = self.meter.create_histogram(
            "personality.latency",
            description="Request latency in seconds"
        )
        
        logger.info("OpenTelemetry initialized successfully")
    
    @contextmanager
    def trace_span(self, name: str, attributes: Dict = None):
        """Context manager for tracing spans"""
        if not self.enabled:
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span


# ============================================================
# CIRCUIT BREAKER PATTERN
# ============================================================

class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, service_name: str, threshold: int = None, timeout: int = None):
        self.service_name = service_name
        self.threshold = threshold or Config.CIRCUIT_BREAKER_THRESHOLD
        self.timeout = timeout or Config.CIRCUIT_BREAKER_TIMEOUT
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.service_name} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker {self.service_name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.service_name} CLOSED")
                    PrometheusMetrics.CIRCUIT_BREAKER_STATE.labels(service=self.service_name).set(0)
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"Circuit breaker {self.service_name} OPEN after {self.failure_count} failures")
                    PrometheusMetrics.CIRCUIT_BREAKER_STATE.labels(service=self.service_name).set(1)
            
            raise e


# ============================================================
# SHARED EMBEDDING SINGLETON (Enhanced)
# ============================================================

class EmbeddingSingleton:
    """Thread-safe singleton for embedding model with lazy loading"""
    
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
        """Load the embedding model with error handling"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
            start_time = time.time()
            
            self.model = SentenceTransformer(
                Config.EMBEDDING_MODEL_NAME,
                device=Config.EMBEDDING_DEVICE
            )
            
            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    def encode(self, texts, **kwargs):
        """Encode text(s) to embeddings"""
        return self.model.encode(texts, **kwargs)


class EmbeddingService:
    """Service for generating embeddings with observability"""
    
    def __init__(self):
        self.model = EmbeddingSingleton()
        self.circuit_breaker = CircuitBreaker("embedding_service")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text to embedding vector"""
        start = time.time()
        
        try:
            vec = self.circuit_breaker.call(
                self.model.encode,
                text,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            latency = time.time() - start
            PrometheusMetrics.EMBEDDING_LATENCY.observe(latency)
            
            return vec
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts in batch"""
        start = time.time()
        
        try:
            vecs = self.circuit_breaker.call(
                self.model.encode,
                texts,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            latency = time.time() - start
            PrometheusMetrics.EMBEDDING_LATENCY.observe(latency)
            
            logger.info(f"Batch encoded {len(texts)} texts in {latency:.3f}s")
            return vecs
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise


# ============================================================
# MODERATION SAFETY LAYER (Real Implementations)
# ============================================================

class ModerationResult:
    """Moderation check result"""
    
    def __init__(self, is_safe: bool, categories: Dict[str, float] = None, reason: str = ""):
        self.is_safe = is_safe
        self.categories = categories or {}
        self.reason = reason


class ModerationService(ABC):
    """Abstract moderation service"""
    
    @abstractmethod
    def check(self, text: str) -> ModerationResult:
        pass


class OpenAIModerationService(ModerationService):
    """OpenAI Moderation API integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.MODERATION_API_KEY
        self.circuit_breaker = CircuitBreaker("openai_moderation")
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided, moderation will be disabled")
    
    def check(self, text: str) -> ModerationResult:
        """Check text using OpenAI Moderation API"""
        if not self.api_key:
            return ModerationResult(is_safe=True)
        
        try:
            import openai
            openai.api_key = self.api_key
            
            def _call_api():
                response = openai.Moderation.create(input=text)
                return response
            
            response = self.circuit_breaker.call(_call_api)
            result = response["results"][0]
            
            is_safe = not result["flagged"]
            categories = {k: v for k, v in result["categories"].items() if v}
            
            PrometheusMetrics.MODERATION_COUNTER.labels(
                result="safe" if is_safe else "flagged"
            ).inc()
            
            if not is_safe:
                logger.warning(f"Content flagged: {categories}")
            
            return ModerationResult(
                is_safe=is_safe,
                categories=categories,
                reason=", ".join(categories.keys()) if categories else ""
            )
            
        except Exception as e:
            logger.error(f"Moderation check failed: {e}")
            # Fail open for availability
            return ModerationResult(is_safe=True)


class PerspectiveAPIModerationService(ModerationService):
    """Google Perspective API integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.MODERATION_API_KEY
        self.circuit_breaker = CircuitBreaker("perspective_moderation")
        self.endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    
    def check(self, text: str) -> ModerationResult:
        """Check text using Perspective API"""
        if not self.api_key:
            return ModerationResult(is_safe=True)
        
        try:
            import requests
            
            def _call_api():
                payload = {
                    "comment": {"text": text},
                    "languages": ["en"],
                    "requestedAttributes": {
                        "TOXICITY": {},
                        "SEVERE_TOXICITY": {},
                        "IDENTITY_ATTACK": {},
                        "INSULT": {},
                        "PROFANITY": {},
                        "THREAT": {}
                    }
                }
                
                response = requests.post(
                    f"{self.endpoint}?key={self.api_key}",
                    json=payload,
                    timeout=5
                )
                response.raise_for_status()
                return response.json()
            
            result = self.circuit_breaker.call(_call_api)
            
            # Check toxicity scores
            scores = {}
            for attr, data in result.get("attributeScores", {}).items():
                score = data["summaryScore"]["value"]
                scores[attr.lower()] = score
            
            # Flag if any score exceeds threshold
            threshold = 0.7
            is_safe = all(score < threshold for score in scores.values())
            
            PrometheusMetrics.MODERATION_COUNTER.labels(
                result="safe" if is_safe else "flagged"
            ).inc()
            
            return ModerationResult(
                is_safe=is_safe,
                categories=scores,
                reason=f"High toxicity: {max(scores, key=scores.get)}" if not is_safe else ""
            )
            
        except Exception as e:
            logger.error(f"Perspective API check failed: {e}")
            return ModerationResult(is_safe=True)


class ModerationServiceFactory:
    """Factory for creating moderation services"""
    
    @staticmethod
    def create() -> ModerationService:
        provider = Config.MODERATION_PROVIDER.lower()
        
        if provider == "openai":
            return OpenAIModerationService()
        elif provider == "perspective":
            return PerspectiveAPIModerationService()
        else:
            logger.warning(f"Unknown moderation provider: {provider}, using passthrough")
            return PassthroughModerationService()


class PassthroughModerationService(ModerationService):
    """Passthrough moderation for testing"""
    
    def check(self, text: str) -> ModerationResult:
        return ModerationResult(is_safe=True)


# ============================================================
# THREAD-SAFE PROFILE STORAGE (Multiple Backends)
# ============================================================

class ProfileStore(ABC):
    """Abstract profile storage interface"""
    
    @abstractmethod
    def load(self, user_id: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def save(self, user_id: str, data: Dict):
        pass
    
    @abstractmethod
    def delete(self, user_id: str):
        pass
    
    @abstractmethod
    def exists(self, user_id: str) -> bool:
        pass


class InMemoryThreadSafeStore(ProfileStore):
    """In-memory storage with thread safety"""
    
    def __init__(self):
        self.store = {}
        self.lock = threading.RLock()
        logger.info("Using in-memory storage backend")
    
    def load(self, user_id: str) -> Optional[Dict]:
        with self.lock:
            data = self.store.get(user_id)
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="load",
                backend="memory",
                status="success" if data else "miss"
            ).inc()
            return data
    
    def save(self, user_id: str, data: Dict):
        with self.lock:
            self.store[user_id] = data
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save",
                backend="memory",
                status="success"
            ).inc()
    
    def delete(self, user_id: str):
        with self.lock:
            self.store.pop(user_id, None)
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="delete",
                backend="memory",
                status="success"
            ).inc()
    
    def exists(self, user_id: str) -> bool:
        with self.lock:
            return user_id in self.store


class RedisProfileStore(ProfileStore):
    """Redis-backed profile storage with connection pooling"""
    
    def __init__(self, url: str = None):
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed")
        
        url = url or Config.REDIS_URL
        self.pool = redis.ConnectionPool.from_url(
            url,
            max_connections=Config.REDIS_MAX_CONNECTIONS,
            decode_responses=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.prefix = "personality:profile:"
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {url}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"
    
    def load(self, user_id: str) -> Optional[Dict]:
        try:
            data = self.client.get(self._key(user_id))
            if data:
                result = json.loads(data)
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load",
                    backend="redis",
                    status="hit"
                ).inc()
                return result
            else:
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="load",
                    backend="redis",
                    status="miss"
                ).inc()
                return None
        except Exception as e:
            logger.error(f"Redis load failed: {e}")
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="load",
                backend="redis",
                status="error"
            ).inc()
            return None
    
    def save(self, user_id: str, data: Dict):
        try:
            self.client.set(
                self._key(user_id),
                json.dumps(data),
                ex=86400 * 30  # 30 day TTL
            )
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save",
                backend="redis",
                status="success"
            ).inc()
        except Exception as e:
            logger.error(f"Redis save failed: {e}")
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save",
                backend="redis",
                status="error"
            ).inc()
    
    def delete(self, user_id: str):
        try:
            self.client.delete(self._key(user_id))
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="delete",
                backend="redis",
                status="success"
            ).inc()
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
    
    def exists(self, user_id: str) -> bool:
        return self.client.exists(self._key(user_id)) > 0


class PostgresProfileStore(ProfileStore):
    """PostgreSQL-backed profile storage with connection pooling"""
    
    def __init__(self, dsn: str = None):
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("psycopg2 package not installed")
        
        dsn = dsn or Config.POSTGRES_DSN
        
        try:
            self.pool = ThreadedConnectionPool(
                Config.POSTGRES_MIN_CONNECTIONS,
                Config.POSTGRES_MAX_CONNECTIONS,
                dsn
            )
            
            # Initialize table
            self._init_table()
            
            logger.info(f"Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def _init_table(self):
        """Create profiles table if not exists"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS personality_profiles (
                        user_id VARCHAR(255) PRIMARY KEY,
                        profile_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_updated_at 
                    ON personality_profiles(updated_at)
                """)
                
                conn.commit()
        finally:
            self.pool.putconn(conn)
    
    def load(self, user_id: str) -> Optional[Dict]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT profile_data FROM personality_profiles WHERE user_id = %s",
                    (user_id,)
                )
                result = cur.fetchone()
                
                if result:
                    PrometheusMetrics.STORAGE_OPERATIONS.labels(
                        operation="load",
                        backend="postgres",
                        status="hit"
                    ).inc()
                    return result[0]
                else:
                    PrometheusMetrics.STORAGE_OPERATIONS.labels(
                        operation="load",
                        backend="postgres",
                        status="miss"
                    ).inc()
                    return None
        except Exception as e:
            logger.error(f"PostgreSQL load failed: {e}")
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="load",
                backend="postgres",
                status="error"
            ).inc()
            return None
        finally:
            self.pool.putconn(conn)
    
    def save(self, user_id: str, data: Dict):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO personality_profiles (user_id, profile_data, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET 
                        profile_data = EXCLUDED.profile_data,
                        updated_at = CURRENT_TIMESTAMP
                """, (user_id, json.dumps(data)))
                
                conn.commit()
                
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="save",
                    backend="postgres",
                    status="success"
                ).inc()
        except Exception as e:
            logger.error(f"PostgreSQL save failed: {e}")
            conn.rollback()
            PrometheusMetrics.STORAGE_OPERATIONS.labels(
                operation="save",
                backend="postgres",
                status="error"
            ).inc()
        finally:
            self.pool.putconn(conn)
    
    def delete(self, user_id: str):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM personality_profiles WHERE user_id = %s",
                    (user_id,)
                )
                conn.commit()
                
                PrometheusMetrics.STORAGE_OPERATIONS.labels(
                    operation="delete",
                    backend="postgres",
                    status="success"
                ).inc()
        except Exception as e:
            logger.error(f"PostgreSQL delete failed: {e}")
            conn.rollback()
        finally:
            self.pool.putconn(conn)
    
    def exists(self, user_id: str) -> bool:
        return self.load(user_id) is not None


class ProfileStoreFactory:
    """Factory for creating storage backends"""
    
    @staticmethod
    def create() -> ProfileStore:
        backend = Config.STORAGE_BACKEND.lower()
        
        if backend == "redis":
            return RedisProfileStore()
        elif backend == "postgres":
            return PostgresProfileStore()
        else:
            return InMemoryThreadSafeStore()


# ============================================================
# PERSONALITY PROFILE (Thread Safe)
# ============================================================

@dataclass
class PersonalityProfile:
    """User personality profile"""
    warmth: float
    verbosity: float
    baseline_warmth: float
    baseline_verbosity: float
    interaction_count: int = 0
    last_updated: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow().isoformat()
    
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
            baseline_warmth=0.7,
            baseline_verbosity=0.5,
            interaction_count=0
        )


class ProfileManager:
    """Manages personality profiles with thread safety"""
    
    def __init__(self, store: ProfileStore):
        self.store = store
        self.lock = threading.RLock()
        self.cache = {}  # Local cache for performance
        self.cache_lock = threading.RLock()
    
    def get_profile(self, user_id: str) -> PersonalityProfile:
        """Get profile with caching"""
        # Check local cache first
        with self.cache_lock:
            if user_id in self.cache:
                return self.cache[user_id]
        
        # Load from storage
        with self.lock:
            data = self.store.load(user_id)
            
            if data:
                profile = PersonalityProfile.from_dict(data)
            else:
                profile = PersonalityProfile.create_default()
                self.store.save(user_id, profile.to_dict())
            
            # Update cache
            with self.cache_lock:
                self.cache[user_id] = profile
            
            return profile
    
    def update_profile(
        self, 
        user_id: str, 
        delta_warmth: float, 
        delta_verbosity: float
    ):
        """Update profile with bounds checking"""
        with self.lock:
            profile = self.get_profile(user_id)
            
            # Apply deltas with bounds
            profile.warmth = np.clip(
                profile.warmth + delta_warmth,
                0.2, 0.95
            )
            profile.verbosity = np.clip(
                profile.verbosity + delta_verbosity,
                0.15, 0.9
            )
            
            profile.interaction_count += 1
            profile.last_updated = datetime.utcnow().isoformat()
            
            # Calculate drift
            drift = abs(profile.warmth - profile.baseline_warmth)
            PrometheusMetrics.TONE_DRIFT_GAUGE.labels(user_id=user_id).set(drift)
            
            # Check for excessive drift
            if drift > Config.MAX_TONE_DRIFT:
                logger.warning(
                    f"User {user_id} tone drift exceeds threshold: {drift:.2f}"
                )
            
            # Persist
            self.store.save(user_id, profile.to_dict())
            
            # Update cache
            with self.cache_lock:
                self.cache[user_id] = profile
    
    def reset_profile(self, user_id: str):
        """Reset profile to baseline"""
        with self.lock:
            profile = PersonalityProfile.create_default()
            self.store.save(user_id, profile.to_dict())
            
            with self.cache_lock:
                self.cache[user_id] = profile


# ============================================================
# LLM CLIENT (Real Verbosity Conditioning)
# ============================================================

class LLMClient:
    """LLM client with advanced verbosity conditioning"""
    
    def __init__(self):
        self.provider = Config.LLM_PROVIDER.lower()
        self.api_key = Config.LLM_API_KEY
        self.circuit_breaker = CircuitBreaker("llm_service")
    
    def _build_system_prompt(self, warmth: float, verbosity: float) -> str:
        """Build dynamic system prompt based on personality"""
        
        # Warmth conditioning
        if warmth < 0.3:
            warmth_instruction = "Be professional and concise. Stick to facts."
        elif warmth < 0.6:
            warmth_instruction = "Be friendly but professional. Balance warmth with efficiency."
        else:
            warmth_instruction = "Be warm, encouraging, and personable. Show empathy and enthusiasm."
        
        # Verbosity conditioning
        if verbosity < 0.3:
            verbosity_instruction = (
                "Be extremely concise. Use 1-2 sentences maximum. "
                "Provide only essential information."
            )
        elif verbosity < 0.5:
            verbosity_instruction = (
                "Be moderately brief. Use 2-4 sentences. "
                "Include key points without excessive detail."
            )
        elif verbosity < 0.7:
            verbosity_instruction = (
                "Provide moderate detail. Use 3-6 sentences. "
                "Explain reasoning and include relevant context."
            )
        else:
            verbosity_instruction = (
                "Provide comprehensive explanations. Use 5-10 sentences. "
                "Include examples, reasoning, and thorough context."
            )
        
        return f"""You are a helpful AI assistant.

Personality parameters:
- Warmth level: {warmth:.2f}/1.0
- Verbosity level: {verbosity:.2f}/1.0

Instructions:
{warmth_instruction}
{verbosity_instruction}

Adapt your response style to match these parameters while remaining helpful and accurate."""
    
    def generate(
        self, 
        user_input: str, 
        warmth: float,
        verbosity: float
    ) -> str:
        """Generate response with personality conditioning"""
        
        system_prompt = self._build_system_prompt(warmth, verbosity)
        
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_input)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_input)
        else:
            # Fallback mock
            return self._generate_mock(warmth, verbosity, user_input)
    
    def _generate_openai(self, system_prompt: str, user_input: str) -> str:
        """Generate using OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            def _call_api():
                response = openai.ChatCompletion.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=Config.LLM_MAX_TOKENS,
                    temperature=Config.LLM_TEMPERATURE
                )
                return response.choices[0].message.content
            
            return self.circuit_breaker.call(_call_api)
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def _generate_anthropic(self, system_prompt: str, user_input: str) -> str:
        """Generate using Anthropic API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            def _call_api():
                message = client.messages.create(
                    model=Config.LLM_MODEL,
                    max_tokens=Config.LLM_MAX_TOKENS,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_input}
                    ]
                )
                return message.content[0].text
            
            return self.circuit_breaker.call(_call_api)
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def _generate_mock(self, warmth: float, verbosity: float, user_input: str) -> str:
        """Mock generation for testing"""
        
        if verbosity < 0.3:
            response = f"Response to: {user_input[:30]}..."
        elif verbosity < 0.7:
            response = f"Here's a moderate response to your query about: {user_input[:50]}. This includes some detail."
        else:
            response = (
                f"Thank you for your question about: {user_input[:50]}. "
                f"Let me provide a comprehensive answer. "
                f"First, consider the context. Second, here are the details. "
                f"Finally, here's additional information that might be helpful."
            )
        
        if warmth > 0.7:
            response = f"Great question! {response} Hope this helps!"
        
        return response


# ============================================================
# ENTERPRISE PERSONALITY SYSTEM
# ============================================================

class EnterprisePersonalityV6Enhanced:
    """Production-ready personality system with full observability"""
    
    def __init__(self):
        logger.info("Initializing Enhanced Enterprise Personality System v6")
        
        # Core services
        self.embedding = EmbeddingService()
        self.moderation = ModerationServiceFactory.create()
        self.profile_manager = ProfileManager(ProfileStoreFactory.create())
        self.llm = LLMClient()
        
        # Observability
        self.otel = OpenTelemetryProvider()
        
        logger.info("System initialized successfully")
    
    def respond(
        self, 
        user_id: str, 
        user_input: str
    ) -> Tuple[str, Dict]:
        """Generate personalized response with full observability"""
        
        start = time.time()
        
        with self.otel.trace_span("personality_respond", {"user_id": user_id}):
            try:
                # Get profile
                profile = self.profile_manager.get_profile(user_id)
                
                # 1. Generate embedding for semantic analysis
                embedding = self.embedding.encode(user_input)
                
                # 2. Semantic similarity (simplified - you'd compare to stored patterns)
                similarity_score = float(np.mean(np.abs(embedding)))
                
                # 3. Adaptive personality update based on interaction
                delta_warmth = 0.0
                delta_verbosity = 0.0
                
                # Example: Adjust based on input length
                input_length = len(user_input.split())
                if input_length > 50:
                    delta_verbosity = Config.VERBOSITY_ADJUSTMENT_RATE
                elif input_length < 10:
                    delta_verbosity = -Config.VERBOSITY_ADJUSTMENT_RATE
                
                # Example: Adjust based on semantic similarity
                if similarity_score > Config.SIMILARITY_THRESHOLD:
                    delta_warmth = Config.WARMTH_ADJUSTMENT_RATE
                
                self.profile_manager.update_profile(
                    user_id,
                    delta_warmth,
                    delta_verbosity
                )
                
                # Refresh profile after update
                profile = self.profile_manager.get_profile(user_id)
                
                # 4. Generate response with personality conditioning
                response = self.llm.generate(
                    user_input,
                    profile.warmth,
                    profile.verbosity
                )
                
                # 5. Moderation check
                moderation_result = self.moderation.check(response)
                if not moderation_result.is_safe:
                    logger.warning(
                        f"Response flagged for user {user_id}: {moderation_result.reason}"
                    )
                    response = "I apologize, but I cannot provide that response."
                
                # 6. Calculate metrics
                drift = abs(profile.warmth - profile.baseline_warmth)
                latency = time.time() - start
                
                # Record metrics
                PrometheusMetrics.REQUEST_COUNTER.labels(
                    user_id=user_id,
                    status="success"
                ).inc()
                PrometheusMetrics.RESPONSE_LATENCY.observe(latency)
                
                # Build metadata
                metadata = {
                    "warmth": profile.warmth,
                    "verbosity": profile.verbosity,
                    "drift": drift,
                    "interaction_count": profile.interaction_count,
                    "similarity_score": similarity_score,
                    "moderation_safe": moderation_result.is_safe,
                    "latency_ms": latency * 1000
                }
                
                logger.info(
                    f"Request completed for {user_id} in {latency:.3f}s "
                    f"(warmth={profile.warmth:.2f}, verbosity={profile.verbosity:.2f})"
                )
                
                return response, metadata
                
            except Exception as e:
                logger.error(f"Request failed for {user_id}: {e}", exc_info=True)
                
                PrometheusMetrics.REQUEST_COUNTER.labels(
                    user_id=user_id,
                    status="error"
                ).inc()
                
                return "I apologize, but an error occurred.", {"error": str(e)}
    
    def health_check(self) -> Dict:
        """System health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check embedding service
        try:
            self.embedding.encode("health check")
            health["components"]["embedding"] = "healthy"
        except Exception as e:
            health["components"]["embedding"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check storage
        try:
            test_profile = PersonalityProfile.create_default()
            self.profile_manager.store.save("_health_check", test_profile.to_dict())
            self.profile_manager.store.delete("_health_check")
            health["components"]["storage"] = "healthy"
        except Exception as e:
            health["components"]["storage"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        return health


# ============================================================
# COMPREHENSIVE LOAD TESTING
# ============================================================

class LoadTestRunner:
    """Comprehensive load testing with statistics"""
    
    def __init__(self, system: EnterprisePersonalityV6Enhanced):
        self.system = system
    
    def embedding_latency_test(
        self, 
        iterations: int = 100,
        text_lengths: List[int] = None
    ) -> Dict:
        """Test embedding generation latency across different text lengths"""
        
        if text_lengths is None:
            text_lengths = [10, 50, 100, 500, 1000]
        
        results = {}
        
        for length in text_lengths:
            text = " ".join(["word"] * length)
            latencies = []
            
            logger.info(f"Testing {iterations} embeddings of length {length}...")
            
            for _ in range(iterations):
                start = time.time()
                self.system.embedding.encode(text)
                latencies.append(time.time() - start)
            
            results[f"length_{length}"] = {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies)
            }
        
        return results
    
    def concurrent_request_test(
        self,
        num_threads: int = 10,
        requests_per_thread: int = 10
    ) -> Dict:
        """Test concurrent request handling"""
        
        results = {
            "threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "latencies": [],
            "errors": 0
        }
        
        def worker(thread_id: int):
            for i in range(requests_per_thread):
                user_id = f"user_{thread_id}"
                try:
                    start = time.time()
                    self.system.respond(user_id, f"Test message {i}")
                    latency = time.time() - start
                    results["latencies"].append(latency)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    results["errors"] += 1
        
        threads = []
        start = time.time()
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        total_time = time.time() - start
        
        results["total_time"] = total_time
        results["total_requests"] = num_threads * requests_per_thread
        results["throughput"] = results["total_requests"] / total_time
        results["mean_latency"] = np.mean(results["latencies"]) if results["latencies"] else 0
        results["p95_latency"] = np.percentile(results["latencies"], 95) if results["latencies"] else 0
        
        return results
    
    def memory_leak_test(self, iterations: int = 1000) -> Dict:
        """Test for memory leaks over many iterations"""
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(iterations):
            self.system.respond(f"user_{i % 100}", f"Test message {i}")
            
            if i % 100 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "delta_mb": final_memory - initial_memory,
            "iterations": iterations
        }
    
    def full_load_test(self) -> Dict:
        """Run comprehensive load test suite"""
        
        logger.info("Starting comprehensive load test...")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_latency": self.embedding_latency_test(iterations=50),
            "concurrent_requests": self.concurrent_request_test(
                num_threads=5,
                requests_per_thread=10
            )
        }
        
        logger.info("Load test completed")
        return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

def main():
    """Example usage of the enhanced system"""
    
    # Initialize system
    system = EnterprisePersonalityV6Enhanced()
    
    # Health check
    health = system.health_check()
    print(f"Health: {health}")
    
    # Example interactions
    user_id = "user_123"
    
    test_inputs = [
        "Hi, how are you?",
        "Can you explain quantum computing in detail?",
        "Thanks!",
        "I need help with a complex machine learning problem involving transformers and attention mechanisms.",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response, metadata = system.respond(user_id, user_input)
        print(f"Assistant: {response}")
        print(f"Metadata: {metadata}")
    
    # Run load test
    load_tester = LoadTestRunner(system)
    results = load_tester.full_load_test()
    
    print(f"\nLoad Test Results:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
