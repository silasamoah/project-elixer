# knowledge_rag_db.py
"""
Complete RAG Memory System with SQLite + FAISS Integration
Combines knowledge collection, vector storage, and retrieval in one system
"""

import sqlite3
import json
import hashlib
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading

logging.basicConfig(level=logging.INFO)

# Try to import FAISS - graceful fallback if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    logging.info("✅ FAISS available for vector search")
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("⚠️ FAISS not available - falling back to similarity search")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("⚠️ SentenceTransformers not available")


# ============================================================
# DATA STRUCTURES
# ============================================================

class SourceType(Enum):
    """Types of knowledge sources"""
    WEB_SEARCH = "web_search"
    PDF_DOCUMENT = "pdf_document"
    USER_MESSAGE = "user_message"
    AI_RESPONSE = "ai_response"
    CACHED_KNOWLEDGE = "cached_knowledge"


class ValidationStatus(Enum):
    """Validation states"""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class KnowledgeChunk:
    """A single piece of knowledge with embeddings"""
    chunk_id: str
    content: str
    source_type: SourceType
    source_name: Optional[str]
    source_url: Optional[str]
    confidence: float
    timestamp: float
    validation_status: ValidationStatus
    embedding: Optional[np.ndarray]  # Vector embedding
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Search result with relevance score"""
    chunk: KnowledgeChunk
    score: float
    retrieval_method: str  # 'vector', 'keyword', 'hybrid'


# ============================================================
# RAG DATABASE CORE
# ============================================================

class RAGKnowledgeDatabase:
    """
    Complete RAG system with:
    - SQLite for structured storage
    - FAISS for vector similarity search
    - Hybrid retrieval (vector + keyword)
    """
    
    def __init__(self, db_path: Path, embedding_model=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Embedding model
        self.embedding_model = embedding_model
        
        # FAISS index
        self.faiss_index = None
        self.faiss_id_map = []  # Maps FAISS index to chunk_id
        
        # Initialize
        self._init_database()
        self._load_faiss_index()
        
        logging.info(f"📚 RAG Knowledge Database initialized: {self.db_path}")
    
    @property
    def conn(self):
        """Thread-local database connection"""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path), 
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                -- Main knowledge chunks table
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT,
                    source_url TEXT,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    validation_status TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );
                
                -- Embeddings stored separately (more efficient)
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    model_name TEXT,
                    FOREIGN KEY (chunk_id) REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE
                );
                
                -- Query cache for repeated searches
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,  -- JSON array of chunk_ids
                    timestamp REAL NOT NULL,
                    expiry_timestamp REAL NOT NULL
                );
                
                -- Chat history (for context)
                CREATE TABLE IF NOT EXISTS chat_history (
                    message_id TEXT PRIMARY KEY,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    sources TEXT,  -- JSON array of chunk_ids used
                    confidence REAL
                );
                
                -- Collections (group related chunks)
                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at REAL NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS collection_chunks (
                    collection_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    PRIMARY KEY (collection_id, chunk_id),
                    FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE CASCADE,
                    FOREIGN KEY (chunk_id) REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE
                );
                
                -- Statistics
                CREATE TABLE IF NOT EXISTS rag_stats (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                );
                
                -- Indices for performance
                CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON knowledge_chunks(timestamp);
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON knowledge_chunks(source_type);
                CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON knowledge_chunks(confidence);
                CREATE INDEX IF NOT EXISTS idx_query_cache_expiry ON query_cache(expiry_timestamp);
                CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp);
                
                -- Initialize stats
                INSERT OR IGNORE INTO rag_stats VALUES 
                    ('total_chunks', 0),
                    ('total_retrievals', 0),
                    ('cache_hits', 0),
                    ('cache_misses', 0);
            """)
            logging.info("✅ Database schema initialized")
    
    def _load_faiss_index(self):
        """Load or create FAISS index"""
        faiss_path = self.db_path.parent / "faiss_index.bin"
        faiss_map_path = self.db_path.parent / "faiss_id_map.pkl"
        
        if not FAISS_AVAILABLE:
            logging.warning("⚠️ FAISS not available - vector search disabled")
            return
        
        if faiss_path.exists() and faiss_map_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(faiss_map_path, 'rb') as f:
                    self.faiss_id_map = pickle.load(f)
                logging.info(f"✅ Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logging.error(f"Error loading FAISS index: {e}")
                self._create_faiss_index()
        else:
            self._create_faiss_index()
    
    def _create_faiss_index(self, dimension=384):
        """Create new FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        # Use HNSW for better performance (change to IndexFlatL2 for exact search)
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        self.faiss_id_map = []
        logging.info(f"✅ Created new FAISS index (dimension={dimension})")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        faiss_path = self.db_path.parent / "faiss_index.bin"
        faiss_map_path = self.db_path.parent / "faiss_id_map.pkl"
        
        try:
            faiss.write_index(self.faiss_index, str(faiss_path))
            with open(faiss_map_path, 'wb') as f:
                pickle.dump(self.faiss_id_map, f)
            logging.info(f"💾 Saved FAISS index: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")

    
    # ========================================================
    # CORE STORAGE METHODS
    # ========================================================
    
    def add_chunk(self, chunk: KnowledgeChunk) -> bool:
        """Add knowledge chunk with embedding"""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                
                # Store chunk data
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge_chunks 
                    (chunk_id, content, source_type, source_name, source_url, 
                     confidence, timestamp, validation_status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.content,
                    chunk.source_type.value,
                    chunk.source_name,
                    chunk.source_url,
                    chunk.confidence,
                    chunk.timestamp,
                    chunk.validation_status.value,
                    json.dumps(chunk.metadata)
                ))
                
                # Store embedding if available
                if chunk.embedding is not None:
                    embedding_blob = chunk.embedding.tobytes()
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunk_embeddings 
                        (chunk_id, embedding, dimension, model_name)
                        VALUES (?, ?, ?, ?)
                    """, (
                        chunk.chunk_id,
                        embedding_blob,
                        len(chunk.embedding),
                        getattr(self.embedding_model, 'model_name', 'unknown')
                    ))
                    
                    # Add to FAISS index
                    self._add_to_faiss(chunk.chunk_id, chunk.embedding)
                
                self.conn.commit()
                self._increment_stat('total_chunks')
                
                return True
                
            except Exception as e:
                logging.error(f"Error adding chunk: {e}")
                self.conn.rollback()
                return False
    
    def _add_to_faiss(self, chunk_id: str, embedding: np.ndarray):
        """Add embedding to FAISS index"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            # Ensure correct shape
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Add to index
            self.faiss_index.add(embedding.astype('float32'))
            self.faiss_id_map.append(chunk_id)
            
        except Exception as e:
            logging.error(f"Error adding to FAISS: {e}")
    
    def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """Retrieve chunk by ID"""
        try:
            cursor = self.conn.cursor()
            
            # Get chunk data
            cursor.execute("""
                SELECT * FROM knowledge_chunks WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get embedding
            cursor.execute("""
                SELECT embedding, dimension FROM chunk_embeddings WHERE chunk_id = ?
            """, (chunk_id,))
            
            emb_row = cursor.fetchone()
            embedding = None
            
            if emb_row:
                embedding = np.frombuffer(emb_row['embedding'], dtype=np.float32)
            
            return KnowledgeChunk(
                chunk_id=row['chunk_id'],
                content=row['content'],
                source_type=SourceType(row['source_type']),
                source_name=row['source_name'],
                source_url=row['source_url'],
                confidence=row['confidence'],
                timestamp=row['timestamp'],
                validation_status=ValidationStatus(row['validation_status']),
                embedding=embedding,
                metadata=json.loads(row['metadata'])
            )
            
        except Exception as e:
            logging.error(f"Error getting chunk: {e}")
            return None
    
    # ========================================================
    # RETRIEVAL METHODS
    # ========================================================
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        method: str = 'hybrid',
        min_confidence: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Main retrieval method with multiple strategies
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: 'vector', 'keyword', or 'hybrid'
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of RetrievalResult objects sorted by score
        """
        self._increment_stat('total_retrievals')
        
        # Check cache first
        cached = self._get_cached_results(query)
        if cached:
            self._increment_stat('cache_hits')
            return cached[:top_k]
        
        self._increment_stat('cache_misses')
        
        # Perform retrieval based on method
        if method == 'vector':
            results = self._vector_search(query, top_k * 2)
        elif method == 'keyword':
            results = self._keyword_search(query, top_k * 2)
        else:  # hybrid
            vector_results = self._vector_search(query, top_k)
            keyword_results = self._keyword_search(query, top_k)
            results = self._merge_results(vector_results, keyword_results)
        
        # Filter by confidence
        results = [r for r in results if r.chunk.confidence >= min_confidence]
        
        # Sort by score and limit
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        
        # Cache results
        self._cache_results(query, results)
        
        return results
    
    def _vector_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Vector similarity search using FAISS"""
        if not FAISS_AVAILABLE or self.faiss_index is None or self.embedding_model is None:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(k, self.faiss_index.ntotal)
            )
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.faiss_id_map):
                    continue
                
                chunk_id = self.faiss_id_map[idx]
                chunk = self.get_chunk(chunk_id)
                
                if chunk:
                    # Convert distance to similarity score (0-1)
                    score = 1.0 / (1.0 + float(dist))
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        retrieval_method='vector'
                    ))
            
            return results
            
        except Exception as e:
            logging.error(f"Vector search error: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Keyword-based search using SQL FTS"""
        try:
            cursor = self.conn.cursor()
            
            # Simple keyword matching (can be improved with FTS5)
            keywords = query.lower().split()
            
            # Build LIKE query
            like_conditions = " OR ".join([f"LOWER(content) LIKE ?" for _ in keywords])
            like_params = [f"%{kw}%" for kw in keywords]
            
            cursor.execute(f"""
                SELECT chunk_id, content,
                       ({" + ".join(["CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END" for _ in keywords])}) as match_count
                FROM knowledge_chunks
                WHERE {like_conditions}
                ORDER BY match_count DESC, confidence DESC
                LIMIT ?
            """, like_params + like_params + [k])
            
            results = []
            for row in cursor.fetchall():
                chunk = self.get_chunk(row['chunk_id'])
                if chunk:
                    # Score based on keyword matches
                    score = min(1.0, row['match_count'] / len(keywords))
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        retrieval_method='keyword'
                    ))
            
            return results
            
        except Exception as e:
            logging.error(f"Keyword search error: {e}")
            return []
    
    def _merge_results(
        self, 
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge and deduplicate results from multiple sources"""
        merged = {}
        
        # Add vector results
        for result in vector_results:
            merged[result.chunk.chunk_id] = result
        
        # Add keyword results (combine scores if duplicate)
        for result in keyword_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in merged:
                # Average the scores
                existing = merged[chunk_id]
                combined_score = (existing.score + result.score) / 2
                merged[chunk_id] = RetrievalResult(
                    chunk=existing.chunk,
                    score=combined_score,
                    retrieval_method='hybrid'
                )
            else:
                result.retrieval_method = 'hybrid'
                merged[chunk_id] = result
        
        return list(merged.values())
    
    # ========================================================
    # CACHE MANAGEMENT
    # ========================================================
    
    def _get_cached_results(self, query: str) -> Optional[List[RetrievalResult]]:
        """Get cached search results"""
        try:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT results FROM query_cache 
                WHERE query_hash = ? AND expiry_timestamp > ?
            """, (query_hash, time.time()))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            chunk_ids = json.loads(row['results'])
            results = []
            
            for chunk_id in chunk_ids:
                chunk = self.get_chunk(chunk_id)
                if chunk:
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=1.0,  # Cached results don't have scores
                        retrieval_method='cached'
                    ))
            
            return results if results else None
            
        except Exception as e:
            logging.error(f"Cache retrieval error: {e}")
            return None
    
    def _cache_results(self, query: str, results: List[RetrievalResult], expiry_hours: int = 24):
        """Cache search results"""
        try:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            chunk_ids = [r.chunk.chunk_id for r in results]
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_cache 
                (query_hash, query, results, timestamp, expiry_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                json.dumps(chunk_ids),
                time.time(),
                time.time() + (expiry_hours * 3600)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Cache storage error: {e}")
    
    # ========================================================
    # CHAT HISTORY
    # ========================================================
    
    def add_chat_message(
        self, 
        user_message: str, 
        ai_response: str,
        sources: List[str] = None,
        confidence: float = 0.0
    ) -> str:
        """Add chat exchange to history"""
        try:
            message_id = hashlib.sha256(
                f"{user_message}{time.time()}".encode()
            ).hexdigest()[:16]
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history 
                (message_id, user_message, ai_response, timestamp, sources, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                user_message,
                ai_response,
                time.time(),
                json.dumps(sources or []),
                confidence
            ))
            
            self.conn.commit()
            return message_id
            
        except Exception as e:
            logging.error(f"Error adding chat message: {e}")
            return ""
    
    def get_chat_history(self, limit: int = 50) -> List[Dict]:
        """Get recent chat history"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'message_id': row['message_id'],
                    'user_message': row['user_message'],
                    'ai_response': row['ai_response'],
                    'timestamp': row['timestamp'],
                    'sources': json.loads(row['sources']),
                    'confidence': row['confidence']
                })
            
            return history
            
        except Exception as e:
            logging.error(f"Error getting chat history: {e}")
            return []
    
    # ========================================================
    # STATISTICS
    # ========================================================
    
    def _increment_stat(self, key: str, amount: int = 1):
        """Increment stat counter"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO rag_stats (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = value + ?
            """, (key, amount, amount))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Stat increment error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT key, value FROM rag_stats")
            stats = {row['key']: row['value'] for row in cursor.fetchall()}
            
            # Add computed stats
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_chunks")
            stats['total_chunks'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM chat_history")
            stats['total_chats'] = cursor.fetchone()['count']
            
            if FAISS_AVAILABLE and self.faiss_index:
                stats['faiss_vectors'] = self.faiss_index.ntotal
            
            # Calculate cache hit rate
            hits = stats.get('cache_hits', 0)
            misses = stats.get('cache_misses', 0)
            total = hits + misses
            stats['cache_hit_rate'] = hits / total if total > 0 else 0.0
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting stats: {e}")
            return {}
    
    # ========================================================
    # MAINTENANCE
    # ========================================================
    
    def clean_expired(self) -> int:
        """Remove expired cache entries"""
        try:
            cursor = self.conn.cursor()
            now = time.time()
            
            cursor.execute("""
                DELETE FROM query_cache WHERE expiry_timestamp < ?
            """, (now,))
            
            deleted = cursor.rowcount
            self.conn.commit()
            
            logging.info(f"🧹 Cleaned {deleted} expired cache entries")
            return deleted
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            return 0
    
    def rebuild_faiss_index(self):
        """Rebuild FAISS index from database"""
        if not FAISS_AVAILABLE or not self.embedding_model:
            logging.warning("Cannot rebuild FAISS index - dependencies missing")
            return
        
        logging.info("🔨 Rebuilding FAISS index...")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT chunk_id, embedding, dimension 
                FROM chunk_embeddings
            """)
            
            embeddings = []
            chunk_ids = []
            
            for row in cursor.fetchall():
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                chunk_ids.append(row['chunk_id'])
            
            if not embeddings:
                logging.warning("No embeddings to rebuild index")
                return
            
            # Create new index
            dimension = embeddings[0].shape[0]
            self._create_faiss_index(dimension)
            
            # Add all embeddings
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.faiss_id_map = chunk_ids
            
            # Save to disk
            self._save_faiss_index()
            
            logging.info(f"✅ Rebuilt FAISS index: {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logging.error(f"Error rebuilding FAISS index: {e}")
    
    def close(self):
        """Close database and save FAISS index"""
        try:
            self._save_faiss_index()
            if hasattr(self._local, 'conn'):
                self._local.conn.close()
            logging.info("✅ Database closed")
        except Exception as e:
            logging.error(f"Error closing database: {e}")


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_rag_database(
    db_path: Path, 
    embedding_model=None
) -> RAGKnowledgeDatabase:
    """Factory function to create RAG database"""
    
    # Initialize embedding model if not provided
    if embedding_model is None and EMBEDDINGS_AVAILABLE:
        logging.info("Loading default embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return RAGKnowledgeDatabase(db_path, embedding_model)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Create database
    db = create_rag_database(Path("./chat_cache/knowledge/knowledge_rag.db"))
    
    # Add some sample chunks
    sample_chunks = [
        KnowledgeChunk(
            chunk_id="chunk_001",
            content="Python is a high-level programming language known for its simplicity.",
            source_type=SourceType.WEB_SEARCH,
            source_name="Python Documentation",
            source_url="https://python.org",
            confidence=0.9,
            timestamp=time.time(),
            validation_status=ValidationStatus.VALID,
            embedding=None,  # Will be generated if model available
            metadata={"topic": "programming"}
        ),
        KnowledgeChunk(
            chunk_id="chunk_002",
            content="Machine learning is a subset of artificial intelligence.",
            source_type=SourceType.PDF_DOCUMENT,
            source_name="AI Textbook",
            source_url=None,
            confidence=0.85,
            timestamp=time.time(),
            validation_status=ValidationStatus.VALID,
            embedding=None,
            metadata={"topic": "AI"}
        )
    ]
    
    # Generate embeddings if model available
    if db.embedding_model:
        for chunk in sample_chunks:
            chunk.embedding = db.embedding_model.encode(chunk.content)
    
    # Add chunks
    for chunk in sample_chunks:
        db.add_chunk(chunk)
    
    print(f"\n✅ Added {len(sample_chunks)} chunks")
    
    # Test retrieval
    results = db.retrieve("What is Python?", top_k=3, method='hybrid')
    
    print(f"\n🔍 Search Results ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.3f} | Method: {result.retrieval_method}")
        print(f"    Content: {result.chunk.content[:100]}...")
    
    # Show stats
    stats = db.get_stats()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    db.close()