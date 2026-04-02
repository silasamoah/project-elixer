# database.py
"""
SQLite database manager for Alex AI Tutor
Handles all structured data storage while FAISS handles vectors
"""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager


class DatabaseManager:
    """Thread-safe SQLite database manager"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_database()

    @property
    def conn(self):
        """Thread-local database connection"""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=10.0
            )
            self._local.conn.row_factory = sqlite3.Row  # Dict-like access
        return self._local.conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _init_database(self):
        """Create all tables if they don't exist"""
        with self.transaction() as conn:
            c = conn.cursor()

            # Users table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    preferences TEXT,  -- JSON
                    interests TEXT,    -- JSON
                    learning_goals TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP
                )
            """
            )

            # User stats table (normalized from user)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    total_questions INTEGER DEFAULT 0,
                    session_count INTEGER DEFAULT 0,
                    topics_discussed TEXT,  -- JSON
                    favorite_topics TEXT,   -- JSON
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                        ON DELETE CASCADE
                )
            """
            )

            # Chat history table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    confidence_score REAL,
                    search_used INTEGER DEFAULT 0,  -- Boolean
                    sources TEXT,  -- JSON array of sources
                    intent TEXT,   -- greeting, question, followup, etc.
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                        ON DELETE CASCADE
                )
            """
            )

            # Create index for fast queries
            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_user_time 
                ON chat_messages(user_id, timestamp DESC)
            """
            )

            # PDF documents table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    pdf_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    metadata TEXT  -- JSON
                )
            """
            )

            # PDF chunks table (links to FAISS index)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    pdf_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    faiss_index INTEGER,  -- Position in FAISS index
                    topics TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_id) REFERENCES pdf_documents(pdf_id)
                        ON DELETE CASCADE
                )
            """
            )

            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_pdf 
                ON pdf_chunks(pdf_id, chunk_index)
            """
            )

            # Knowledge facts cache
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_facts (
                    fact_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    source_name TEXT,
                    confidence REAL NOT NULL,
                    validation_status TEXT NOT NULL,
                    tags TEXT,  -- JSON array
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """
            )

            # Search cache
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """
            )

            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_search_expiry 
                ON search_cache(expires_at)
            """
            )

    # ==================== USER OPERATIONS ====================

    def create_user(self, user_id: str, name: str = None, email: str = None) -> bool:
        """Create a new user"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()

                c.execute(
                    """
                    INSERT INTO users (user_id, name, email, last_active, preferences)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (user_id, name, email, datetime.now().isoformat(), json.dumps({})),
                )

                # Initialize stats
                c.execute(
                    """
                    INSERT INTO user_stats (user_id)
                    VALUES (?)
                """,
                    (user_id,),
                )

                return True
        except sqlite3.IntegrityError:
            return False  # User already exists

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user profile with stats"""
        c = self.conn.cursor()

        c.execute(
            """
            SELECT u.*, s.total_questions, s.session_count, 
                   s.topics_discussed, s.favorite_topics
            FROM users u
            LEFT JOIN user_stats s ON u.user_id = s.user_id
            WHERE u.user_id = ?
        """,
            (user_id,),
        )

        row = c.fetchone()
        if not row:
            return None

        user = dict(row)

        # Parse JSON fields
        for field in [
            "preferences",
            "interests",
            "learning_goals",
            "topics_discussed",
            "favorite_topics",
        ]:
            if user.get(field):
                user[field] = json.loads(user[field])
            else:
                user[field] = {} if field in ["topics_discussed", "preferences"] else []

        return user

    def update_user(self, user_id: str, **fields) -> bool:
        """Update user fields"""
        if not fields:
            return False

        # Convert lists/dicts to JSON
        for key in ["preferences", "interests", "learning_goals"]:
            if key in fields and isinstance(fields[key], (dict, list)):
                fields[key] = json.dumps(fields[key])

        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [user_id]

        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
                return c.rowcount > 0
        except Exception:
            return False

    def update_user_stats(self, user_id: str, **stats) -> bool:
        """Update user statistics"""
        if not stats:
            return False

        # Convert dicts to JSON
        for key in ["topics_discussed", "favorite_topics"]:
            if key in stats and isinstance(stats[key], (dict, list)):
                stats[key] = json.dumps(stats[key])

        set_clause = ", ".join(f"{k} = ?" for k in stats.keys())
        values = list(stats.values()) + [user_id]

        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(
                    f"UPDATE user_stats SET {set_clause} WHERE user_id = ?", values
                )
                return c.rowcount > 0
        except Exception:
            return False

    # ==================== CHAT HISTORY ====================

    def add_chat_message(
        self,
        user_id: str,
        user_message: str,
        ai_response: str,
        confidence: float = 0.0,
        search_used: bool = False,
        sources: List[Tuple[str, str]] = None,
        intent: str = "question",
    ) -> int:
        """Add a chat message and return its ID"""
        with self.transaction() as conn:
            c = conn.cursor()

            c.execute(
                """
                INSERT INTO chat_messages 
                (user_id, user_message, ai_response, confidence_score, 
                 search_used, sources, intent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    user_message,
                    ai_response,
                    confidence,
                    1 if search_used else 0,
                    json.dumps(sources) if sources else None,
                    intent,
                ),
            )

            return c.lastrowid

    def get_chat_history(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict]:
        """Get chat history for a user"""
        c = self.conn.cursor()

        c.execute(
            """
            SELECT * FROM chat_messages
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """,
            (user_id, limit, offset),
        )

        messages = []
        for row in c.fetchall():
            msg = dict(row)
            if msg["sources"]:
                msg["sources"] = json.loads(msg["sources"])
            msg["search_used"] = bool(msg["search_used"])
            messages.append(msg)

        return messages

    def get_recent_context(self, user_id: str, num_messages: int = 3) -> List[Dict]:
        """Get recent messages for context building"""
        c = self.conn.cursor()

        c.execute(
            """
            SELECT user_message, ai_response, intent, timestamp
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (user_id, num_messages),
        )

        return [dict(row) for row in c.fetchall()]

    def clear_chat_history(self, user_id: str) -> bool:
        """Clear all chat history for a user"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
                return True
        except Exception:
            return False

    # ==================== PDF OPERATIONS ====================

    def add_pdf(
        self,
        pdf_id: str,
        filename: str,
        file_hash: str,
        file_path: str,
        metadata: Dict = None,
    ) -> bool:
        """Add a PDF document"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO pdf_documents 
                    (pdf_id, filename, file_hash, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        pdf_id,
                        filename,
                        file_hash,
                        file_path,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                return True
        except sqlite3.IntegrityError:
            return False  # Duplicate hash

    def add_pdf_chunk(
        self,
        chunk_id: str,
        pdf_id: str,
        chunk_index: int,
        content: str,
        faiss_index: int,
        topics: List[str] = None,
    ) -> bool:
        """Add a PDF chunk (links to FAISS)"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO pdf_chunks 
                    (chunk_id, pdf_id, chunk_index, content, faiss_index, topics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk_id,
                        pdf_id,
                        chunk_index,
                        content,
                        faiss_index,
                        json.dumps(topics) if topics else None,
                    ),
                )

                # Update chunk count
                c.execute(
                    """
                    UPDATE pdf_documents 
                    SET chunk_count = (
                        SELECT COUNT(*) FROM pdf_chunks WHERE pdf_id = ?
                    )
                    WHERE pdf_id = ?
                """,
                    (pdf_id, pdf_id),
                )

                return True
        except Exception:
            return False

    def get_pdf_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Check if PDF already exists by hash"""
        c = self.conn.cursor()
        c.execute("SELECT * FROM pdf_documents WHERE file_hash = ?", (file_hash,))
        row = c.fetchone()

        if row:
            pdf = dict(row)
            if pdf["metadata"]:
                pdf["metadata"] = json.loads(pdf["metadata"])
            return pdf
        return None

    def get_all_pdfs(self) -> List[Dict]:
        """Get all PDF documents"""
        c = self.conn.cursor()
        c.execute("SELECT * FROM pdf_documents ORDER BY upload_date DESC")

        pdfs = []
        for row in c.fetchall():
            pdf = dict(row)
            if pdf["metadata"]:
                pdf["metadata"] = json.loads(pdf["metadata"])
            pdfs.append(pdf)
        return pdfs

    def get_chunk_by_faiss_index(self, faiss_idx: int) -> Optional[Dict]:
        """Get chunk metadata by FAISS index"""
        c = self.conn.cursor()
        c.execute(
            """
            SELECT c.*, p.filename as pdf_filename
            FROM pdf_chunks c
            JOIN pdf_documents p ON c.pdf_id = p.pdf_id
            WHERE c.faiss_index = ?
        """,
            (faiss_idx,),
        )

        row = c.fetchone()
        if row:
            chunk = dict(row)
            if chunk["topics"]:
                chunk["topics"] = json.loads(chunk["topics"])
            return chunk
        return None

    def delete_pdf(self, pdf_id: str) -> bool:
        """Delete a PDF and all its chunks (CASCADE)"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM pdf_documents WHERE pdf_id = ?", (pdf_id,))
                return c.rowcount > 0
        except Exception:
            return False

    # ==================== SEARCH CACHE ====================

    def cache_search(
        self,
        query: str,
        query_hash: str,
        results: List[Tuple[str, str]],
        expiry_days: int = 7,
    ) -> bool:
        """Cache search results"""
        try:
            expires_at = datetime.now().timestamp() + (expiry_days * 86400)

            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT OR REPLACE INTO search_cache
                    (query_hash, query, results, expires_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        query_hash,
                        query,
                        json.dumps(results),
                        datetime.fromtimestamp(expires_at).isoformat(),
                    ),
                )
                return True
        except Exception:
            return False

    def get_cached_search(self, query_hash: str) -> Optional[List[Tuple[str, str]]]:
        """Get cached search results if not expired"""
        c = self.conn.cursor()
        c.execute(
            """
            SELECT results FROM search_cache
            WHERE query_hash = ? AND expires_at > ?
        """,
            (query_hash, datetime.now().isoformat()),
        )

        row = c.fetchone()
        if row:
            return json.loads(row["results"])
        return None

    def clean_expired_cache(self) -> int:
        """Remove expired cache entries"""
        try:
            with self.transaction() as conn:
                c = conn.cursor()
                c.execute(
                    "DELETE FROM search_cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),),
                )
                return c.rowcount
        except Exception:
            return 0

    # ==================== STATISTICS ====================

    def get_stats(self) -> Dict:
        """Get database statistics"""
        c = self.conn.cursor()

        stats = {}

        # User stats
        c.execute("SELECT COUNT(*) as count FROM users")
        stats["total_users"] = c.fetchone()["count"]

        # Chat stats
        c.execute("SELECT COUNT(*) as count FROM chat_messages")
        stats["total_messages"] = c.fetchone()["count"]

        # PDF stats
        c.execute("SELECT COUNT(*) as count FROM pdf_documents")
        stats["total_pdfs"] = c.fetchone()["count"]

        c.execute("SELECT COUNT(*) as count FROM pdf_chunks")
        stats["total_chunks"] = c.fetchone()["count"]

        # Cache stats
        c.execute(
            "SELECT COUNT(*) as count FROM search_cache WHERE expires_at > ?",
            (datetime.now().isoformat(),),
        )
        stats["cached_searches"] = c.fetchone()["count"]

        return stats

    def close(self):
        """Close database connection"""
        if hasattr(self._local, "conn"):
            self._local.conn.close()


# ==================== GLOBAL INSTANCE ====================

DB_PATH = Path("chat_cache/alex_tutor.db")
db = DatabaseManager(DB_PATH)
