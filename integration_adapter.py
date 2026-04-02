# integration_adapter.py
"""
Adapter to integrate RAGKnowledgeDatabase with existing yan.py code
Provides backward compatibility while using new RAG system
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
import hashlib
import json
import os

from knowledge_rag_db import (
    RAGKnowledgeDatabase,
    KnowledgeChunk,
    SourceType,
    ValidationStatus,
    RetrievalResult,
    create_rag_database,
)

logging.basicConfig(level=logging.INFO)


class RAGIntegrationAdapter:
    """
    Adapter that makes RAGKnowledgeDatabase compatible with existing code
    """

    def __init__(self, cache_dir: Path, embedding_model=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Initialize RAG database
        db_path = self.cache_dir / "knowledge_rag.db"
        self.rag_db = create_rag_database(db_path, embedding_model)

        logging.info(f"✅ RAG Integration Adapter initialized")

    def delete_pdf_by_name(self, pdf_name: str) -> bool:
        """
        Delete all chunks associated with a PDF file.

        Args:
            pdf_name: Name of the PDF file to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            cursor = self.rag_db.conn.cursor()

            # Check if exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM knowledge_chunks
                WHERE source_name = ? AND source_type = 'pdf_document'
            """,
                (pdf_name,),
            )

            count = cursor.fetchone()[0]

            if count == 0:
                return False

            # Delete chunks
            cursor.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE source_name = ? AND source_type = 'pdf_document'
            """,
                (pdf_name,),
            )

            self.rag_db.conn.commit()

            # Rebuild FAISS index
            self.rebuild_index()

            debug_log(f"ðŸ—'ï¸ Deleted {count} chunks for PDF: {pdf_name}")
            return True

        except Exception as e:
            debug_log(f"Error deleting PDF {pdf_name}: {e}")
            self.rag_db.conn.rollback()
            return False

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
            results = self.rag_db.retrieve(query, top_k=top_k * 6, method="hybrid")
            
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
            
    def get_cached_web_search(
        self, query: str, max_age_days: int = 7
    ) -> List[Tuple[str, str, str]]:
        """
        Get cached web search results from database

        Args:
            query: Search query
            max_age_days: Maximum age of cache in days

        Returns:
            List of (title, url, content) tuples or empty list if not cached
        """
        try:
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            cursor = self.rag_db.conn.cursor()

            logging.debug(f"🔍 Looking up cache for query: '{query}'")
            logging.debug(f"   Query hash: {query_hash}")

            # Check query cache table
            cursor.execute(
                """
                SELECT results, timestamp FROM query_cache
                WHERE query_hash = ? AND expiry_timestamp > ?
            """,
                (query_hash, time.time()),
            )

            row = cursor.fetchone()

            if not row:
                logging.debug(f"❌ No cache entry found")
                cursor.execute("SELECT COUNT(*) as count FROM query_cache")
                total_cache = cursor.fetchone()["count"]
                logging.debug(f"   Total cache entries in DB: {total_cache}")
                return []

            # Check age
            age_seconds = time.time() - row["timestamp"]
            age_days = age_seconds / 86400

            logging.info(f"✅ Found cache entry (age: {age_days:.1f} days)")

            if age_days > max_age_days:
                logging.debug(
                    f"⏰ Cache too old ({age_days:.1f} days > {max_age_days} days)"
                )
                return []

            # Retrieve chunks
            chunk_ids = json.loads(row["results"])
            logging.info(f"📋 Cache contains {len(chunk_ids)} chunk IDs")

            results = []

            for i, chunk_id in enumerate(chunk_ids):
                chunk = self.rag_db.get_chunk(chunk_id)

                if not chunk:
                    logging.warning(f"  ⚠️ Chunk {i+1} not found: {chunk_id}")
                    continue

                # 🔥 CRITICAL CHECK: Verify source type
                if chunk.source_type != SourceType.WEB_SEARCH:
                    logging.warning(f"  ⚠️ Chunk {i+1} wrong type: {chunk.source_type}")
                    continue

                results.append(
                    (
                        chunk.source_name or "Unknown",
                        chunk.source_url or "",
                        chunk.content,
                    )
                )

                logging.debug(f"  ✅ Retrieved chunk {i+1}: {chunk.source_name[:50]}")

            if results:
                logging.info(f"✅ Successfully retrieved {len(results)} cached results")
            else:
                logging.warning(f"⚠️ Found cache entry but no valid chunks")

            return results

        except Exception as e:
            logging.error(f"❌ Error getting cached web search: {e}")
            import traceback

            traceback.print_exc()
            return []

    def clear_web_search_cache(self, max_age_days: int = 7) -> int:
        """
        Clear old web search results from database

        Args:
            max_age_days: Remove results older than this many days

        Returns:
            Number of items deleted
        """
        try:
            import time

            cutoff_timestamp = time.time() - (max_age_days * 86400)
            cursor = self.rag_db.conn.cursor()

            # Count before deletion
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM knowledge_chunks
                WHERE source_type = ? AND timestamp < ?
            """,
                (SourceType.WEB_SEARCH.value, cutoff_timestamp),
            )

            count_before = cursor.fetchone()["count"]

            # Delete old chunks
            cursor.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE source_type = ? AND timestamp < ?
            """,
                (SourceType.WEB_SEARCH.value, cutoff_timestamp),
            )

            self.rag_db.conn.commit()

            logging.info(f"🧹 Cleared {count_before} old web search results")
            return count_before

        except Exception as e:
            logging.error(f"Error clearing web search cache: {e}")
            return 0

    def get_web_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached web searches

        Returns:
            Dictionary with cache statistics
        """
        try:
            cursor = self.rag_db.conn.cursor()

            # Count web search chunks
            cursor.execute(
                """
                SELECT COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM knowledge_chunks
                WHERE source_type = ?
            """,
                (SourceType.WEB_SEARCH.value,),
            )

            row = cursor.fetchone()

            # Count cached queries
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM query_cache
                WHERE expiry_timestamp > ?
            """,
                (time.time(),),
            )

            query_count = cursor.fetchone()["count"]

            import time

            now = time.time()

            stats = {
                "total_web_results": row["total"],
                "avg_confidence": round(row["avg_confidence"] or 0, 3),
                "oldest_age_hours": round((now - (row["oldest"] or now)) / 3600, 1),
                "newest_age_hours": round((now - (row["newest"] or now)) / 3600, 1),
                "cached_queries": query_count,
            }

            return stats

        except Exception as e:
            logging.error(f"Error getting web search stats: {e}")
            return {}

    # Utility function to view cached web searches
    def view_cached_web_searches(limit: int = 10):
        """
        View all cached web search results
        """
        if not RAG_ADAPTER:
            return "❌ RAG_ADAPTER not initialized"

        try:
            cursor = RAG_ADAPTER.rag_db.conn.cursor()

            # Get all web search chunks
            cursor.execute(
                """
                SELECT source_name, source_url, confidence, timestamp,
                    LENGTH(content) as content_length
                FROM knowledge_chunks
                WHERE source_type = 'web_search'
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()

            if not rows:
                return "📭 No cached web searches found"

            output = f"🌐 **Cached Web Searches ({len(rows)} results)**\n\n"

            for i, row in enumerate(rows, 1):
                age = time.time() - row["timestamp"]

                output += f"{i}. **{row['source_name'][:60]}**\n"
                output += f"   URL: {row['source_url'][:70]}\n"
                output += f"   Age: {format_age(age)} | "
                output += f"Confidence: {row['confidence']:.2f} | "
                output += f"Size: {row['content_length']} chars\n\n"

            # Also show query cache
            cursor.execute(
                """
                SELECT query, timestamp
                FROM query_cache
                ORDER BY timestamp DESC
                LIMIT 5
            """
            )

            cache_rows = cursor.fetchall()

            if cache_rows:
                output += "\n📋 **Recent Query Cache:**\n"
                for row in cache_rows:
                    age = time.time() - row["timestamp"]
                    output += f"• '{row['query']}' (age: {format_age(age)})\n"

            return output

        except Exception as e:
            return f"❌ Error: {e}"

    # Clean up old web search cache
    def clean_old_web_searches(days_old: int = 7):
        """
        Remove web search results older than specified days
        """
        if not RAG_ADAPTER:
            return "❌ RAG_ADAPTER not initialized"

        try:
            import time

            cutoff_timestamp = time.time() - (days_old * 86400)
            cursor = RAG_ADAPTER.rag_db.conn.cursor()

            # Delete old web search chunks
            cursor.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE source_type = 'web_search'
                AND timestamp < ?
            """,
                (cutoff_timestamp,),
            )

            deleted_chunks = cursor.rowcount

            # Clean expired query cache
            cursor.execute(
                """
                DELETE FROM query_cache
                WHERE expiry_timestamp < ?
            """,
                (time.time(),),
            )

            deleted_queries = cursor.rowcount

            RAG_ADAPTER.rag_db.conn.commit()

            return f"🧹 Cleaned {deleted_chunks} old web results and {deleted_queries} expired queries"

        except Exception as e:
            return f"❌ Error: {e}"

    # ========================================================
    # LEGACY COMPATIBILITY - Process PDFs
    # ========================================================

    def process_pdfs_legacy(
        self,
        pdf_files,  # Can be List[paths] OR Dict[filename: text]
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Tuple[List[str], List[str], np.ndarray, List[List[str]]]:
        """
        Process PDFs - Handles BOTH formats:
        1. List of file paths (from flask_server.py)
        2. Dict of {filename: text} (from yan.py after text extraction)

        Args:
            pdf_files: List of PDF file paths OR Dict[filename, text]
            chunk_size: Size of text chunks (not used, kept for compatibility)
            chunk_overlap: Overlap between chunks (not used, kept for compatibility)

        Returns:
            tuple: (chunks, sources, embeddings, topics)
        """

        # Safety check: ensure we return proper format even on complete failure
        if not pdf_files:
            logging.warning("⚠️ No PDF files provided")
            return ([], [], np.array([]), [])

        import PyPDF2
        from nltk.tokenize import sent_tokenize
        from pathlib import Path

        all_chunks = []
        all_sources = []
        all_embeddings = []
        all_topics = []

        # ============================================================
        # DICT INPUT: {filename: text}
        # ============================================================
        if isinstance(pdf_files, dict):
            logging.info(f"📚 Processing {len(pdf_files)} PDFs from text dict...")

            for filename, text in pdf_files.items():
                try:
                    if not text or not text.strip():
                        logging.warning(f"⚠️ No text for {filename}")
                        continue

                    # Sentence split
                    try:
                        sentences = sent_tokenize(text)
                    except LookupError:
                        import nltk
                        nltk.download("punkt", quiet=True)
                        sentences = sent_tokenize(text)

                    chunks = []
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 1000:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "

                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    if not chunks:
                        logging.warning(f"⚠️ No chunks created for {filename}")
                        continue

                    added = 0
                    for chunk_text in chunks:
                        chunk = KnowledgeChunk(
                            chunk_id=hashlib.sha256(
                                f"{filename}_{chunk_text[:50]}".encode()
                            ).hexdigest()[:16],
                            content=chunk_text,
                            source_type=SourceType.PDF_DOCUMENT,
                            source_name=filename,
                            source_url=None,
                            confidence=0.9,
                            timestamp=time.time(),
                            validation_status=ValidationStatus.VALID,
                            embedding=None,  # explicitly set
                            metadata={"chunk_index": len(all_chunks) + added},
                        )

                        if self.rag_db.add_chunk(chunk):
                            all_chunks.append(chunk_text)
                            all_sources.append(filename)
                            added += 1

                    logging.info(f"✅ {filename}: {added} chunks added")

                except Exception as e:
                    logging.error(f"❌ Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()

        # ============================================================
        # LIST INPUT: [file paths]
        # ============================================================
        else:
            logging.info(f"📚 Processing {len(pdf_files)} PDF files from paths...")

            for pdf_path in pdf_files:
                try:
                    logging.info(f"🔍 Processing path: {pdf_path}")

                    pdf_path_str = str(pdf_path).replace("\\", "/")
                    pdf_file = Path(pdf_path_str)

                    if not pdf_file.is_absolute():
                        if (Path.cwd() / pdf_file).exists():
                            pdf_file = (Path.cwd() / pdf_file).resolve()
                        else:
                            logging.error(f"❌ File not found: {pdf_file}")
                            continue

                    if not pdf_file.exists():
                        logging.error(f"❌ File does not exist: {pdf_file}")
                        continue

                    logging.info(f"✅ Found file: {pdf_file}")

                    # Extract text
                    with open(pdf_file, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"

                    if not text.strip():
                        logging.warning(f"⚠️ No text extracted from {pdf_file.name}")
                        continue

                    try:
                        sentences = sent_tokenize(text)
                    except LookupError:
                        import nltk
                        nltk.download("punkt", quiet=True)
                        sentences = sent_tokenize(text)

                    chunks = []
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 1000:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "

                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    added = 0
                    for chunk_text in chunks:
                        chunk = KnowledgeChunk(
                            chunk_id=hashlib.sha256(
                                f"{pdf_file.name}_{chunk_text[:50]}".encode()
                            ).hexdigest()[:16],
                            content=chunk_text,
                            source_type=SourceType.PDF_DOCUMENT,
                            source_name=pdf_file.name,
                            source_url=None,
                            confidence=0.9,
                            timestamp=time.time(),
                            validation_status=ValidationStatus.VALID,
                            metadata={"chunk_index": len(all_chunks) + added},
                        )

                        if self.rag_db.add_chunk(chunk):
                            all_chunks.append(chunk_text)
                            all_sources.append(pdf_file.name)
                            added += 1

                    logging.info(f"✅ {pdf_file.name}: {added} chunks added")

                except Exception as e:
                    logging.error(f"❌ Error processing {pdf_path}: {e}")
                    import traceback
                    traceback.print_exc()

        # ============================================================
        # EMBEDDINGS
        # ============================================================
        if all_chunks and self.rag_db.embedding_model:
            try:
                logging.info("🔄 Generating embeddings...")
                embeddings_list = self.rag_db.embedding_model.encode(
                    all_chunks,
                    show_progress_bar=False,
                )
                all_embeddings = np.array(embeddings_list)

                self.rebuild_index()
                logging.info(f"✅ Generated {len(all_embeddings)} embeddings")

            except Exception as e:
                logging.error(f"❌ Embedding generation failed: {e}")
                all_embeddings = np.array([])

        else:
            if not self.rag_db.embedding_model:
                logging.warning("⚠️ No embeddings generated (embedding model may be missing)")
            all_embeddings = np.array([])

        # Placeholder topics
        all_topics = [[] for _ in all_chunks]

        logging.info(
            f"✅ Total: {len(all_chunks)} chunks from {len(set(all_sources))} PDFs"
        )

        return (all_chunks, all_sources, all_embeddings, all_topics)

    # ========================================================
    # LEGACY COMPATIBILITY - Retrieval
    # ========================================================

    def retrieve_legacy(
        self, query: str, top_k: int = 5
    ) -> Tuple[str, List[str], np.ndarray, float]:
        """
        Retrieve and return in legacy format (context, chunks, scores, max_score)
        Compatible with existing RAG calls
        """
        # Perform retrieval
        results = self.rag_db.retrieve(query, top_k=top_k, method="hybrid")

        if not results:
            return "", [], np.array([]), 0.0

        # Build context string
        context = f"Retrieved Knowledge for: '{query}'\n"
        context += "=" * 70 + "\n\n"

        chunks = []
        scores = []

        for i, result in enumerate(results, 1):
            context += f"[{i}] Source: {result.chunk.source_name or 'Unknown'}\n"
            context += f"    Relevance: {result.score:.3f}\n"
            context += f"    Content: {result.chunk.content}\n\n"
            context += "-" * 70 + "\n\n"

            chunks.append(result.chunk.content)
            scores.append(result.score)

        max_score = max(scores) if scores else 0.0
        scores_array = np.array(scores)

        return context, chunks, scores_array, max_score

    # ========================================================
    # NEW API - Direct Access
    # ========================================================

    def add_web_search_results(
        self,
        query: str,
        results: List[Tuple[str, str, str]],  # (title, url, content)
        confidence: float = 0.7,
    ) -> int:
        """
        Add web search results to knowledge base AND query cache

        🔥 CRITICAL FIX:
        1. Handle empty snippets by using title as fallback
        2. Ensure source_type is WEB_SEARCH (not PDF_DOCUMENT!)

        Returns: Number of chunks added
        """
        count = 0
        chunk_ids = []

        logging.info(
            f"📥 Adding {len(results)} web search results for query: '{query}'"
        )

        for i, (title, url, snippet) in enumerate(results):
            # 🔥 FIX 1: Handle empty snippets
            # Use title as content if snippet is empty
            content = snippet if snippet and len(snippet.strip()) > 20 else title

            # Still skip if both are too short
            if not content or len(content.strip()) < 10:
                logging.warning(f"  ⚠️ Skipping result {i+1} - insufficient content")
                continue

            logging.debug(
                f"  📦 Processing result {i+1}: {title[:50]} (content: {len(content)} chars)"
            )

            # Generate stable chunk ID
            chunk_id = hashlib.sha256(
                f"{url}_{title}_{content[:100]}".encode()
            ).hexdigest()[:16]

            # Generate embedding
            embedding = None
            if self.rag_db.embedding_model:
                try:
                    embedding = self.rag_db.embedding_model.encode(content)
                    logging.debug(f"    ✅ Generated embedding (dim={len(embedding)})")
                except Exception as e:
                    logging.warning(f"    ⚠️ Embedding generation failed: {e}")

            # 🔥 FIX 2: Ensure correct source_type
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                content=content,
                source_type=SourceType.WEB_SEARCH,  # ✅ CRITICAL: Must be WEB_SEARCH!
                source_name=title or "Unknown",
                source_url=url,
                confidence=confidence,
                timestamp=time.time(),
                validation_status=ValidationStatus.VALID,
                embedding=embedding,
                metadata={
                    "query": query,
                    "url": url,
                    "title": title,
                    "index": i,
                    "content_source": "snippet" if snippet else "title",
                },
            )

            # Add to database
            if self.rag_db.add_chunk(chunk):
                chunk_ids.append(chunk_id)
                count += 1
                logging.debug(f"    ✅ Chunk added successfully")
            else:
                logging.warning(f"    ❌ Failed to add chunk")

        logging.info(f"✅ Successfully added {count}/{len(results)} chunks")

        # Create query cache entry only if we added chunks
        if chunk_ids:
            try:
                query_hash = hashlib.md5(query.lower().encode()).hexdigest()
                cursor = self.rag_db.conn.cursor()

                # Store the query-to-chunks mapping
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO query_cache
                    (query_hash, query, results, timestamp, expiry_timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        query_hash,
                        query,
                        json.dumps(chunk_ids),
                        time.time(),
                        time.time() + (7 * 24 * 3600),  # 7 day expiry
                    ),
                )

                self.rag_db.conn.commit()
                logging.info(
                    f"✅ Query cache created: '{query}' → {len(chunk_ids)} chunks"
                )

                # Verify it was created
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM query_cache WHERE query_hash = ?
                """,
                    (query_hash,),
                )

                verify_count = cursor.fetchone()["count"]
                if verify_count > 0:
                    logging.info(f"   ✅ Cache entry verified in database")
                else:
                    logging.error(f"   ❌ Cache entry NOT found after insert!")

            except Exception as e:
                logging.error(f"❌ Failed to create query cache: {e}")
                import traceback

                traceback.print_exc()
        else:
            logging.warning(f"⚠️ No chunks added - query cache NOT created")

        return count

    def add_chat_exchange(
        self,
        user_message: str,
        ai_response: str,
        sources: List[str] = None,
        confidence: float = 0.0,
    ):
        """Add chat exchange to history"""
        return self.rag_db.add_chat_message(
            user_message, ai_response, sources, confidence
        )

    def get_context_for_llm(
        self, query: str, top_k: int = 5, max_length: int = 2000
    ) -> Tuple[str, List[RetrievalResult]]:
        """Get formatted context string for LLM prompt"""
        results = self.rag_db.retrieve(query, top_k=top_k, method="hybrid")

        if not results:
            return "", []

        context = f"Relevant Knowledge:\n"
        context += "=" * 70 + "\n\n"

        current_length = 0
        included_results = []

        for i, result in enumerate(results, 1):
            chunk_text = result.chunk.content

            if current_length + len(chunk_text) > max_length:
                break

            context += f"[{i}] {result.chunk.source_name or 'Unknown'} "
            context += f"(Relevance: {result.score:.2f})\n"
            context += f"{chunk_text}\n\n"

            current_length += len(chunk_text)
            included_results.append(result)

        context += "=" * 70 + "\n"

        return context, included_results

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.rag_db.get_stats()

    def clean_expired(self):
        """Clean expired cache entries"""
        return self.rag_db.clean_expired()

    def rebuild_index(self):
        """Rebuild FAISS index"""
        self.rag_db.rebuild_faiss_index()

    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.3,
        max_length: int = 2000
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve relevant context with score filtering
        
        This is an ALIAS for get_context_for_llm_with_filtering
        to maintain backward compatibility with yan.py
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum relevance score (0.0 to 1.0)
            max_length: Maximum context length
        
        Returns:
            tuple: (context_string, list_of_retrieval_results)
        """
        return self.get_context_for_llm_with_filtering(
            query=query,
            top_k=top_k,
            max_length=max_length,
            min_score=min_score
        )
    
    def retrieve_from_file(
        self,
        query: str,
        filename: str,
        top_k: int = 5,
        min_score: float = 0.3,
        max_length: int = 2000
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve content from a specific PDF file only
        
        Args:
            query: Search query
            filename: Name of the PDF file to search in
            top_k: Number of results to return
            min_score: Minimum relevance score
            max_length: Maximum context length
        
        Returns:
            tuple: (context_string, list_of_retrieval_results)
        """
        try:
            # Get more results than needed for filtering
            results = self.rag_db.retrieve(query, top_k=top_k * 3, method="hybrid")
            
            if not results:
                logging.debug(f"⚠️ No results for query: {query}")
                return "", []
            
            # Filter by filename
            filename_lower = filename.lower()
            file_results = [
                r for r in results
                if r.chunk.source_name and filename_lower in r.chunk.source_name.lower()
            ]
            
            if not file_results:
                logging.debug(f"⚠️ No results found in file: {filename}")
                return "", []
            
            # Filter by score
            filtered_results = [r for r in file_results if r.score >= min_score]
            
            if not filtered_results:
                # Fallback to best result if nothing passes threshold
                if file_results[0].score >= 0.25:
                    filtered_results = [file_results[0]]
                else:
                    return "", []
            
            # Take top_k
            filtered_results = sorted(
                filtered_results, key=lambda x: x.score, reverse=True
            )[:top_k]
            
            # Build context
            context_parts = []
            total_length = 0
            
            for r in filtered_results:
                chunk_text = f"[Source: {r.chunk.source_name}]\n{r.chunk.content}"
                
                if total_length + len(chunk_text) > max_length:
                    break
                
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            logging.info(
                f"✅ Retrieved {len(filtered_results)} chunks from {filename}"
            )
            
            return context, filtered_results
            
        except Exception as e:
            logging.error(f"❌ Error retrieving from file: {e}")
            import traceback
            traceback.print_exc()
            return "", []

    def close(self):
        """Close database"""
        self.rag_db.close()


# ============================================================
# MIGRATION HELPER
# ============================================================


def migrate_legacy_data(
    adapter: RAGIntegrationAdapter,
    legacy_chunks: List[str],
    legacy_sources: List[str],
    legacy_embeddings: np.ndarray = None,
):
    """
    Migrate data from old persistent_data format to new RAG database

    Args:
        adapter: RAGIntegrationAdapter instance
        legacy_chunks: List of text chunks
        legacy_sources: List of source names
        legacy_embeddings: Optional numpy array of embeddings
    """
    import hashlib
    import time

    logging.info(f"🔄 Migrating {len(legacy_chunks)} legacy chunks...")

    migrated = 0

    for i, (chunk_text, source) in enumerate(zip(legacy_chunks, legacy_sources)):
        # Generate chunk ID
        chunk_id = hashlib.sha256(
            f"{source}_{i}_{chunk_text[:50]}".encode()
        ).hexdigest()[:16]

        # Get embedding if available
        embedding = None
        if legacy_embeddings is not None and i < len(legacy_embeddings):
            embedding = legacy_embeddings[i]

        # Create chunk
        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            content=chunk_text,
            source_type=SourceType.PDF_DOCUMENT,
            source_name=source,
            source_url=None,
            confidence=0.8,
            timestamp=time.time(),
            validation_status=ValidationStatus.VALID,
            embedding=embedding,
            metadata={"migrated": True, "original_index": i},
        )

        if adapter.rag_db.add_chunk(chunk):
            migrated += 1

    logging.info(f"✅ Migrated {migrated} chunks")

    # Rebuild FAISS index
    if legacy_embeddings is not None:
        adapter.rebuild_index()

    return migrated


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

    # Initialize adapter
    adapter = RAGIntegrationAdapter(
        cache_dir=Path("./chat_cache"),
        embedding_model=None,  # Will auto-load if available
    )

    # Example 1: Process PDFs (legacy compatible)
    pdf_texts = {
        "sample.pdf": "This is sample text from a PDF document. It contains information about Python programming."
    }

    chunks, sources, embeddings, topics = adapter.process_pdfs_legacy(pdf_texts)
    print(f"\n✅ Processed PDFs: {len(chunks)} chunks")

    # Example 2: Retrieve (legacy compatible)
    context, ret_chunks, scores, max_score = adapter.retrieve_legacy(
        "Python programming", top_k=3
    )
    print(f"\n🔍 Retrieved: {len(ret_chunks)} chunks, max score: {max_score:.3f}")

    # Example 3: Add web search results (new API)
    web_results = [
        (
            "Python Tutorial",
            "https://example.com/python",
            "Python is a programming language...",
        )
    ]

    adapter.add_web_search_results("Python tutorial", web_results)

    # Example 4: Get LLM context (new API)
    llm_context, results = adapter.get_context_for_llm("What is Python?", top_k=3)
    print(f"\n💬 LLM Context length: {len(llm_context)} chars")

    # Show stats
    stats = adapter.get_stats()
    print(f"\n📊 Stats: {stats}")

    adapter.close()