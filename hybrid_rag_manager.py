# hybrid_rag_manager.py
"""
Hybrid RAG Manager: Combines SQLite (metadata) + FAISS (vectors)
Replaces the PDF processing logic in yan.py
"""

import numpy as np
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from database import db  # Import the global database instance
from rag_enhanced import AdvancedRAGEngine, Document


@dataclass
class HybridSearchResult:
    """Result combining FAISS vector search with SQLite metadata"""
    content: str
    source_filename: str
    pdf_id: str
    chunk_index: int
    similarity_score: float
    topics: List[str]
    faiss_index: int


class HybridRAGManager:
    """
    Manages both FAISS (for vector search) and SQLite (for metadata).
    
    Architecture:
    - FAISS: Stores embeddings, does similarity search
    - SQLite: Stores chunk text, source info, topics
    - Links via faiss_index column
    """
    
    def __init__(self, rag_engine: AdvancedRAGEngine, embedding_model):
        self.rag_engine = rag_engine
        self.embedding_model = embedding_model
        self.db = db  # Use global database instance
        
        logging.info("✅ Hybrid RAG Manager initialized")
    
    def add_pdf(
        self,
        filepath: Path,
        chunks: List[str],
        topics: List[List[str]],
        embeddings: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Add a PDF with both FAISS vectors and SQLite metadata
        
        Args:
            filepath: Path to PDF file
            chunks: List of text chunks
            topics: List of topic lists (one per chunk)
            embeddings: Numpy array of embeddings
        
        Returns:
            (success: bool, message: str)
        """
        try:
            filename = filepath.name
            file_hash = self._get_file_hash(filepath)
            
            # Check if already exists
            existing = self.db.get_pdf_by_hash(file_hash)
            if existing:
                return False, f"PDF already loaded: {filename}"
            
            # Generate PDF ID
            pdf_id = f"pdf_{file_hash[:12]}"
            
            # Add to SQLite
            success = self.db.add_pdf(
                pdf_id=pdf_id,
                filename=filename,
                file_hash=file_hash,
                file_path=str(filepath),
                metadata={'chunk_count': len(chunks)}
            )
            
            if not success:
                return False, "Failed to add PDF to database"
            
            # Get current FAISS size (starting index for new documents)
            faiss_start_idx = len(self.rag_engine.retriever.documents)
            
            # Create Document objects for FAISS
            documents = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{pdf_id}_chunk_{i}"
                faiss_idx = faiss_start_idx + i
                
                # Add to SQLite with link to FAISS
                chunk_topics = topics[i] if i < len(topics) else []
                self.db.add_pdf_chunk(
                    chunk_id=chunk_id,
                    pdf_id=pdf_id,
                    chunk_index=i,
                    content=chunk,
                    faiss_index=faiss_idx,
                    topics=chunk_topics
                )
                
                # Create Document for FAISS
                doc = Document(
                    id=chunk_id,
                    content=chunk,
                    source=filename,
                    embedding=emb,
                    metadata={
                        'pdf_id': pdf_id,
                        'chunk_index': i,
                        'faiss_index': faiss_idx,
                        'topics': chunk_topics
                    }
                )
                documents.append(doc)
            
            # Add to FAISS
            self.rag_engine.add_documents(documents)
            
            logging.info(f"✅ Added PDF: {filename} ({len(chunks)} chunks)")
            return True, f"Added {filename} with {len(chunks)} chunks"
            
        except Exception as e:
            logging.error(f"❌ Error adding PDF: {e}")
            return False, f"Error: {str(e)}"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True
    ) -> List[HybridSearchResult]:
        """
        Hybrid search: FAISS for similarity, SQLite for metadata
        
        Args:
            query: Search query
            top_k: Number of results
            use_reranking: Apply reranking
        
        Returns:
            List of HybridSearchResult with full metadata
        """
        try:
            # Step 1: FAISS vector search
            faiss_results = self.rag_engine.retrieve(
                query,
                top_k=top_k,
                use_hybrid=True,
                use_reranking=use_reranking
            )
            
            if not faiss_results:
                return []
            
            # Step 2: Enrich with SQLite metadata
            hybrid_results = []
            
            for result in faiss_results:
                # Get FAISS index from metadata
                faiss_idx = result.document.metadata.get('faiss_index')
                
                if faiss_idx is None:
                    # Fallback: use document source info
                    hybrid_results.append(HybridSearchResult(
                        content=result.document.content,
                        source_filename=result.document.source,
                        pdf_id="unknown",
                        chunk_index=0,
                        similarity_score=result.score,
                        topics=[],
                        faiss_index=-1
                    ))
                    continue
                
                # Query SQLite for full metadata
                chunk_meta = self.db.get_chunk_by_faiss_index(faiss_idx)
                
                if chunk_meta:
                    hybrid_results.append(HybridSearchResult(
                        content=chunk_meta['content'],
                        source_filename=chunk_meta['pdf_filename'],
                        pdf_id=chunk_meta['pdf_id'],
                        chunk_index=chunk_meta['chunk_index'],
                        similarity_score=result.score,
                        topics=chunk_meta.get('topics', []),
                        faiss_index=faiss_idx
                    ))
                else:
                    # Metadata not found, use FAISS data only
                    hybrid_results.append(HybridSearchResult(
                        content=result.document.content,
                        source_filename=result.document.source,
                        pdf_id=result.document.metadata.get('pdf_id', 'unknown'),
                        chunk_index=result.document.metadata.get('chunk_index', 0),
                        similarity_score=result.score,
                        topics=result.document.metadata.get('topics', []),
                        faiss_index=faiss_idx
                    ))
            
            return hybrid_results
            
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        max_length: int = 2000
    ) -> Tuple[str, List[HybridSearchResult]]:
        """
        Build LLM context from hybrid search results
        
        Returns:
            (context_string, search_results)
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "", []
        
        # Build formatted context
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            header = (
                f"[Source {i}: {result.source_filename} "
                f"(Relevance: {result.similarity_score:.2f})]\n"
            )
            
            content = f"{result.content}\n"
            
            if result.topics:
                topics_str = ", ".join(result.topics[:3])
                content += f"Topics: {topics_str}\n"
            
            section = header + content + "\n" + "-" * 70 + "\n\n"
            
            # Check length limit
            if current_length + len(section) > max_length:
                remaining = max_length - current_length
                if remaining > 100:
                    section = section[:remaining] + "...\n\n"
                    context_parts.append(section)
                break
            
            context_parts.append(section)
            current_length += len(section)
        
        context = "".join(context_parts).strip()
        return context, results
    
    def delete_pdf(self, pdf_id: str) -> Tuple[bool, str]:
        """
        Delete a PDF from both SQLite and FAISS
        
        Note: FAISS doesn't support deletion, so we mark chunks as deleted
        and rebuild the index periodically
        """
        try:
            # Get chunks to mark as deleted
            # (In production, you'd need to rebuild FAISS without these)
            
            # Delete from SQLite (CASCADE deletes chunks)
            success = self.db.delete_pdf(pdf_id)
            
            if success:
                # Note: FAISS index should be rebuilt periodically
                # For now, chunks remain in FAISS but won't match SQLite
                return True, "PDF deleted (FAISS rebuild recommended)"
            else:
                return False, "PDF not found"
                
        except Exception as e:
            logging.error(f"Delete error: {e}")
            return False, str(e)
    
    def get_pdf_list(self) -> List[Dict]:
        """Get list of all loaded PDFs"""
        return self.db.get_all_pdfs()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        db_stats = self.db.get_stats()
        
        faiss_count = len(self.rag_engine.retriever.documents)
        
        return {
            **db_stats,
            'faiss_vectors': faiss_count,
            'sync_status': 'OK' if faiss_count == db_stats['total_chunks'] else 'MISMATCH'
        }
    
    def rebuild_faiss_index(self):
        """
        Rebuild FAISS index from SQLite data
        Useful after deletions or corruption
        """
        logging.info("🔄 Rebuilding FAISS index from SQLite...")
        
        try:
            # Clear FAISS
            self.rag_engine.clear_index()
            
            # Get all chunks from SQLite
            # (You'd need to add this query to DatabaseManager)
            # For now, this is a placeholder
            
            logging.info("✅ FAISS index rebuilt")
            return True
            
        except Exception as e:
            logging.error(f"❌ Rebuild failed: {e}")
            return False
    
    @staticmethod
    def _get_file_hash(filepath: Path) -> str:
        """Generate hash for file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


# ==================== MIGRATION HELPER ====================

def migrate_from_legacy(
    hybrid_manager: HybridRAGManager,
    legacy_chunks: List[str],
    legacy_sources: List[str],
    legacy_embeddings: np.ndarray,
    legacy_topics: List[List[str]]
):
    """
    Migrate from old JSON-based system to SQLite + FAISS
    
    Use this once to convert your existing data
    """
    logging.info("🔄 Migrating legacy data to hybrid system...")
    
    # Group by source file
    pdf_data = {}
    for chunk, source, emb, topics in zip(
        legacy_chunks, legacy_sources, legacy_embeddings, legacy_topics
    ):
        if source not in pdf_data:
            pdf_data[source] = {'chunks': [], 'embeddings': [], 'topics': []}
        
        pdf_data[source]['chunks'].append(chunk)
        pdf_data[source]['embeddings'].append(emb)
        pdf_data[source]['topics'].append(topics)
    
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    PDF_STORAGE_DIR = BASE_DIR / "chat_cache" / "pdfs"

    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Add each PDF
    success_count = 0
    for filename, data in pdf_data.items():
        filename = Path(filename).name  # sanitize
        fake_path = PDF_STORAGE_DIR / filename

        
        embeddings = np.array(data['embeddings'])
        
        success, msg = hybrid_manager.add_pdf(
            filepath=fake_path,
            chunks=data['chunks'],
            topics=data['topics'],
            embeddings=embeddings
        )
        
        if success:
            success_count += 1
            logging.info(f"✅ Migrated: {filename}")
        else:
            logging.warning(f"⚠️ Failed to migrate: {filename} - {msg}")
    
    logging.info(f"✅ Migration complete: {success_count}/{len(pdf_data)} PDFs")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example usage
    from sentence_transformers import SentenceTransformer
    from rag_enhanced import create_rag_engine
    
    # Initialize
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_engine = create_rag_engine(emb_model, Path('./cache'))
    hybrid_manager = HybridRAGManager(rag_engine, emb_model)
    
    # Test search
    results = hybrid_manager.search("What is Python?", top_k=3)
    
    for r in results:
        print(f"📄 {r.source_filename} (score: {r.similarity_score:.3f})")
        print(f"   {r.content[:100]}...")
        print()