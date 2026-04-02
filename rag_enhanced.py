"""
Advanced RAG Engine with Hybrid Search, Reranking, and Query Enhancement
Integrates seamlessly with existing Alex AI Tutor system
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import json
from collections import defaultdict
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("⚠️ rank-bm25 not installed. Run: pip install rank-bm25")
    BM25Okapi = None

try:
    import faiss
except ImportError:
    print("⚠️ faiss not installed. Run: pip install faiss-cpu")
    faiss = None


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    """Enhanced document representation"""
    id: str
    content: str
    source: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)


@dataclass
class RetrievalResult:
    """Result from retrieval with scoring details"""
    document: Document
    score: float
    retrieval_method: str = "hybrid"
    rank: int = 0
    
    def __repr__(self):
        return f"RetrievalResult(source={self.document.source}, score={self.score:.3f}, method={self.retrieval_method})"


# ============================================================
# QUERY ENHANCEMENT
# ============================================================

class QueryEnhancer:
    """Enhance queries for better retrieval"""
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """Generate query variations"""
        variations = [query]
        
        # Add question variations
        if not query.endswith('?'):
            variations.append(query + '?')
        
        # Add "what is" variation for definitions
        if len(query.split()) <= 3 and not query.lower().startswith(('what', 'how', 'why')):
            variations.append(f"What is {query}?")
            variations.append(f"Explain {query}")
        
        return variations
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who'}
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) retrieval"""
    
    def __init__(self, embedding_model, alpha: float = 0.5):
        """
        Args:
            embedding_model: Sentence transformer model
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
        """
        self.embedding_model = embedding_model
        self.alpha = alpha
        
        # Vector store (FAISS)
        self.dimension = None
        self.index = None
        
        # Sparse retrieval (BM25)
        self.bm25 = None
        self.bm25_corpus = []
        
        # Document storage
        self.documents: List[Document] = []
        self.id_to_idx: Dict[str, int] = {}
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both dense and sparse indices"""
        if not documents:
            return
        
        # Add to document store
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.id_to_idx[doc.id] = start_idx + i
        
        # Build dense index (FAISS)
        if faiss:
            embeddings = []
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = self.embedding_model.encode(
                        doc.content, 
                        convert_to_numpy=True
                    ).astype(np.float32)
                embeddings.append(doc.embedding)
            
            embeddings = np.vstack(embeddings).astype(np.float32)
            
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)
        
        # Build sparse index (BM25)
        if BM25Okapi:
            new_corpus = [self._tokenize(doc.content) for doc in documents]
            self.bm25_corpus.extend(new_corpus)
            self.bm25 = BM25Okapi(self.bm25_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return re.findall(r'\w+', text.lower())
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        use_dense: bool = True,
        use_sparse: bool = True
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining dense and sparse methods
        """
        if not self.documents:
            return []
        
        dense_scores = {}
        sparse_scores = {}
        
        # Dense retrieval (Vector similarity)
        if use_dense and self.index:
            query_emb = self.embedding_model.encode(
                query, 
                convert_to_numpy=True
            ).astype(np.float32).reshape(1, -1)
            
            faiss.normalize_L2(query_emb)
            
            # Search more than top_k for fusion
            search_k = min(top_k * 3, len(self.documents))
            scores, indices = self.index.search(query_emb, search_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    dense_scores[idx] = float(score)
        
        # Sparse retrieval (BM25)
        if use_sparse and self.bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k BM25 results
            top_indices = np.argsort(bm25_scores)[-top_k * 3:][::-1]
            
            for idx in top_indices:
                sparse_scores[idx] = float(bm25_scores[idx])
        
        # Combine scores
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        combined_scores = {}
        for idx in all_indices:
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            
            # Normalize sparse scores to [0, 1] range
            if sparse_scores:
                max_sparse = max(sparse_scores.values())
                s_score = s_score / max_sparse if max_sparse > 0 else 0.0
            
            # Weighted combination
            combined_scores[idx] = self.alpha * d_score + (1 - self.alpha) * s_score
        
        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Create results
        results = []
        for rank, (idx, score) in enumerate(sorted_indices):
            doc = self.documents[idx]
            results.append(RetrievalResult(
                document=doc,
                score=score,
                retrieval_method="hybrid",
                rank=rank
            ))
        
        return results


# ============================================================
# SIMPLE RERANKER
# ============================================================

class SimpleReranker:
    """Lightweight reranking based on keyword overlap and position"""
    
    @staticmethod
    def rerank(
        query: str, 
        results: List[RetrievalResult], 
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Rerank results based on query relevance"""
        if not results:
            return []
        
        query_terms = set(query.lower().split())
        
        for result in results:
            content_lower = result.document.content.lower()
            
            # Keyword overlap bonus
            content_terms = set(content_lower.split())
            overlap = len(query_terms & content_terms)
            overlap_score = overlap / len(query_terms) if query_terms else 0
            
            # Position bonus (earlier mentions = higher score)
            position_score = 0.0
            for term in query_terms:
                pos = content_lower.find(term)
                if pos != -1:
                    # Earlier position = higher score
                    position_score += 1.0 / (1 + pos / 100.0)
            
            # Combine with original score
            result.score = (
                0.6 * result.score + 
                0.3 * overlap_score + 
                0.1 * (position_score / len(query_terms) if query_terms else 0)
            )
        
        # Resort and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============================================================
# ADVANCED RAG ENGINE
# ============================================================

class AdvancedRAGEngine:
    """Complete RAG system with hybrid retrieval and reranking"""
    
    def __init__(
        self, 
        embedding_model, 
        cache_dir: Path,
        alpha: float = 0.5
    ):
        """
        Args:
            embedding_model: Sentence transformer model
            cache_dir: Directory for persistent storage
            alpha: Weight for dense vs sparse retrieval
        """
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.retriever = HybridRetriever(embedding_model, alpha)
        self.reranker = SimpleReranker()
        self.query_enhancer = QueryEnhancer()
        
        self.index_file = self.cache_dir / "rag_index.pkl"
        
        # Load existing index if available
        self._load_index()
    def rerank(self, query, results, top_k=5):
        # delegate to real reranker
        return self.reranker.rerank(query, results, top_k)
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the RAG system"""
        self.retriever.add_documents(documents)
        self._save_index()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            use_hybrid: Use both dense and sparse retrieval
            use_reranking: Apply reranking
        
        Returns:
            List of retrieval results
        """
        # Enhance query
        query_variations = self.query_enhancer.expand_query(query)
        
        # Retrieve using hybrid method
        all_results = []
        for q in query_variations:
            results = self.retriever.retrieve(
                q, 
                top_k=top_k * 2,  # Get more for reranking
                use_dense=use_hybrid,
                use_sparse=use_hybrid
            )
            all_results.extend(results)
        
        # Deduplicate by document ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.document.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.document.id)
        
        # Rerank if enabled
        if use_reranking:
            unique_results = self.reranker.rerank(query, unique_results, top_k)
        else:
            unique_results = sorted(
                unique_results, 
                key=lambda x: x.score, 
                reverse=True
            )[:top_k]
        
        return unique_results
    
    def build_context(
        self, 
        query: str, 
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        max_context_length: int = 2000
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Build context string from retrieved documents
        
        Returns:
            (context_string, retrieval_results)
        """
        results = self.retrieve(query, top_k, use_hybrid, use_reranking)
        
        if not results:
            return "", []
        
        # Build context with source attribution
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            chunk = result.document.content
            source = result.document.source
            score = result.score
            
            # Add source header
            header = f"[Source {i}: {source} (Relevance: {score:.2f})]\n"
            content = f"{chunk}\n\n"
            
            section = header + content
            
            # Check length limit
            if current_length + len(section) > max_context_length:
                # Truncate to fit
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful space left
                    section = section[:remaining] + "...\n\n"
                    context_parts.append(section)
                break
            
            context_parts.append(section)
            current_length += len(section)
        
        context = "".join(context_parts).strip()
        return context, results
    
    def clear_index(self):
        """Clear all documents and indices"""
        self.retriever = HybridRetriever(self.embedding_model, self.retriever.alpha)
        if self.index_file.exists():
            self.index_file.unlink()
    
    def _save_index(self):
        """Save index to disk"""
        try:
            data = {
                'documents': self.retriever.documents,
                'id_to_idx': self.retriever.id_to_idx,
                'bm25_corpus': self.retriever.bm25_corpus,
                'alpha': self.retriever.alpha
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(data, f)
            
        except Exception as e:
            print(f"⚠️ Failed to save RAG index: {e}")
    
    def _load_index(self):
        """Load index from disk"""
        if not self.index_file.exists():
            return
        
        try:
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
            
            self.retriever.documents = data['documents']
            self.retriever.id_to_idx = data['id_to_idx']
            self.retriever.bm25_corpus = data['bm25_corpus']
            self.retriever.alpha = data['alpha']
            
            # Rebuild indices
            if self.retriever.documents:
                if faiss:
                    embeddings = np.vstack([
                        doc.embedding for doc in self.retriever.documents
                    ]).astype(np.float32)
                    
                    self.retriever.dimension = embeddings.shape[1]
                    self.retriever.index = faiss.IndexFlatIP(self.retriever.dimension)
                    faiss.normalize_L2(embeddings)
                    self.retriever.index.add(embeddings)
                
                if BM25Okapi and self.retriever.bm25_corpus:
                    self.retriever.bm25 = BM25Okapi(self.retriever.bm25_corpus)
            
            print(f"✅ Loaded RAG index: {len(self.retriever.documents)} documents")
            
        except Exception as e:
            print(f"⚠️ Failed to load RAG index: {e}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_rag_engine(
    embedding_model, 
    cache_dir: Path, 
    alpha: float = 0.5
) -> AdvancedRAGEngine:
    """
    Factory function to create a RAG engine
    
    Args:
        embedding_model: Sentence transformer model
        cache_dir: Cache directory
        alpha: Weight for dense vs sparse (0.5 = balanced)
    """
    return AdvancedRAGEngine(embedding_model, cache_dir, alpha)


def documents_from_chunks(
    chunks: List[str], 
    sources: List[str], 
    embeddings: np.ndarray
) -> List[Document]:
    """
    Convert legacy chunk format to Document objects
    
    Args:
        chunks: List of text chunks
        sources: List of source filenames
        embeddings: Numpy array of embeddings
    
    Returns:
        List of Document objects
    """
    documents = []
    
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        emb = embeddings[i] if i < len(embeddings) else None
        
        doc = Document(
            id=f"{source}_{i}",
            content=chunk,
            source=source,
            embedding=emb,
            metadata={'chunk_index': i}
        )
        documents.append(doc)
    
    return documents


# ============================================================
# TESTING & DEBUGGING
# ============================================================

def test_rag_engine():
    """Simple test function"""
    print("🧪 Testing RAG Engine...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        engine = create_rag_engine(model, Path('./test_cache'))
        
        # Create test documents
        docs = [
            Document(
                id="doc1",
                content="Python is a high-level programming language.",
                source="python_guide.pdf"
            ),
            Document(
                id="doc2",
                content="Machine learning is a subset of artificial intelligence.",
                source="ml_basics.pdf"
            ),
            Document(
                id="doc3",
                content="Python is widely used in machine learning and data science.",
                source="python_ml.pdf"
            ),
        ]
        
        engine.add_documents(docs)
        
        # Test retrieval
        query = "What is Python used for?"
        context, results = engine.build_context(query, top_k=2)
        
        print(f"\n📝 Query: {query}")
        print(f"\n🎯 Results: {len(results)}")
        for r in results:
            print(f"  - {r.document.source}: {r.score:.3f}")
        
        print(f"\n📄 Context:\n{context}")
        print("\n✅ RAG Engine test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_rag_engine()