"""
Module 10: RAG Basics - Practice Exercises

Practice implementing RAG (Retrieval-Augmented Generation) systems.
Build vector stores, retrieval systems, and complete RAG pipelines.

Author: Claude
Date: 2024
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Exercise 1: Build a Simple Vector Store
# ================================
print("=" * 50)
print("Exercise 1: Build a Simple Vector Store")
print("=" * 50)

"""
Task: Implement a simple in-memory vector store with add, search, and delete operations.

Requirements:
1. Store documents with embeddings
2. Support cosine similarity search
3. Handle document metadata
4. Implement CRUD operations

Your implementation should:
- Use numpy for vector operations
- Support top-k retrieval
- Handle duplicate documents
"""

class SimpleVectorStore:
    """In-memory vector store implementation."""

    def __init__(self):
        """Initialize the vector store."""
        # TODO: Initialize storage for documents, embeddings, and metadata
        pass

    def add(self, document: str, metadata: Dict = None) -> str:
        """
        Add a document to the store.

        Args:
            document: Text to store
            metadata: Optional metadata

        Returns:
            Document ID
        """
        # TODO: Generate embedding and store document
        pass

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of results with documents and scores
        """
        # TODO: Implement similarity search
        pass

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            Success status
        """
        # TODO: Remove document from store
        pass

    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        Update document metadata.

        Args:
            doc_id: Document ID
            metadata: New metadata

        Returns:
            Success status
        """
        # TODO: Update metadata for document
        pass

# Test your implementation
store = SimpleVectorStore()

# Add documents
doc1_id = store.add("Python is a versatile programming language.", {"type": "programming"})
doc2_id = store.add("Machine learning uses algorithms to learn from data.", {"type": "ml"})
doc3_id = store.add("Natural language processing handles human language.", {"type": "nlp"})

# Search
results = store.search("What is Python used for?", k=2)
print(f"Search results: {len(results)} documents found")

# ================================
# Exercise 2: Implement Document Chunking
# ================================
print("\n" + "=" * 50)
print("Exercise 2: Implement Document Chunking")
print("=" * 50)

"""
Task: Create a document chunker that supports multiple chunking strategies.

Requirements:
1. Fixed-size chunking with overlap
2. Sentence-based chunking
3. Paragraph-based chunking
4. Semantic chunking (bonus)

Your implementation should:
- Preserve context between chunks
- Handle edge cases (short documents)
- Maintain chunk metadata (position, source)
"""

class DocumentChunker:
    """Document chunking utilities."""

    def fixed_size_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Create fixed-size chunks with overlap.

        Args:
            text: Document text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        # TODO: Implement fixed-size chunking
        pass

    def sentence_chunks(
        self,
        text: str,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1
    ) -> List[Dict]:
        """
        Create sentence-based chunks.

        Args:
            text: Document text
            sentences_per_chunk: Sentences per chunk
            overlap_sentences: Overlapping sentences

        Returns:
            List of chunks with metadata
        """
        # TODO: Split by sentences and chunk
        pass

    def paragraph_chunks(
        self,
        text: str,
        min_size: int = 100,
        max_size: int = 500
    ) -> List[Dict]:
        """
        Create paragraph-based chunks.

        Args:
            text: Document text
            min_size: Minimum chunk size
            max_size: Maximum chunk size

        Returns:
            List of chunks with metadata
        """
        # TODO: Split by paragraphs and merge if needed
        pass

    def semantic_chunks(
        self,
        text: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Create semantic chunks based on meaning.

        Args:
            text: Document text
            similarity_threshold: Threshold for splitting

        Returns:
            List of chunks with metadata
        """
        # TODO: Use embeddings to find semantic boundaries
        pass

# Test your chunker
chunker = DocumentChunker()

sample_text = """
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction.
The goal is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly.
"""

# Test different chunking strategies
fixed_chunks = chunker.fixed_size_chunks(sample_text, chunk_size=100, overlap=20)
print(f"Fixed chunks: {len(fixed_chunks)}")

# ================================
# Exercise 3: Build a Hybrid Search System
# ================================
print("\n" + "=" * 50)
print("Exercise 3: Build a Hybrid Search System")
print("=" * 50)

"""
Task: Implement a hybrid search system combining keyword and semantic search.

Requirements:
1. BM25 for keyword search
2. Vector similarity for semantic search
3. Reciprocal Rank Fusion for combining results
4. Adjustable weights for each method

Your implementation should:
- Build inverted index for keywords
- Generate and store embeddings
- Combine rankings effectively
"""

class HybridSearch:
    """Hybrid search combining keyword and semantic search."""

    def __init__(self):
        """Initialize the hybrid search system."""
        # TODO: Initialize both search indexes
        pass

    def index_document(self, doc_id: str, content: str):
        """
        Index a document for both search methods.

        Args:
            doc_id: Document identifier
            content: Document content
        """
        # TODO: Index for both keyword and semantic search
        pass

    def bm25_score(
        self,
        query: str,
        doc_id: str,
        k1: float = 1.2,
        b: float = 0.75
    ) -> float:
        """
        Calculate BM25 score.

        Args:
            query: Search query
            doc_id: Document ID
            k1: BM25 parameter
            b: BM25 parameter

        Returns:
            BM25 score
        """
        # TODO: Implement BM25 scoring
        pass

    def semantic_score(self, query: str, doc_id: str) -> float:
        """
        Calculate semantic similarity score.

        Args:
            query: Search query
            doc_id: Document ID

        Returns:
            Similarity score
        """
        # TODO: Calculate cosine similarity
        pass

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results
            keyword_weight: Weight for keyword search
            semantic_weight: Weight for semantic search

        Returns:
            Ranked results
        """
        # TODO: Combine both search methods
        pass

    def reciprocal_rank_fusion(
        self,
        keyword_results: List[str],
        semantic_results: List[str],
        k: int = 60
    ) -> List[str]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        Args:
            keyword_results: Ranked list from keyword search
            semantic_results: Ranked list from semantic search
            k: RRF parameter

        Returns:
            Fused ranking
        """
        # TODO: Implement RRF algorithm
        pass

# Test hybrid search
hybrid = HybridSearch()

# Index documents
documents = {
    "doc1": "Python programming language is known for its simplicity and readability.",
    "doc2": "Machine learning models require large amounts of training data.",
    "doc3": "Python is widely used in data science and machine learning projects."
}

for doc_id, content in documents.items():
    hybrid.index_document(doc_id, content)

# Search with different weight combinations
results = hybrid.hybrid_search("Python for data science", k=3)
print(f"Hybrid search results: {len(results)}")

# ================================
# Exercise 4: Implement Query Expansion
# ================================
print("\n" + "=" * 50)
print("Exercise 4: Implement Query Expansion")
print("=" * 50)

"""
Task: Build a query expansion system to improve retrieval.

Requirements:
1. Synonym expansion using word embeddings
2. Query rewriting with LLM
3. Pseudo-relevance feedback
4. Multi-query generation

Your implementation should:
- Generate query variations
- Find related terms
- Use initial results to refine query
"""

class QueryExpander:
    """Query expansion for improved retrieval."""

    def expand_with_synonyms(self, query: str, n_synonyms: int = 3) -> List[str]:
        """
        Expand query with synonyms.

        Args:
            query: Original query
            n_synonyms: Number of synonyms per term

        Returns:
            Expanded queries
        """
        # TODO: Find and add synonyms
        pass

    def rewrite_query(self, query: str, context: str = None) -> str:
        """
        Rewrite query using LLM.

        Args:
            query: Original query
            context: Optional context

        Returns:
            Rewritten query
        """
        # TODO: Use LLM to rewrite query
        pass

    def pseudo_relevance_feedback(
        self,
        query: str,
        initial_results: List[str],
        n_terms: int = 5
    ) -> str:
        """
        Expand query using initial results.

        Args:
            query: Original query
            initial_results: Top initial results
            n_terms: Number of terms to add

        Returns:
            Expanded query
        """
        # TODO: Extract terms from initial results
        pass

    def generate_multi_queries(
        self,
        query: str,
        n_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations.

        Args:
            query: Original query
            n_variations: Number of variations

        Returns:
            Query variations
        """
        # TODO: Generate diverse query variations
        pass

    def combine_expansions(
        self,
        original: str,
        expansions: List[str]
    ) -> str:
        """
        Combine original query with expansions.

        Args:
            original: Original query
            expansions: Expansion terms/queries

        Returns:
            Combined query
        """
        # TODO: Intelligently combine queries
        pass

# Test query expansion
expander = QueryExpander()

query = "machine learning algorithms"

# Test different expansion methods
synonyms = expander.expand_with_synonyms(query)
print(f"Synonym expansion: {synonyms}")

rewritten = expander.rewrite_query(query, context="Focus on neural networks")
print(f"Rewritten query: {rewritten}")

# ================================
# Exercise 5: Build a RAG Evaluation System
# ================================
print("\n" + "=" * 50)
print("Exercise 5: Build a RAG Evaluation System")
print("=" * 50)

"""
Task: Create an evaluation framework for RAG systems.

Requirements:
1. Relevance scoring for retrieved documents
2. Answer quality assessment
3. Faithfulness to source documents
4. Performance metrics (latency, tokens)

Your implementation should:
- Provide comprehensive metrics
- Support batch evaluation
- Generate evaluation reports
"""

@dataclass
class RAGMetrics:
    """Metrics for RAG evaluation."""
    relevance_score: float
    answer_quality: float
    faithfulness: float
    latency_ms: float
    tokens_used: int
    sources_used: int

class RAGEvaluator:
    """Evaluation framework for RAG systems."""

    def evaluate_relevance(
        self,
        query: str,
        documents: List[str]
    ) -> float:
        """
        Evaluate relevance of retrieved documents.

        Args:
            query: Search query
            documents: Retrieved documents

        Returns:
            Relevance score (0-1)
        """
        # TODO: Score document relevance
        pass

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        ground_truth: str = None
    ) -> float:
        """
        Evaluate quality of generated answer.

        Args:
            question: Original question
            answer: Generated answer
            ground_truth: Optional correct answer

        Returns:
            Quality score (0-1)
        """
        # TODO: Assess answer quality
        pass

    def evaluate_faithfulness(
        self,
        answer: str,
        source_documents: List[str]
    ) -> float:
        """
        Check if answer is faithful to sources.

        Args:
            answer: Generated answer
            source_documents: Source documents

        Returns:
            Faithfulness score (0-1)
        """
        # TODO: Check answer grounding
        pass

    def evaluate_rag_pipeline(
        self,
        rag_function,
        test_queries: List[Dict]
    ) -> Dict:
        """
        Evaluate complete RAG pipeline.

        Args:
            rag_function: RAG query function
            test_queries: Test queries with ground truth

        Returns:
            Evaluation report
        """
        # TODO: Run comprehensive evaluation
        pass

    def generate_report(
        self,
        metrics: List[RAGMetrics]
    ) -> str:
        """
        Generate evaluation report.

        Args:
            metrics: List of metrics

        Returns:
            Formatted report
        """
        # TODO: Create detailed report
        pass

# Test evaluation
evaluator = RAGEvaluator()

# Sample data for evaluation
test_query = "What is machine learning?"
test_documents = [
    "Machine learning is a type of AI that learns from data.",
    "ML algorithms improve through experience."
]
test_answer = "Machine learning is a branch of AI that enables systems to learn from data."

# Evaluate components
relevance = evaluator.evaluate_relevance(test_query, test_documents)
print(f"Relevance score: {relevance}")

quality = evaluator.evaluate_answer_quality(test_query, test_answer)
print(f"Answer quality: {quality}")

faithfulness = evaluator.evaluate_faithfulness(test_answer, test_documents)
print(f"Faithfulness score: {faithfulness}")

# ================================
# Challenge: Production RAG Service
# ================================
print("\n" + "=" * 50)
print("Challenge: Production RAG Service")
print("=" * 50)

"""
Challenge: Build a complete production-ready RAG service with all features.

Requirements:
1. Document management (CRUD operations)
2. Multiple retrieval strategies
3. Caching and optimization
4. Monitoring and metrics
5. Error handling and retries
6. API interface

Your implementation should include:
- Async operations
- Rate limiting
- Security (input validation)
- Logging
- Configuration management
- Health checks
"""

class ProductionRAGService:
    """Production-ready RAG service."""

    def __init__(self, config: Dict):
        """
        Initialize the service.

        Args:
            config: Service configuration
        """
        # TODO: Initialize all components
        pass

    async def add_document(
        self,
        content: str,
        metadata: Dict,
        doc_id: str = None
    ) -> str:
        """
        Add document to the service.

        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        # TODO: Implement document addition
        pass

    async def query(
        self,
        question: str,
        filters: Dict = None,
        options: Dict = None
    ) -> Dict:
        """
        Process a RAG query.

        Args:
            question: User question
            filters: Metadata filters
            options: Query options

        Returns:
            Query result with answer and metadata
        """
        # TODO: Implement complete RAG pipeline
        pass

    def get_metrics(self) -> Dict:
        """
        Get service metrics.

        Returns:
            Service metrics
        """
        # TODO: Return comprehensive metrics
        pass

    def health_check(self) -> Dict:
        """
        Perform health check.

        Returns:
            Health status
        """
        # TODO: Check all components
        pass

# Configuration for production service
config = {
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-5",
    "vector_store": "faiss",
    "cache_ttl": 3600,
    "max_documents": 10000,
    "rate_limit": 100,  # queries per minute
}

# Initialize service
service = ProductionRAGService(config)

print("\n" + "=" * 50)
print("Exercises Complete!")
print("Implement each function and test your solutions")
print("=" * 50)