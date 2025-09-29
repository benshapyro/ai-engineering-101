"""
Module 11: Advanced RAG - Practice Exercises

Practice implementing advanced RAG techniques including hybrid search,
query processing, and sophisticated reranking strategies.

Author: Claude
Date: 2024
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Exercise 1: Build a Hybrid Retriever
# ================================
print("=" * 50)
print("Exercise 1: Build a Hybrid Retriever")
print("=" * 50)

"""
Task: Implement a hybrid retriever that combines multiple retrieval methods.

Requirements:
1. Implement BM25 sparse retrieval
2. Implement dense retrieval using embeddings
3. Create a fusion mechanism (RRF or weighted combination)
4. Support different fusion strategies
5. Include result deduplication

Your implementation should:
- Handle edge cases (empty queries, no results)
- Optimize for performance
- Support customizable weights
"""

@dataclass
class SearchResult:
    """Search result with scores."""
    doc_id: str
    content: str
    sparse_score: float = 0.0
    dense_score: float = 0.0
    final_score: float = 0.0

class HybridRetriever:
    """Hybrid retriever combining sparse and dense methods."""

    def __init__(self):
        """Initialize the hybrid retriever."""
        # TODO: Initialize components for sparse and dense retrieval
        pass

    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for both retrieval methods.

        Args:
            documents: List of document strings
        """
        # TODO: Build sparse index (BM25/TF-IDF)
        # TODO: Generate and store embeddings for dense retrieval
        pass

    def bm25_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform BM25 sparse retrieval.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of search results with BM25 scores
        """
        # TODO: Implement BM25 scoring
        # TODO: Return top-k documents
        pass

    def dense_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform dense retrieval using embeddings.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of search results with similarity scores
        """
        # TODO: Generate query embedding
        # TODO: Calculate similarities with document embeddings
        # TODO: Return top-k documents
        pass

    def reciprocal_rank_fusion(
        self,
        results_list: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            results_list: List of result sets from different methods
            k: RRF parameter

        Returns:
            Fused results
        """
        # TODO: Implement RRF algorithm
        # TODO: Handle deduplication
        pass

    def weighted_fusion(
        self,
        sparse_results: List[SearchResult],
        dense_results: List[SearchResult],
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7
    ) -> List[SearchResult]:
        """
        Combine results using weighted scores.

        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            sparse_weight: Weight for sparse scores
            dense_weight: Weight for dense scores

        Returns:
            Fused results
        """
        # TODO: Combine scores with weights
        # TODO: Sort by final score
        pass

    def search(
        self,
        query: str,
        k: int = 5,
        fusion_method: str = "weighted",
        sparse_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results
            fusion_method: 'weighted' or 'rrf'
            sparse_weight: Weight for sparse retrieval (if using weighted)

        Returns:
            Final search results
        """
        # TODO: Execute both search methods
        # TODO: Apply chosen fusion method
        # TODO: Return top-k results
        pass

# Test your implementation
retriever = HybridRetriever()

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables text understanding.",
    "Computer vision processes visual information.",
    "Reinforcement learning uses reward signals."
]

# Index documents
retriever.index_documents(documents)

# Test search
query = "How do neural networks work?"
results = retriever.search(query, k=3)
print(f"Search results for: {query}")
# TODO: Print results

# ================================
# Exercise 2: Implement Query Understanding Pipeline
# ================================
print("\n" + "=" * 50)
print("Exercise 2: Implement Query Understanding Pipeline")
print("=" * 50)

"""
Task: Build a comprehensive query understanding pipeline.

Requirements:
1. Query classification (factual, procedural, comparison, etc.)
2. Entity extraction and disambiguation
3. Query expansion with synonyms and related terms
4. Query decomposition for complex queries
5. Intent detection and routing

Your implementation should:
- Handle multi-turn conversations
- Resolve coreferences
- Support context carryover
"""

class QueryUnderstandingPipeline:
    """Pipeline for query understanding and processing."""

    def __init__(self):
        """Initialize the pipeline."""
        # TODO: Initialize components
        self.conversation_history = []
        pass

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query type and intent.

        Args:
            query: Input query

        Returns:
            Classification results with confidence
        """
        # TODO: Implement pattern-based classification
        # TODO: Use LLM for complex cases
        # TODO: Return intent and confidence score
        pass

    def extract_entities(self, query: str) -> List[Dict[str, str]]:
        """
        Extract and disambiguate entities.

        Args:
            query: Input query

        Returns:
            List of extracted entities with types
        """
        # TODO: Extract named entities
        # TODO: Disambiguate ambiguous entities
        # TODO: Link to knowledge base if available
        pass

    def expand_query(self, query: str, method: str = "all") -> List[str]:
        """
        Expand query with variations.

        Args:
            query: Original query
            method: Expansion method ('synonyms', 'related', 'all')

        Returns:
            List of expanded queries
        """
        # TODO: Generate synonyms
        # TODO: Add related terms
        # TODO: Create query variations
        pass

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        # TODO: Identify query complexity
        # TODO: Split into logical sub-queries
        # TODO: Maintain query relationships
        pass

    def resolve_coreferences(self, query: str) -> str:
        """
        Resolve pronouns and references using context.

        Args:
            query: Query with potential coreferences

        Returns:
            Resolved query
        """
        # TODO: Identify pronouns and references
        # TODO: Find antecedents in conversation history
        # TODO: Replace with resolved entities
        pass

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query processing pipeline.

        Args:
            query: Input query

        Returns:
            Processed query with all enhancements
        """
        # TODO: Apply all processing steps
        # TODO: Update conversation history
        # TODO: Return comprehensive results
        pass

# Test your implementation
pipeline = QueryUnderstandingPipeline()

# Test queries
test_queries = [
    "Compare BERT and GPT models",
    "How does it work?",  # Coreference
    "What are the steps to train a neural network?"
]

for query in test_queries:
    result = pipeline.process_query(query)
    print(f"\nQuery: {query}")
    # TODO: Print processing results

# ================================
# Exercise 3: Create Custom Reranking Algorithm
# ================================
print("\n" + "=" * 50)
print("Exercise 3: Create Custom Reranking Algorithm")
print("=" * 50)

"""
Task: Implement a custom reranking algorithm with multiple strategies.

Requirements:
1. Cross-encoder style relevance scoring
2. Diversity-aware ranking (MMR implementation)
3. Personalization based on user profile
4. Position-aware scoring
5. Freshness and recency factors

Your implementation should:
- Support different reranking strategies
- Handle edge cases gracefully
- Optimize for latency
"""

@dataclass
class Document:
    """Document for reranking."""
    id: str
    content: str
    initial_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_score: float = 0.0

@dataclass
class UserProfile:
    """User profile for personalization."""
    interests: List[str] = field(default_factory=list)
    expertise_level: str = "intermediate"
    interaction_history: List[str] = field(default_factory=list)

class CustomReranker:
    """Custom reranking with multiple strategies."""

    def __init__(self):
        """Initialize reranker."""
        # TODO: Initialize components
        pass

    def cross_encoder_score(
        self,
        query: str,
        document: Document
    ) -> float:
        """
        Calculate cross-encoder style relevance score.

        Args:
            query: Search query
            document: Document to score

        Returns:
            Relevance score (0-1)
        """
        # TODO: Implement relevance scoring
        # TODO: Use LLM or similarity metrics
        pass

    def mmr_rerank(
        self,
        query: str,
        documents: List[Document],
        lambda_param: float = 0.5,
        k: int = 5
    ) -> List[Document]:
        """
        Rerank using Maximal Marginal Relevance.

        Args:
            query: Search query
            documents: Documents to rerank
            lambda_param: Trade-off parameter
            k: Number of documents to select

        Returns:
            Reranked documents with diversity
        """
        # TODO: Implement MMR algorithm
        # TODO: Balance relevance and diversity
        pass

    def personalized_rerank(
        self,
        documents: List[Document],
        user_profile: UserProfile,
        k: int = 5
    ) -> List[Document]:
        """
        Rerank based on user profile.

        Args:
            documents: Documents to rerank
            user_profile: User preferences
            k: Number of documents

        Returns:
            Personalized ranking
        """
        # TODO: Score based on user interests
        # TODO: Adjust for expertise level
        # TODO: Consider interaction history
        pass

    def position_aware_rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = 5
    ) -> List[Document]:
        """
        Rerank considering position bias.

        Args:
            query: Search query
            documents: Documents to rerank
            k: Number of documents

        Returns:
            Position-aware ranking
        """
        # TODO: Apply position decay
        # TODO: Consider click-through patterns
        pass

    def temporal_rerank(
        self,
        documents: List[Document],
        k: int = 5,
        freshness_weight: float = 0.3
    ) -> List[Document]:
        """
        Rerank considering temporal factors.

        Args:
            documents: Documents to rerank
            k: Number of documents
            freshness_weight: Weight for recency

        Returns:
            Temporally-adjusted ranking
        """
        # TODO: Extract temporal metadata
        # TODO: Apply freshness scoring
        # TODO: Balance with relevance
        pass

    def ensemble_rerank(
        self,
        query: str,
        documents: List[Document],
        user_profile: Optional[UserProfile] = None,
        strategies: List[str] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Ensemble reranking using multiple strategies.

        Args:
            query: Search query
            documents: Documents to rerank
            user_profile: Optional user profile
            strategies: List of strategies to use
            k: Number of documents

        Returns:
            Final reranked results
        """
        # TODO: Apply selected strategies
        # TODO: Combine rankings
        # TODO: Return final results
        pass

# Test your implementation
reranker = CustomReranker()

# Create test documents
test_docs = [
    Document("doc1", "Machine learning basics", 0.8),
    Document("doc2", "Deep learning fundamentals", 0.7),
    Document("doc3", "NLP applications", 0.6),
    Document("doc4", "Computer vision guide", 0.5),
    Document("doc5", "Reinforcement learning", 0.4)
]

# Test reranking
query = "machine learning tutorial"
reranked = reranker.ensemble_rerank(query, test_docs, k=3)
print(f"\nReranked results for: {query}")
# TODO: Print results

# ================================
# Exercise 4: Build Production RAG Pipeline
# ================================
print("\n" + "=" * 50)
print("Exercise 4: Build Production RAG Pipeline")
print("=" * 50)

"""
Task: Create a production-ready RAG pipeline with all components.

Requirements:
1. Document ingestion and preprocessing
2. Multi-stage retrieval pipeline
3. Query processing and routing
4. Reranking with fallback strategies
5. Response generation with citations
6. Caching and optimization
7. Monitoring and metrics

Your implementation should:
- Handle errors gracefully
- Support async operations
- Include health checks
- Track performance metrics
"""

class ProductionRAGPipeline:
    """Production-ready RAG pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        # TODO: Initialize all components
        self.config = config
        self.metrics = defaultdict(int)
        pass

    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Ingest and process a document.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Document ID
        """
        # TODO: Preprocess document
        # TODO: Chunk document
        # TODO: Generate embeddings
        # TODO: Index in vector store
        # TODO: Update metadata store
        pass

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Multi-stage retrieval.

        Args:
            query: Search query
            k: Number of documents
            filters: Metadata filters

        Returns:
            Retrieved documents
        """
        # TODO: Query processing
        # TODO: Initial retrieval (hybrid)
        # TODO: Apply filters
        # TODO: Cache results
        pass

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        strategy: str = "auto"
    ) -> List[Document]:
        """
        Rerank documents with fallback.

        Args:
            query: Search query
            documents: Documents to rerank
            strategy: Reranking strategy

        Returns:
            Reranked documents
        """
        # TODO: Select reranking strategy
        # TODO: Apply reranking with timeout
        # TODO: Fallback on error
        pass

    async def generate_response(
        self,
        query: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate response with citations.

        Args:
            query: User query
            documents: Context documents

        Returns:
            Generated response with metadata
        """
        # TODO: Format context
        # TODO: Generate response
        # TODO: Extract citations
        # TODO: Add metadata
        pass

    async def query(
        self,
        query: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query pipeline.

        Args:
            query: User query
            user_id: Optional user ID

        Returns:
            Query results with response
        """
        # TODO: Process query
        # TODO: Retrieve documents
        # TODO: Rerank results
        # TODO: Generate response
        # TODO: Track metrics
        # TODO: Return comprehensive results
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics.

        Returns:
            Performance metrics
        """
        # TODO: Calculate latency stats
        # TODO: Get cache hit rates
        # TODO: Return metrics summary
        pass

    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check.

        Returns:
            Health status of components
        """
        # TODO: Check vector store
        # TODO: Check LLM availability
        # TODO: Check cache
        # TODO: Return status
        pass

# Test your implementation
config = {
    "retrieval_k": 10,
    "rerank_k": 5,
    "cache_ttl": 3600,
    "max_retries": 3
}

pipeline = ProductionRAGPipeline(config)

# Test pipeline (async)
import asyncio

async def test_pipeline():
    # Ingest documents
    doc_id = await pipeline.ingest_document(
        "Sample document content",
        {"source": "test", "date": "2024"}
    )
    print(f"Ingested document: {doc_id}")

    # Query pipeline
    result = await pipeline.query("test query")
    print(f"Query result: {result}")

    # Check health
    health = pipeline.health_check()
    print(f"Health status: {health}")

# Run test
# asyncio.run(test_pipeline())

# ================================
# Exercise 5: Implement Evaluation Metrics
# ================================
print("\n" + "=" * 50)
print("Exercise 5: Implement Evaluation Metrics")
print("=" * 50)

"""
Task: Build a comprehensive evaluation system for RAG pipelines.

Requirements:
1. Retrieval metrics (precision, recall, MRR, NDCG)
2. Generation quality metrics (faithfulness, relevance, coherence)
3. End-to-end metrics (latency, throughput, success rate)
4. A/B testing framework
5. Online evaluation with user feedback

Your implementation should:
- Support batch evaluation
- Generate detailed reports
- Compare different configurations
"""

class RAGEvaluator:
    """Evaluation system for RAG pipelines."""

    def __init__(self):
        """Initialize evaluator."""
        # TODO: Initialize components
        pass

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate precision@k.

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            k: Cutoff position

        Returns:
            Precision score
        """
        # TODO: Calculate precision at k
        pass

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Calculate recall@k.

        Args:
            retrieved: Retrieved document IDs
            relevant: Relevant document IDs
            k: Cutoff position

        Returns:
            Recall score
        """
        # TODO: Calculate recall at k
        pass

    def mean_reciprocal_rank(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_lists: List of retrieved results
            relevant_lists: List of relevant documents

        Returns:
            MRR score
        """
        # TODO: Calculate MRR across queries
        pass

    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate NDCG@k.

        Args:
            retrieved: Retrieved document IDs
            relevance_scores: Relevance scores for documents
            k: Cutoff position

        Returns:
            NDCG score
        """
        # TODO: Calculate DCG
        # TODO: Calculate IDCG
        # TODO: Return NDCG
        pass

    def evaluate_faithfulness(
        self,
        response: str,
        source_documents: List[str]
    ) -> float:
        """
        Evaluate response faithfulness to sources.

        Args:
            response: Generated response
            source_documents: Source documents

        Returns:
            Faithfulness score (0-1)
        """
        # TODO: Check claims in response
        # TODO: Verify against sources
        # TODO: Return faithfulness score
        pass

    def evaluate_relevance(
        self,
        query: str,
        response: str
    ) -> float:
        """
        Evaluate response relevance to query.

        Args:
            query: User query
            response: Generated response

        Returns:
            Relevance score (0-1)
        """
        # TODO: Assess query-response alignment
        # TODO: Use LLM or similarity metrics
        pass

    def run_ab_test(
        self,
        pipeline_a,
        pipeline_b,
        test_queries: List[str],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Run A/B test between pipelines.

        Args:
            pipeline_a: First pipeline
            pipeline_b: Second pipeline
            test_queries: Test queries
            metrics: Metrics to compare

        Returns:
            Comparison results
        """
        # TODO: Run both pipelines
        # TODO: Calculate metrics
        # TODO: Statistical significance
        # TODO: Return comparison
        pass

    def generate_report(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Generate evaluation report.

        Args:
            evaluation_results: Evaluation data

        Returns:
            Formatted report
        """
        # TODO: Format results
        # TODO: Add visualizations (text-based)
        # TODO: Include recommendations
        pass

# Test your implementation
evaluator = RAGEvaluator()

# Test retrieval metrics
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = ["doc1", "doc3", "doc6"]

precision = evaluator.precision_at_k(retrieved, relevant, k=3)
recall = evaluator.recall_at_k(retrieved, relevant, k=5)
print(f"Precision@3: {precision}")
print(f"Recall@5: {recall}")

# ================================
# Challenge: Complete Advanced RAG System
# ================================
print("\n" + "=" * 50)
print("Challenge: Complete Advanced RAG System")
print("=" * 50)

"""
Challenge: Build a complete advanced RAG system combining all techniques.

Requirements:
1. Hybrid retrieval with multiple strategies
2. Advanced query processing pipeline
3. Multi-stage reranking with diversity
4. Adaptive routing based on query type
5. Caching at multiple levels
6. Real-time performance monitoring
7. A/B testing capability
8. Fallback mechanisms
9. User personalization
10. Evaluation and metrics

Your system should:
- Be production-ready
- Handle high throughput
- Maintain low latency
- Provide detailed observability
- Support incremental improvements
"""

class AdvancedRAGSystem:
    """Complete advanced RAG system."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced RAG system.

        Args:
            config: System configuration
        """
        # TODO: Initialize all components
        # TODO: Set up monitoring
        # TODO: Initialize caches
        pass

    async def process_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a RAG request end-to-end.

        Args:
            query: User query
            user_id: Optional user ID
            session_id: Optional session ID

        Returns:
            Complete response with metadata
        """
        # TODO: Implement complete pipeline
        pass

    def optimize_performance(self):
        """Optimize system performance based on metrics."""
        # TODO: Analyze performance data
        # TODO: Adjust parameters
        # TODO: Update caching strategies
        pass

    def run_experiments(self, experiments: List[Dict[str, Any]]):
        """Run experiments to improve the system."""
        # TODO: Set up A/B tests
        # TODO: Collect results
        # TODO: Analyze outcomes
        # TODO: Deploy improvements
        pass

# Create your advanced system
advanced_config = {
    "retrieval": {
        "methods": ["sparse", "dense", "hybrid"],
        "fusion": "adaptive"
    },
    "reranking": {
        "strategies": ["cross_encoder", "mmr", "personalized"],
        "multi_stage": True
    },
    "caching": {
        "levels": ["query", "retrieval", "response"],
        "ttl": 3600
    },
    "monitoring": {
        "metrics": ["latency", "throughput", "quality"],
        "alerts": True
    }
}

# system = AdvancedRAGSystem(advanced_config)

print("\n" + "=" * 50)
print("Exercises Complete!")
print("Implement each class and test your solutions")
print("=" * 50)