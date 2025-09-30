"""
Module 11: Advanced RAG - Hybrid Retrieval Examples

This file demonstrates advanced retrieval strategies combining multiple
search methods for optimal performance and accuracy.

Author: Claude
Date: 2024
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Example 1: Basic Hybrid Retrieval
# ================================
print("=" * 50)
print("Example 1: Basic Hybrid Retrieval")
print("=" * 50)

@dataclass
class Document:
    """Document with multiple representations."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0

class HybridRetriever:
    """Combines dense and sparse retrieval methods."""

    def __init__(self):
        self.documents = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = None
        self.embeddings = []

    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Create document objects
        for i, content in enumerate(documents):
            doc = Document(
                id=f"doc_{i}",
                content=content
            )

            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )
            doc.embedding = np.array(response.data[0].embedding)

            self.documents.append(doc)
            self.embeddings.append(doc.embedding)

        # Build TF-IDF matrix
        texts = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        print(f"Indexed {len(documents)} documents")

    def sparse_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform sparse retrieval using TF-IDF."""
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k documents
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.bm25_score = similarities[idx]
            results.append(doc)

        return results

    def dense_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform dense retrieval using embeddings."""
        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate similarities
        similarities = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(sim)

        # Get top-k documents
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.semantic_score = similarities[idx]
            results.append(doc)

        return results

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
        """
        Perform hybrid search combining sparse and dense methods.

        Args:
            query: Search query
            k: Number of results
            alpha: Weight for dense search (1-alpha for sparse)
        """
        # Get results from both methods
        sparse_results = self.sparse_search(query, k=k*2)
        dense_results = self.dense_search(query, k=k*2)

        # Create score dictionary
        scores = {}

        # Add sparse scores
        for doc in sparse_results:
            scores[doc.id] = (1 - alpha) * doc.bm25_score

        # Add dense scores
        for doc in dense_results:
            if doc.id in scores:
                scores[doc.id] += alpha * doc.semantic_score
            else:
                scores[doc.id] = alpha * doc.semantic_score

        # Sort by hybrid score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Get documents
        results = []
        for doc_id, score in sorted_ids:
            doc = next(d for d in self.documents if d.id == doc_id)
            doc.hybrid_score = score
            results.append(doc)

        return results

# Test hybrid retrieval
retriever = HybridRetriever()

# Add sample documents
documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing allows computers to understand and generate human language.",
    "Computer vision enables machines to interpret and analyze visual information from images.",
    "Reinforcement learning trains agents through trial and error with reward signals."
]

retriever.add_documents(documents)

# Test different search methods
query = "How do neural networks learn from data?"

print(f"\nQuery: {query}")

# Sparse search
sparse_results = retriever.sparse_search(query, k=3)
print("\nSparse Search Results (TF-IDF):")
for i, doc in enumerate(sparse_results, 1):
    print(f"{i}. Score: {doc.bm25_score:.3f} - {doc.content[:60]}...")

# Dense search
dense_results = retriever.dense_search(query, k=3)
print("\nDense Search Results (Embeddings):")
for i, doc in enumerate(dense_results, 1):
    print(f"{i}. Score: {doc.semantic_score:.3f} - {doc.content[:60]}...")

# Hybrid search
hybrid_results = retriever.hybrid_search(query, k=3)
print("\nHybrid Search Results:")
for i, doc in enumerate(hybrid_results, 1):
    print(f"{i}. Score: {doc.hybrid_score:.3f} - {doc.content[:60]}...")

# ================================
# Example 2: Reciprocal Rank Fusion
# ================================
print("\n" + "=" * 50)
print("Example 2: Reciprocal Rank Fusion (RRF)")
print("=" * 50)

class RRFRetriever:
    """Retriever using Reciprocal Rank Fusion for combining results."""

    def __init__(self):
        self.documents = []
        self.retrievers = {}

    def add_retriever(self, name: str, retriever_func):
        """Add a retriever method."""
        self.retrievers[name] = retriever_func

    def reciprocal_rank_fusion(
        self,
        results_dict: Dict[str, List[Document]],
        k: int = 60
    ) -> List[Document]:
        """
        Combine rankings using Reciprocal Rank Fusion.

        RRF formula: RRF(d) = Σ 1/(k + rank(d))
        """
        rrf_scores = defaultdict(float)
        doc_map = {}

        # Calculate RRF scores
        for retriever_name, results in results_dict.items():
            for rank, doc in enumerate(results, 1):
                rrf_scores[doc.id] += 1.0 / (k + rank)
                doc_map[doc.id] = doc

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return documents
        final_results = []
        for doc_id, score in sorted_docs:
            doc = doc_map[doc_id]
            doc.hybrid_score = score
            final_results.append(doc)

        return final_results

    def multi_retriever_search(
        self,
        query: str,
        top_k: int = 5,
        k_param: int = 60
    ) -> List[Document]:
        """Search using multiple retrievers and fuse results."""
        results_dict = {}

        # Get results from each retriever
        for name, retriever_func in self.retrievers.items():
            results = retriever_func(query, k=top_k*2)
            results_dict[name] = results
            print(f"Retrieved {len(results)} documents from {name}")

        # Fuse results
        fused = self.reciprocal_rank_fusion(results_dict, k=k_param)

        return fused[:top_k]

# Create RRF retriever
rrf_retriever = RRFRetriever()

# Add different retrieval methods
def keyword_retriever(query: str, k: int) -> List[Document]:
    """Simple keyword-based retrieval."""
    results = []
    query_words = set(query.lower().split())

    for doc in retriever.documents:
        doc_words = set(doc.content.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            doc.bm25_score = overlap / len(query_words)
            results.append(doc)

    results.sort(key=lambda x: x.bm25_score, reverse=True)
    return results[:k]

# Add retrievers
rrf_retriever.add_retriever("sparse", retriever.sparse_search)
rrf_retriever.add_retriever("dense", retriever.dense_search)
rrf_retriever.add_retriever("keyword", keyword_retriever)

# Test RRF
rrf_results = rrf_retriever.multi_retriever_search(query, top_k=3)
print("\nRRF Fusion Results:")
for i, doc in enumerate(rrf_results, 1):
    print(f"{i}. RRF Score: {doc.hybrid_score:.4f} - {doc.content[:60]}...")

# ================================
# Example 3: Multi-Vector Retrieval
# ================================
print("\n" + "=" * 50)
print("Example 3: Multi-Vector Retrieval")
print("=" * 50)

class MultiVectorRetriever:
    """Use multiple representations per document."""

    def __init__(self):
        self.doc_embeddings = {}  # Full document embeddings
        self.chunk_embeddings = {}  # Chunk embeddings
        self.summary_embeddings = {}  # Summary embeddings
        self.documents = {}

    def index_document(self, doc_id: str, content: str):
        """Index document with multiple representations."""
        self.documents[doc_id] = content

        # Full document embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        self.doc_embeddings[doc_id] = np.array(response.data[0].embedding)

        # Chunk embeddings
        chunks = self._chunk_text(content, chunk_size=50)
        chunk_embs = []
        for chunk in chunks:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            chunk_embs.append(np.array(response.data[0].embedding))
        self.chunk_embeddings[doc_id] = chunk_embs

        # Summary embedding
        summary = self._generate_summary(content)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=summary
        )
        self.summary_embeddings[doc_id] = np.array(response.data[0].embedding)

        print(f"Indexed document {doc_id} with {len(chunks)} chunks")

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - 10):  # 10 word overlap
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _generate_summary(self, text: str) -> str:
        """Generate summary of text."""
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the text in one sentence."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=50
        )
        return response.choices[0].message.content

    def retrieve(
        self,
        query: str,
        k: int = 3,
        strategy: str = "ensemble"
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents using specified strategy.

        Strategies:
        - 'document': Use only document embeddings
        - 'chunk': Use only chunk embeddings
        - 'summary': Use only summary embeddings
        - 'ensemble': Combine all representations
        """
        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_emb = np.array(response.data[0].embedding)

        scores = {}

        if strategy in ["document", "ensemble"]:
            # Search document embeddings
            for doc_id, doc_emb in self.doc_embeddings.items():
                sim = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                scores[doc_id] = scores.get(doc_id, 0) + sim

        if strategy in ["chunk", "ensemble"]:
            # Search chunk embeddings
            for doc_id, chunk_embs in self.chunk_embeddings.items():
                max_sim = 0
                for chunk_emb in chunk_embs:
                    sim = np.dot(query_emb, chunk_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
                    )
                    max_sim = max(max_sim, sim)
                if strategy == "ensemble":
                    scores[doc_id] = scores.get(doc_id, 0) + max_sim
                else:
                    scores[doc_id] = max_sim

        if strategy in ["summary", "ensemble"]:
            # Search summary embeddings
            for doc_id, summary_emb in self.summary_embeddings.items():
                sim = np.dot(query_emb, summary_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(summary_emb)
                )
                if strategy == "ensemble":
                    scores[doc_id] = scores.get(doc_id, 0) + sim
                else:
                    scores[doc_id] = sim

        # Normalize ensemble scores
        if strategy == "ensemble":
            for doc_id in scores:
                scores[doc_id] /= 3  # Average of three methods

        # Sort and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [(doc_id, score) for doc_id, score in sorted_docs]

# Test multi-vector retrieval
mv_retriever = MultiVectorRetriever()

# Index documents
for i, doc in enumerate(documents):
    mv_retriever.index_document(f"doc_{i}", doc)

# Test different strategies
strategies = ["document", "chunk", "summary", "ensemble"]

for strategy in strategies:
    results = mv_retriever.retrieve(query, k=3, strategy=strategy)
    print(f"\n{strategy.capitalize()} Strategy Results:")
    for doc_id, score in results:
        content = mv_retriever.documents[doc_id]
        print(f"  {doc_id}: Score={score:.3f} - {content[:50]}...")

# ================================
# Example 4: Adaptive Retrieval
# ================================
print("\n" + "=" * 50)
print("Example 4: Adaptive Retrieval")
print("=" * 50)

class AdaptiveRetriever:
    """Adjusts retrieval strategy based on query characteristics."""

    def __init__(self):
        self.retrievers = {
            "short": self._short_query_retrieval,
            "long": self._long_query_retrieval,
            "technical": self._technical_retrieval,
            "question": self._question_retrieval,
            "keyword": self._keyword_retrieval
        }
        self.documents = []

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics."""
        words = query.split()

        analysis = {
            "length": len(words),
            "is_question": query.strip().endswith("?"),
            "has_technical_terms": self._has_technical_terms(query),
            "complexity": self._assess_complexity(query),
            "query_type": self._classify_query_type(query)
        }

        return analysis

    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical terms."""
        technical_terms = [
            "algorithm", "neural", "network", "machine learning",
            "deep learning", "embedding", "vector", "model",
            "training", "inference", "optimization"
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in technical_terms)

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        words = query.split()
        if len(words) < 5:
            return "simple"
        elif len(words) < 15:
            return "moderate"
        else:
            return "complex"

    def _classify_query_type(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()

        if query.endswith("?"):
            if "how" in query_lower:
                return "how_to"
            elif "what" in query_lower:
                return "definition"
            elif "why" in query_lower:
                return "explanation"
            else:
                return "question"
        elif len(query.split()) <= 3:
            return "keyword"
        else:
            return "statement"

    def select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select retrieval strategy based on analysis."""
        if analysis["length"] <= 3:
            return "keyword"
        elif analysis["is_question"]:
            return "question"
        elif analysis["has_technical_terms"]:
            return "technical"
        elif analysis["complexity"] == "simple":
            return "short"
        else:
            return "long"

    def _short_query_retrieval(self, query: str, k: int) -> List[Dict]:
        """Retrieval for short queries."""
        # Use keyword matching with boost
        results = []
        query_words = set(query.lower().split())

        for doc in self.documents:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / len(query_words) * 1.5  # Boost for exact matches
                results.append({
                    "document": doc,
                    "score": score,
                    "method": "short_query"
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _long_query_retrieval(self, query: str, k: int) -> List[Dict]:
        """Retrieval for long queries."""
        # Use semantic search with context
        return self._semantic_retrieval(query, k, "long_query")

    def _technical_retrieval(self, query: str, k: int) -> List[Dict]:
        """Retrieval for technical queries."""
        # Emphasize technical terms
        return self._semantic_retrieval(query, k, "technical")

    def _question_retrieval(self, query: str, k: int) -> List[Dict]:
        """Retrieval for questions."""
        # Focus on finding answers
        return self._semantic_retrieval(query, k, "question")

    def _keyword_retrieval(self, query: str, k: int) -> List[Dict]:
        """Simple keyword retrieval."""
        results = []
        query_lower = query.lower()

        for doc in self.documents:
            if query_lower in doc["content"].lower():
                results.append({
                    "document": doc,
                    "score": 1.0,
                    "method": "keyword"
                })

        return results[:k]

    def _semantic_retrieval(self, query: str, k: int, method: str) -> List[Dict]:
        """Generic semantic retrieval."""
        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_emb = np.array(response.data[0].embedding)

        results = []
        for doc in self.documents:
            sim = np.dot(query_emb, doc["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc["embedding"])
            )
            results.append({
                "document": doc,
                "score": sim,
                "method": method
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Adaptive retrieval based on query analysis."""
        # Analyze query
        analysis = self.analyze_query(query)
        print(f"Query Analysis: {analysis}")

        # Select strategy
        strategy = self.select_strategy(analysis)
        print(f"Selected Strategy: {strategy}")

        # Execute retrieval
        retriever = self.retrievers[strategy]
        results = retriever(query, k)

        return results

    def add_documents(self, documents: List[str]):
        """Add documents to retriever."""
        for i, content in enumerate(documents):
            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )

            self.documents.append({
                "id": f"doc_{i}",
                "content": content,
                "embedding": np.array(response.data[0].embedding)
            })

# Test adaptive retrieval
adaptive_retriever = AdaptiveRetriever()
adaptive_retriever.add_documents(documents)

# Test different query types
test_queries = [
    "neural networks",  # Short keyword query
    "What is machine learning?",  # Question
    "How do neural networks process information through multiple layers?",  # Technical question
    "learning",  # Single keyword
    "Explain the difference between supervised and unsupervised learning approaches in machine learning",  # Long complex
]

for test_query in test_queries:
    print(f"\n{'='*30}")
    print(f"Query: {test_query}")
    results = adaptive_retriever.retrieve(test_query, k=2)
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Method: {result['method']}, Score: {result['score']:.3f}")
        print(f"   {result['document']['content'][:60]}...")

# ================================
# Example 5: Ensemble Retrieval
# ================================
print("\n" + "=" * 50)
print("Example 5: Ensemble Retrieval")
print("=" * 50)

class EnsembleRetriever:
    """Combine multiple retrieval methods with voting."""

    def __init__(self):
        self.methods = []
        self.weights = []

    def add_method(self, method, weight: float = 1.0):
        """Add a retrieval method with weight."""
        self.methods.append(method)
        self.weights.append(weight)

    def retrieve_with_voting(
        self,
        query: str,
        k: int = 5,
        voting: str = "weighted"
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using ensemble voting.

        Voting strategies:
        - 'weighted': Weight scores by method weights
        - 'rank': Use rank-based voting
        - 'borda': Borda count method
        """
        all_results = []

        # Get results from all methods
        for method, weight in zip(self.methods, self.weights):
            results = method(query, k=k*2)
            all_results.append((results, weight))

        if voting == "weighted":
            return self._weighted_voting(all_results, k)
        elif voting == "rank":
            return self._rank_voting(all_results, k)
        elif voting == "borda":
            return self._borda_voting(all_results, k)
        else:
            raise ValueError(f"Unknown voting method: {voting}")

    def _weighted_voting(
        self,
        all_results: List[Tuple[List, float]],
        k: int
    ) -> List[Tuple[str, float]]:
        """Weighted score aggregation."""
        scores = defaultdict(float)

        for results, weight in all_results:
            for doc in results:
                doc_id = doc.id if hasattr(doc, 'id') else str(doc)
                score = doc.hybrid_score if hasattr(doc, 'hybrid_score') else 1.0
                scores[doc_id] += score * weight

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

    def _rank_voting(
        self,
        all_results: List[Tuple[List, float]],
        k: int
    ) -> List[Tuple[str, float]]:
        """Rank-based voting."""
        rank_scores = defaultdict(float)

        for results, weight in all_results:
            for rank, doc in enumerate(results):
                doc_id = doc.id if hasattr(doc, 'id') else str(doc)
                rank_scores[doc_id] += weight / (rank + 1)

        sorted_docs = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

    def _borda_voting(
        self,
        all_results: List[Tuple[List, float]],
        k: int
    ) -> List[Tuple[str, float]]:
        """Borda count voting."""
        borda_scores = defaultdict(float)

        for results, weight in all_results:
            n = len(results)
            for rank, doc in enumerate(results):
                doc_id = doc.id if hasattr(doc, 'id') else str(doc)
                borda_scores[doc_id] += (n - rank) * weight

        sorted_docs = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

# Create ensemble retriever
ensemble = EnsembleRetriever()

# Add different methods with weights
ensemble.add_method(lambda q, k: retriever.sparse_search(q, k), weight=0.3)
ensemble.add_method(lambda q, k: retriever.dense_search(q, k), weight=0.5)
ensemble.add_method(lambda q, k: keyword_retriever(q, k), weight=0.2)

# Test different voting methods
voting_methods = ["weighted", "rank", "borda"]

for voting_method in voting_methods:
    results = ensemble.retrieve_with_voting(query, k=3, voting=voting_method)
    print(f"\n{voting_method.capitalize()} Voting Results:")
    for doc_id, score in results:
        print(f"  {doc_id}: Score={score:.3f}")

# ================================
# Example 6: Performance Comparison
# ================================
print("\n" + "=" * 50)
print("Example 6: Performance Comparison")
print("=" * 50)

class RetrievalBenchmark:
    """Benchmark different retrieval methods."""

    def __init__(self):
        self.methods = {}
        self.results = defaultdict(dict)

    def add_method(self, name: str, method):
        """Add a retrieval method to benchmark."""
        self.methods[name] = method

    def benchmark(self, queries: List[str], k: int = 5):
        """Benchmark all methods on queries."""
        for query in queries:
            print(f"\nBenchmarking query: {query[:50]}...")

            for name, method in self.methods.items():
                start_time = time.time()

                try:
                    results = method(query, k)
                    elapsed = time.time() - start_time

                    self.results[query][name] = {
                        "results": results,
                        "time": elapsed,
                        "success": True
                    }
                except Exception as e:
                    self.results[query][name] = {
                        "results": [],
                        "time": 0,
                        "success": False,
                        "error": str(e)
                    }

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("Benchmark Summary")
        print("=" * 50)

        # Calculate average times
        method_times = defaultdict(list)

        for query, methods in self.results.items():
            for method_name, result in methods.items():
                if result["success"]:
                    method_times[method_name].append(result["time"])

        print("\nAverage Retrieval Times:")
        for method_name, times in method_times.items():
            avg_time = np.mean(times)
            print(f"  {method_name}: {avg_time:.4f}s")

        # Print success rates
        print("\nSuccess Rates:")
        for method_name in self.methods:
            total = len(self.results)
            successful = sum(
                1 for q in self.results.values()
                if q[method_name]["success"]
            )
            print(f"  {method_name}: {successful}/{total} ({successful/total*100:.1f}%)")

# Create benchmark
benchmark = RetrievalBenchmark()

# Add methods to benchmark
benchmark.add_method("sparse", retriever.sparse_search)
benchmark.add_method("dense", retriever.dense_search)
benchmark.add_method("hybrid", retriever.hybrid_search)
benchmark.add_method("rrf", lambda q, k: rrf_retriever.multi_retriever_search(q, k))

# Run benchmark
test_queries_benchmark = [
    "machine learning",
    "How do neural networks work?",
    "computer vision applications"
]

benchmark.benchmark(test_queries_benchmark, k=3)
benchmark.print_summary()

# ================================
# Example 7: Production Hybrid Search
# ================================
print("\n" + "=" * 50)
print("Example 7: Production Hybrid Search")
print("=" * 50)

class ProductionHybridSearch:
    """Production-ready hybrid search with caching and optimization."""

    def __init__(self, cache_size: int = 100):
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.documents = []
        self.index_built = False

    async def async_search(
        self,
        query: str,
        k: int = 5,
        use_cache: bool = True
    ) -> List[Dict]:
        """Asynchronous search with caching."""
        # Check cache
        cache_key = f"{query}_{k}"
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        # Perform parallel searches
        tasks = [
            self._async_sparse_search(query, k),
            self._async_dense_search(query, k),
            self._async_rerank(query, k)
        ]

        results = await asyncio.gather(*tasks)

        # Combine results
        combined = self._combine_results(results, k)

        # Update cache
        if use_cache:
            self._update_cache(cache_key, combined)

        return combined

    async def _async_sparse_search(self, query: str, k: int) -> List[Dict]:
        """Async sparse search."""
        await asyncio.sleep(0.01)  # Simulate I/O
        # Actual sparse search implementation
        return []

    async def _async_dense_search(self, query: str, k: int) -> List[Dict]:
        """Async dense search."""
        await asyncio.sleep(0.02)  # Simulate I/O
        # Actual dense search implementation
        return []

    async def _async_rerank(self, query: str, k: int) -> List[Dict]:
        """Async reranking."""
        await asyncio.sleep(0.015)  # Simulate I/O
        # Actual reranking implementation
        return []

    def _combine_results(self, results_list: List[List[Dict]], k: int) -> List[Dict]:
        """Combine results from multiple searches."""
        # Simple combination logic
        combined = []
        seen = set()

        for results in results_list:
            for doc in results:
                doc_id = doc.get("id", str(doc))
                if doc_id not in seen:
                    combined.append(doc)
                    seen.add(doc_id)
                    if len(combined) >= k:
                        return combined

        return combined

    def _update_cache(self, key: str, value: List[Dict]):
        """Update cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for demo)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "total_documents": len(self.documents)
        }

# Test production search
production_search = ProductionHybridSearch()

# Simulate searches
async def test_production_search():
    """Test production search system."""
    queries = [
        "machine learning basics",
        "neural network architecture",
        "machine learning basics",  # Duplicate to test cache
        "deep learning applications"
    ]

    for query in queries:
        results = await production_search.async_search(query, k=3)
        print(f"Query: {query} - Results: {len(results)}")

    # Print statistics
    stats = production_search.get_stats()
    print(f"\nSearch Statistics:")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Misses: {stats['cache_misses']}")

# Run async test
asyncio.run(test_production_search())

print("\n" + "=" * 50)
print("Hybrid Retrieval Examples Complete!")
print("=" * 50)