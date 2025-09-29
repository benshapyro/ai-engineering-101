"""
Module 10: RAG Basics - Retrieval Systems

Learn to build and optimize retrieval systems for RAG.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
from dataclasses import dataclass, field
import json
import heapq
from collections import defaultdict


# ===== Example 1: Basic Vector Store =====

def example_1_basic_vector_store():
    """Implement a basic vector store for similarity search."""
    print("Example 1: Basic Vector Store")
    print("=" * 50)

    class SimpleVectorStore:
        """Basic vector store implementation."""

        def __init__(self, dimension: int):
            self.dimension = dimension
            self.vectors = []
            self.documents = []
            self.metadata = []
            self.index_built = False

        def add_documents(self, documents: List[str], embeddings: np.ndarray,
                         metadata: Optional[List[Dict]] = None):
            """Add documents with their embeddings."""
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                self.documents.append(doc)
                self.vectors.append(emb)
                if metadata:
                    self.metadata.append(metadata[i])
                else:
                    self.metadata.append({})

            self.index_built = False

        def build_index(self):
            """Build index for faster search (convert to numpy array)."""
            if self.vectors:
                self.vectors = np.array(self.vectors)
                self.index_built = True

        def search(self, query_embedding: np.ndarray, k: int = 5,
                  filter_metadata: Optional[Dict] = None) -> List[Tuple[int, float, str]]:
            """Search for k nearest neighbors."""
            if not self.index_built:
                self.build_index()

            if len(self.vectors) == 0:
                return []

            # Calculate similarities
            similarities = np.dot(self.vectors, query_embedding)

            # Apply metadata filter if provided
            if filter_metadata:
                valid_indices = []
                for i, meta in enumerate(self.metadata):
                    if all(meta.get(key) == value for key, value in filter_metadata.items()):
                        valid_indices.append(i)

                if not valid_indices:
                    return []

                filtered_similarities = [(i, similarities[i]) for i in valid_indices]
                filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                top_k = filtered_similarities[:k]
            else:
                # Get top k indices
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                top_k = [(idx, similarities[idx]) for idx in top_k_indices]

            # Return results with documents
            results = []
            for idx, score in top_k:
                results.append((idx, float(score), self.documents[idx]))

            return results

        def get_statistics(self) -> Dict:
            """Get store statistics."""
            return {
                "document_count": len(self.documents),
                "dimension": self.dimension,
                "index_built": self.index_built,
                "metadata_keys": set().union(*[set(m.keys()) for m in self.metadata]) if self.metadata else set()
            }

    # Create vector store
    store = SimpleVectorStore(dimension=128)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision helps machines interpret visual information",
        "Reinforcement learning trains agents through rewards",
        "Data science combines statistics and programming",
        "Python is popular for machine learning development",
        "Cloud computing provides scalable infrastructure"
    ]

    # Create simple embeddings (in practice, use real embedding model)
    def create_embedding(text: str, dim: int = 128) -> np.ndarray:
        """Create pseudo-embedding for demonstration."""
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(dim)
        return embedding / np.linalg.norm(embedding)

    embeddings = np.array([create_embedding(doc) for doc in documents])

    # Add metadata
    metadata = [
        {"category": "ml", "difficulty": "basic"},
        {"category": "ml", "difficulty": "advanced"},
        {"category": "nlp", "difficulty": "basic"},
        {"category": "cv", "difficulty": "basic"},
        {"category": "ml", "difficulty": "advanced"},
        {"category": "data", "difficulty": "basic"},
        {"category": "programming", "difficulty": "basic"},
        {"category": "infrastructure", "difficulty": "basic"}
    ]

    # Add documents to store
    store.add_documents(documents, embeddings, metadata)
    store.build_index()

    # Test search
    print("Vector Store Statistics:")
    print(json.dumps(store.get_statistics(), indent=2, default=str))

    # Search examples
    queries = [
        "What is deep learning?",
        "How to process text data?",
        "Programming for AI"
    ]

    for query_text in queries:
        print(f"\n" + "-" * 30)
        print(f"Query: {query_text}")

        query_embedding = create_embedding(query_text)
        results = store.search(query_embedding, k=3)

        print("Top 3 results:")
        for idx, score, doc in results:
            print(f"  Score: {score:.3f} - {doc[:50]}...")

    # Test filtered search
    print(f"\n" + "-" * 30)
    print("Filtered Search (category='ml'):")

    query_embedding = create_embedding("neural network architectures")
    results = store.search(query_embedding, k=3, filter_metadata={"category": "ml"})

    for idx, score, doc in results:
        meta = store.metadata[idx]
        print(f"  Score: {score:.3f} [{meta['category']}] - {doc[:40]}...")


# ===== Example 2: Similarity Metrics =====

def example_2_similarity_metrics():
    """Compare different similarity metrics for retrieval."""
    print("\nExample 2: Similarity Metrics")
    print("=" * 50)

    class SimilarityMetrics:
        """Different similarity metrics for vector search."""

        @staticmethod
        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            """Cosine similarity (angle between vectors)."""
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        @staticmethod
        def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
            """Euclidean (L2) distance."""
            return -np.linalg.norm(a - b)  # Negative for consistency (higher is better)

        @staticmethod
        def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
            """Manhattan (L1) distance."""
            return -np.sum(np.abs(a - b))  # Negative for consistency

        @staticmethod
        def dot_product(a: np.ndarray, b: np.ndarray) -> float:
            """Dot product (for normalized vectors, same as cosine)."""
            return np.dot(a, b)

        @staticmethod
        def jaccard_similarity(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
            """Jaccard similarity for binary or thresholded vectors."""
            a_binary = a > threshold
            b_binary = b > threshold
            intersection = np.sum(a_binary & b_binary)
            union = np.sum(a_binary | b_binary)
            return intersection / union if union > 0 else 0

    class MultiMetricVectorStore:
        """Vector store supporting multiple similarity metrics."""

        def __init__(self, metric: str = "cosine"):
            self.metric = metric
            self.metrics = SimilarityMetrics()
            self.vectors = []
            self.documents = []

        def add_batch(self, documents: List[str], embeddings: np.ndarray):
            """Add documents and embeddings."""
            self.documents.extend(documents)
            self.vectors = embeddings

        def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, str]]:
            """Search using configured metric."""
            metric_func = getattr(self.metrics, f"{self.metric}_similarity", None)
            if self.metric in ["euclidean_distance", "manhattan_distance"]:
                metric_func = getattr(self.metrics, self.metric)

            if not metric_func:
                raise ValueError(f"Unknown metric: {self.metric}")

            scores = []
            for i, vec in enumerate(self.vectors):
                score = metric_func(query, vec)
                scores.append((score, self.documents[i]))

            # Sort by score (higher is better)
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[:k]

    # Test documents and embeddings
    documents = [
        "Vector databases store high-dimensional data",
        "Similarity search finds nearest neighbors",
        "Cosine similarity measures angle between vectors",
        "Euclidean distance measures straight-line distance",
        "Dot product is useful for normalized vectors"
    ]

    # Create embeddings
    def create_embeddings(texts: List[str], normalize: bool = True) -> np.ndarray:
        """Create embeddings for texts."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(64)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    embeddings = create_embeddings(documents)

    # Test different metrics
    metrics_to_test = ["cosine", "euclidean_distance", "manhattan_distance", "dot_product"]
    query_text = "How to measure vector similarity?"
    query_embedding = create_embeddings([query_text])[0]

    print(f"Query: {query_text}\n")

    for metric in metrics_to_test:
        print(f"{metric.replace('_', ' ').title()}:")
        print("-" * 30)

        store = MultiMetricVectorStore(metric=metric)
        store.add_batch(documents, embeddings)
        results = store.search(query_embedding, k=3)

        for score, doc in results:
            print(f"  Score: {score:8.4f} - {doc[:40]}...")
        print()

    # Compare metrics visually
    print("Metric Comparison Matrix:")
    print("-" * 30)

    metrics_obj = SimilarityMetrics()
    metric_funcs = {
        "Cosine": metrics_obj.cosine_similarity,
        "Euclidean": metrics_obj.euclidean_distance,
        "Manhattan": metrics_obj.manhattan_distance,
        "Dot Product": metrics_obj.dot_product
    }

    # Calculate all pairwise similarities for first 3 documents
    for i in range(min(3, len(embeddings))):
        print(f"\nDocument {i+1}: {documents[i][:30]}...")
        for metric_name, metric_func in metric_funcs.items():
            scores = []
            for j in range(min(3, len(embeddings))):
                if i != j:
                    score = metric_func(embeddings[i], embeddings[j])
                    scores.append(f"{score:.3f}")
            print(f"  {metric_name:10s}: {' | '.join(scores)}")


# ===== Example 3: Hybrid Search =====

def example_3_hybrid_search():
    """Implement hybrid search combining keyword and semantic search."""
    print("\nExample 3: Hybrid Search")
    print("=" * 50)

    class HybridSearchEngine:
        """Combines keyword (BM25-like) and semantic search."""

        def __init__(self, embedding_dim: int = 128):
            self.embedding_dim = embedding_dim
            self.documents = []
            self.embeddings = []
            self.inverted_index = defaultdict(list)  # term -> doc_ids
            self.doc_lengths = []
            self.avg_doc_length = 0
            self.idf_scores = {}

        def index_documents(self, documents: List[str], embeddings: np.ndarray):
            """Index documents for both keyword and semantic search."""
            self.documents = documents
            self.embeddings = embeddings

            # Build inverted index for keyword search
            doc_freqs = defaultdict(int)

            for doc_id, doc in enumerate(documents):
                terms = doc.lower().split()
                self.doc_lengths.append(len(terms))

                unique_terms = set(terms)
                for term in unique_terms:
                    self.inverted_index[term].append(doc_id)
                    doc_freqs[term] += 1

            self.avg_doc_length = np.mean(self.doc_lengths)

            # Calculate IDF scores
            num_docs = len(documents)
            for term, freq in doc_freqs.items():
                self.idf_scores[term] = np.log((num_docs - freq + 0.5) / (freq + 0.5))

        def keyword_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
            """BM25-like keyword search."""
            query_terms = query.lower().split()
            doc_scores = defaultdict(float)

            k1 = 1.2  # BM25 parameter
            b = 0.75  # BM25 parameter

            for term in query_terms:
                if term not in self.inverted_index:
                    continue

                idf = self.idf_scores.get(term, 0)

                for doc_id in self.inverted_index[term]:
                    doc = self.documents[doc_id]
                    tf = doc.lower().count(term)
                    doc_len = self.doc_lengths[doc_id]

                    # BM25 scoring
                    score = idf * (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * doc_len / self.avg_doc_length)
                    )
                    doc_scores[doc_id] += score

            # Sort by score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_docs[:k]

        def semantic_search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
            """Semantic similarity search."""
            similarities = np.dot(self.embeddings, query_embedding)
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                results.append((idx, float(similarities[idx])))

            return results

        def hybrid_search(self, query: str, query_embedding: np.ndarray,
                         k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float, Dict]]:
            """
            Hybrid search combining keyword and semantic scores.
            alpha: weight for semantic search (1-alpha for keyword search)
            """
            # Get keyword search results
            keyword_results = self.keyword_search(query, k=k*2)
            keyword_scores = {doc_id: score for doc_id, score in keyword_results}

            # Normalize keyword scores
            if keyword_scores:
                max_keyword = max(keyword_scores.values())
                if max_keyword > 0:
                    keyword_scores = {k: v/max_keyword for k, v in keyword_scores.items()}

            # Get semantic search results
            semantic_results = self.semantic_search(query_embedding, k=k*2)
            semantic_scores = {doc_id: score for doc_id, score in semantic_results}

            # Normalize semantic scores (already normalized if using cosine)
            if semantic_scores:
                max_semantic = max(semantic_scores.values())
                if max_semantic > 0:
                    semantic_scores = {k: v/max_semantic for k, v in semantic_scores.items()}

            # Combine scores
            all_doc_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
            hybrid_scores = []

            for doc_id in all_doc_ids:
                keyword_score = keyword_scores.get(doc_id, 0)
                semantic_score = semantic_scores.get(doc_id, 0)
                hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score

                hybrid_scores.append((
                    self.documents[doc_id],
                    hybrid_score,
                    {
                        "keyword_score": keyword_score,
                        "semantic_score": semantic_score,
                        "doc_id": doc_id
                    }
                ))

            # Sort by hybrid score
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            return hybrid_scores[:k]

    # Create search engine
    search_engine = HybridSearchEngine()

    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity",
        "Machine learning models can be trained using Python libraries",
        "Deep learning is a subset of machine learning using neural networks",
        "Natural language processing helps computers understand human language",
        "Data science combines programming, statistics, and domain knowledge",
        "JavaScript is primarily used for web development and frontend programming",
        "Cloud computing provides on-demand computing resources over the internet",
        "Artificial intelligence aims to create intelligent machines"
    ]

    # Create embeddings
    def create_embedding(text: str, dim: int = 128) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(dim)
        return emb / np.linalg.norm(emb)

    embeddings = np.array([create_embedding(doc) for doc in documents])

    # Index documents
    search_engine.index_documents(documents, embeddings)

    # Test searches
    test_queries = [
        "Python programming for machine learning",
        "deep neural networks",
        "web development languages"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        query_embedding = create_embedding(query)

        # Test different search methods
        methods = [
            ("Keyword Only", lambda: [
                (search_engine.documents[doc_id], score, {"method": "keyword"})
                for doc_id, score in search_engine.keyword_search(query, k=3)
            ]),
            ("Semantic Only", lambda: [
                (search_engine.documents[doc_id], score, {"method": "semantic"})
                for doc_id, score in search_engine.semantic_search(query_embedding, k=3)
            ]),
            ("Hybrid (α=0.5)", lambda: search_engine.hybrid_search(query, query_embedding, k=3, alpha=0.5)),
            ("Hybrid (α=0.7)", lambda: search_engine.hybrid_search(query, query_embedding, k=3, alpha=0.7))
        ]

        for method_name, search_func in methods:
            print(f"\n{method_name}:")
            results = search_func()

            for i, (doc, score, metadata) in enumerate(results[:3], 1):
                print(f"  {i}. Score: {score:.3f}")
                print(f"     {doc[:60]}...")
                if "keyword_score" in metadata and "semantic_score" in metadata:
                    print(f"     (KW: {metadata['keyword_score']:.2f}, "
                          f"Sem: {metadata['semantic_score']:.2f})")


# ===== Example 4: Metadata Filtering =====

def example_4_metadata_filtering():
    """Implement advanced metadata filtering for retrieval."""
    print("\nExample 4: Metadata Filtering")
    print("=" * 50)

    @dataclass
    class Document:
        """Document with metadata."""
        id: str
        text: str
        embedding: np.ndarray
        metadata: Dict[str, Any]

    class FilterableVectorStore:
        """Vector store with advanced filtering capabilities."""

        def __init__(self):
            self.documents: List[Document] = []
            self.metadata_index = defaultdict(lambda: defaultdict(set))

        def add_document(self, doc: Document):
            """Add document and index its metadata."""
            doc_idx = len(self.documents)
            self.documents.append(doc)

            # Index metadata for fast filtering
            for key, value in doc.metadata.items():
                self.metadata_index[key][value].add(doc_idx)

        def filter_documents(self, filters: Dict[str, Any]) -> List[int]:
            """Get document indices matching all filters."""
            if not filters:
                return list(range(len(self.documents)))

            matching_sets = []
            for key, value in filters.items():
                if key in self.metadata_index:
                    if isinstance(value, list):
                        # OR condition for list values
                        matches = set()
                        for v in value:
                            matches.update(self.metadata_index[key].get(v, set()))
                        matching_sets.append(matches)
                    else:
                        # Exact match
                        matching_sets.append(self.metadata_index[key].get(value, set()))
                else:
                    # No documents match this filter
                    return []

            # Intersection of all filter matches (AND condition)
            if matching_sets:
                result = matching_sets[0]
                for s in matching_sets[1:]:
                    result = result.intersection(s)
                return list(result)

            return []

        def search(self, query_embedding: np.ndarray, k: int = 5,
                  filters: Optional[Dict] = None,
                  filter_mode: str = "pre") -> List[Tuple[Document, float]]:
            """
            Search with metadata filtering.
            filter_mode: 'pre' (filter then search) or 'post' (search then filter)
            """
            if filter_mode == "pre":
                # Pre-filtering: filter first, then search
                if filters:
                    valid_indices = self.filter_documents(filters)
                    if not valid_indices:
                        return []
                else:
                    valid_indices = list(range(len(self.documents)))

                # Calculate similarities only for filtered documents
                similarities = []
                for idx in valid_indices:
                    doc = self.documents[idx]
                    sim = np.dot(doc.embedding, query_embedding)
                    similarities.append((doc, sim))

                # Sort and return top k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:k]

            else:  # post-filtering
                # Post-filtering: search first, then filter
                all_similarities = []
                for doc in self.documents:
                    sim = np.dot(doc.embedding, query_embedding)
                    all_similarities.append((doc, sim))

                # Sort by similarity
                all_similarities.sort(key=lambda x: x[1], reverse=True)

                # Filter results
                if filters:
                    filtered = []
                    for doc, sim in all_similarities:
                        if self._matches_filters(doc, filters):
                            filtered.append((doc, sim))
                            if len(filtered) >= k:
                                break
                    return filtered
                else:
                    return all_similarities[:k]

        def _matches_filters(self, doc: Document, filters: Dict) -> bool:
            """Check if document matches all filters."""
            for key, value in filters.items():
                doc_value = doc.metadata.get(key)
                if isinstance(value, list):
                    if doc_value not in value:
                        return False
                else:
                    if doc_value != value:
                        return False
            return True

        def get_metadata_statistics(self) -> Dict:
            """Get statistics about metadata in the store."""
            stats = {}
            for key in self.metadata_index:
                values = self.metadata_index[key]
                stats[key] = {
                    "unique_values": len(values),
                    "distribution": {
                        str(v): len(docs) for v, docs in values.items()
                    }
                }
            return stats

    # Create store
    store = FilterableVectorStore()

    # Add sample documents
    sample_docs = [
        ("doc1", "Introduction to Python programming", {"language": "python", "level": "beginner", "type": "tutorial", "year": 2023}),
        ("doc2", "Advanced Python techniques", {"language": "python", "level": "advanced", "type": "tutorial", "year": 2024}),
        ("doc3", "Machine learning with Python", {"language": "python", "level": "intermediate", "type": "guide", "year": 2023}),
        ("doc4", "JavaScript fundamentals", {"language": "javascript", "level": "beginner", "type": "tutorial", "year": 2024}),
        ("doc5", "React.js advanced patterns", {"language": "javascript", "level": "advanced", "type": "guide", "year": 2024}),
        ("doc6", "Data science best practices", {"language": "python", "level": "intermediate", "type": "guide", "year": 2023}),
        ("doc7", "Web development with Node.js", {"language": "javascript", "level": "intermediate", "type": "tutorial", "year": 2024}),
        ("doc8", "Deep learning fundamentals", {"language": "python", "level": "advanced", "type": "guide", "year": 2024})
    ]

    for doc_id, text, metadata in sample_docs:
        embedding = create_embedding(text)
        doc = Document(id=doc_id, text=text, embedding=embedding, metadata=metadata)
        store.add_document(doc)

    # Show metadata statistics
    print("Metadata Statistics:")
    stats = store.get_metadata_statistics()
    for key, info in stats.items():
        print(f"\n{key}:")
        print(f"  Unique values: {info['unique_values']}")
        for value, count in info['distribution'].items():
            print(f"    {value}: {count} documents")

    # Test different filtering scenarios
    test_cases = [
        ("Python tutorials for beginners", {"language": "python", "level": "beginner"}),
        ("Advanced content from 2024", {"level": "advanced", "year": 2024}),
        ("JavaScript guides", {"language": "javascript", "type": "guide"}),
        ("Any Python or JavaScript content", {"language": ["python", "javascript"]})
    ]

    for query_text, filters in test_cases:
        print(f"\n" + "-" * 50)
        print(f"Query: {query_text}")
        print(f"Filters: {filters}")

        query_embedding = create_embedding(query_text)

        # Compare pre-filtering vs post-filtering
        for mode in ["pre", "post"]:
            print(f"\n{mode.capitalize()}-filtering:")
            results = store.search(query_embedding, k=3, filters=filters, filter_mode=mode)

            if results:
                for doc, score in results:
                    print(f"  {score:.3f}: {doc.text[:40]}...")
                    print(f"         [{doc.metadata['language']}, {doc.metadata['level']}]")
            else:
                print("  No results found")


# ===== Example 5: Re-ranking Strategies =====

def example_5_reranking():
    """Implement re-ranking strategies to improve retrieval quality."""
    print("\nExample 5: Re-ranking Strategies")
    print("=" * 50)

    class ReRanker:
        """Various re-ranking strategies."""

        @staticmethod
        def diversity_rerank(results: List[Tuple[str, float]], lambda_param: float = 0.5) -> List[Tuple[str, float]]:
            """MMR (Maximal Marginal Relevance) re-ranking for diversity."""
            if not results:
                return []

            # Extract documents and scores
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]

            # Selected documents
            selected = []
            selected_indices = set()
            remaining_indices = set(range(len(results)))

            # Select first document (highest score)
            first_idx = 0
            selected.append((docs[first_idx], scores[first_idx]))
            selected_indices.add(first_idx)
            remaining_indices.remove(first_idx)

            # Iteratively select diverse documents
            while remaining_indices and len(selected) < len(results):
                best_score = -float('inf')
                best_idx = None

                for idx in remaining_indices:
                    # Calculate relevance score
                    relevance = scores[idx]

                    # Calculate maximum similarity to selected documents
                    max_sim = 0
                    for sel_idx in selected_indices:
                        # Simple text similarity (character overlap)
                        sim = len(set(docs[idx]) & set(docs[sel_idx])) / max(len(docs[idx]), len(docs[sel_idx]))
                        max_sim = max(max_sim, sim)

                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected.append((docs[best_idx], best_score))
                    selected_indices.add(best_idx)
                    remaining_indices.remove(best_idx)

            return selected

        @staticmethod
        def relevance_feedback_rerank(results: List[Tuple[str, float]],
                                     positive_feedback: List[str],
                                     negative_feedback: List[str],
                                     alpha: float = 1.0,
                                     beta: float = 0.8,
                                     gamma: float = 0.1) -> List[Tuple[str, float]]:
            """Rocchio algorithm for relevance feedback."""
            reranked = []

            for doc, original_score in results:
                # Count term overlaps with feedback
                doc_terms = set(doc.lower().split())

                positive_overlap = sum(
                    len(doc_terms & set(pos.lower().split()))
                    for pos in positive_feedback
                ) / max(len(positive_feedback), 1)

                negative_overlap = sum(
                    len(doc_terms & set(neg.lower().split()))
                    for neg in negative_feedback
                ) / max(len(negative_feedback), 1)

                # Rocchio formula
                adjusted_score = (alpha * original_score +
                                beta * positive_overlap -
                                gamma * negative_overlap)

                reranked.append((doc, adjusted_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

        @staticmethod
        def cross_encoder_rerank(results: List[Tuple[str, float]], query: str) -> List[Tuple[str, float]]:
            """Simulate cross-encoder re-ranking (would use real model in practice)."""
            reranked = []

            for doc, initial_score in results:
                # Simulate cross-encoder score based on query-document interaction
                query_terms = set(query.lower().split())
                doc_terms = set(doc.lower().split())

                # Exact matches boost
                exact_matches = len(query_terms & doc_terms)

                # Partial matches
                partial_matches = sum(
                    1 for q_term in query_terms
                    for d_term in doc_terms
                    if q_term in d_term or d_term in q_term
                ) / max(len(query_terms), 1)

                # Term proximity (simplified)
                if query_terms & doc_terms:
                    proximity_score = 1.0
                else:
                    proximity_score = 0.5

                # Combine scores
                cross_encoder_score = (
                    0.4 * initial_score +
                    0.3 * (exact_matches / max(len(query_terms), 1)) +
                    0.2 * partial_matches +
                    0.1 * proximity_score
                )

                reranked.append((doc, cross_encoder_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked

    # Test documents
    initial_results = [
        ("Python is a versatile programming language for data science", 0.85),
        ("Machine learning with Python is powerful", 0.82),
        ("Python web development with Django framework", 0.80),
        ("Data analysis using Python pandas library", 0.78),
        ("Python for beginners: getting started guide", 0.75),
        ("Advanced Python programming techniques", 0.73),
        ("Python vs JavaScript comparison", 0.70),
        ("Scientific computing with Python NumPy", 0.68)
    ]

    reranker = ReRanker()

    # Test diversity re-ranking
    print("Diversity Re-ranking (MMR):")
    print("-" * 30)
    diverse_results = reranker.diversity_rerank(initial_results, lambda_param=0.7)

    print("Original ranking:")
    for i, (doc, score) in enumerate(initial_results[:5], 1):
        print(f"  {i}. [{score:.2f}] {doc[:50]}...")

    print("\nAfter diversity re-ranking:")
    for i, (doc, score) in enumerate(diverse_results[:5], 1):
        print(f"  {i}. [{score:.2f}] {doc[:50]}...")

    # Test relevance feedback
    print("\n" + "-" * 30)
    print("Relevance Feedback Re-ranking:")
    print("-" * 30)

    positive_feedback = [
        "data science machine learning",
        "pandas numpy analysis"
    ]
    negative_feedback = [
        "web development django",
        "javascript comparison"
    ]

    feedback_results = reranker.relevance_feedback_rerank(
        initial_results,
        positive_feedback,
        negative_feedback
    )

    print(f"Positive feedback: {positive_feedback}")
    print(f"Negative feedback: {negative_feedback}")
    print("\nAfter relevance feedback:")
    for i, (doc, score) in enumerate(feedback_results[:5], 1):
        print(f"  {i}. [{score:.2f}] {doc[:50]}...")

    # Test cross-encoder re-ranking
    print("\n" + "-" * 30)
    print("Cross-Encoder Re-ranking:")
    print("-" * 30)

    query = "Python data science machine learning"
    cross_encoder_results = reranker.cross_encoder_rerank(initial_results, query)

    print(f"Query: {query}")
    print("\nAfter cross-encoder re-ranking:")
    for i, (doc, score) in enumerate(cross_encoder_results[:5], 1):
        print(f"  {i}. [{score:.2f}] {doc[:50]}...")


# ===== Example 6: Query Expansion =====

def example_6_query_expansion():
    """Implement query expansion techniques for better retrieval."""
    print("\nExample 6: Query Expansion")
    print("=" * 50)

    class QueryExpander:
        """Various query expansion techniques."""

        def __init__(self):
            # Simulated synonym dictionary
            self.synonyms = {
                "python": ["py", "python3", "python language"],
                "programming": ["coding", "development", "software"],
                "machine": ["ML", "artificial", "AI"],
                "learning": ["training", "modeling", "algorithms"],
                "data": ["information", "dataset", "records"],
                "web": ["website", "online", "internet"],
                "database": ["DB", "datastore", "storage"]
            }

            # Simulated term associations
            self.associations = {
                "python": ["pandas", "numpy", "scikit-learn"],
                "machine learning": ["neural networks", "deep learning", "AI"],
                "web development": ["HTML", "CSS", "JavaScript"],
                "data science": ["statistics", "visualization", "analytics"]
            }

        def synonym_expansion(self, query: str) -> List[str]:
            """Expand query with synonyms."""
            terms = query.lower().split()
            expanded_queries = [query]

            for term in terms:
                if term in self.synonyms:
                    for synonym in self.synonyms[term]:
                        # Replace term with synonym
                        expanded = query.lower().replace(term, synonym)
                        if expanded not in expanded_queries:
                            expanded_queries.append(expanded)

            return expanded_queries

        def pseudo_relevance_feedback(self, query: str, top_docs: List[str], num_terms: int = 3) -> str:
            """Expand query using terms from top retrieved documents."""
            # Extract terms from top documents
            term_freq = defaultdict(int)
            query_terms = set(query.lower().split())

            for doc in top_docs:
                doc_terms = doc.lower().split()
                for term in doc_terms:
                    if term not in query_terms and len(term) > 3:  # Skip short words
                        term_freq[term] += 1

            # Select top frequent terms
            top_expansion_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:num_terms]
            expansion_terms = [term for term, _ in top_expansion_terms]

            # Create expanded query
            expanded_query = query + " " + " ".join(expansion_terms)
            return expanded_query

        def association_expansion(self, query: str) -> List[str]:
            """Expand query with associated terms."""
            expanded_queries = [query]

            # Check for known associations
            for key_phrase, associations in self.associations.items():
                if key_phrase in query.lower():
                    for assoc in associations:
                        expanded = f"{query} {assoc}"
                        expanded_queries.append(expanded)

            return expanded_queries

        def query_decomposition(self, query: str) -> List[str]:
            """Decompose complex query into sub-queries."""
            # Simple decomposition based on conjunctions
            sub_queries = []

            # Split by 'and', 'or'
            parts = query.replace(" and ", " AND ").replace(" or ", " OR ").split()
            current_query = []

            for word in parts:
                if word in ["AND", "OR"]:
                    if current_query:
                        sub_queries.append(" ".join(current_query))
                        current_query = []
                else:
                    current_query.append(word)

            if current_query:
                sub_queries.append(" ".join(current_query))

            # If no decomposition possible, return original
            if len(sub_queries) <= 1:
                # Try decomposition by topic detection
                if "python" in query.lower() and "javascript" in query.lower():
                    sub_queries = [
                        " ".join([w for w in query.split() if "javascript" not in w.lower()]),
                        " ".join([w for w in query.split() if "python" not in w.lower()])
                    ]

            return sub_queries if sub_queries else [query]

    # Create query expander
    expander = QueryExpander()

    # Test queries
    test_queries = [
        "python programming",
        "machine learning algorithms",
        "web development and data science"
    ]

    for original_query in test_queries:
        print(f"\nOriginal Query: '{original_query}'")
        print("=" * 50)

        # Synonym expansion
        print("\n1. Synonym Expansion:")
        synonym_expanded = expander.synonym_expansion(original_query)
        for i, expanded in enumerate(synonym_expanded, 1):
            print(f"   {i}. {expanded}")

        # Pseudo-relevance feedback
        print("\n2. Pseudo-Relevance Feedback:")
        # Simulate top retrieved documents
        top_docs = [
            "Python is great for data analysis and machine learning",
            "NumPy and pandas are essential Python libraries",
            "Scikit-learn provides machine learning algorithms"
        ]
        prf_expanded = expander.pseudo_relevance_feedback(original_query, top_docs)
        print(f"   Expanded: {prf_expanded}")

        # Association expansion
        print("\n3. Association Expansion:")
        assoc_expanded = expander.association_expansion(original_query)
        for i, expanded in enumerate(assoc_expanded, 1):
            print(f"   {i}. {expanded}")

        # Query decomposition
        print("\n4. Query Decomposition:")
        decomposed = expander.query_decomposition(original_query)
        for i, sub_query in enumerate(decomposed, 1):
            print(f"   Sub-query {i}: {sub_query}")


# ===== Example 7: Index Optimization =====

def example_7_index_optimization():
    """Optimize vector index for better performance."""
    print("\nExample 7: Index Optimization")
    print("=" * 50)

    class OptimizedVectorIndex:
        """Optimized vector index with various techniques."""

        def __init__(self, dimension: int, index_type: str = "flat"):
            self.dimension = dimension
            self.index_type = index_type
            self.vectors = None
            self.documents = []

            # Index structures
            self.flat_index = None
            self.clustered_index = None
            self.pq_index = None  # Product quantization

            # Statistics
            self.search_count = 0
            self.build_time = 0

        def add_documents(self, documents: List[str], embeddings: np.ndarray):
            """Add documents and build index."""
            start_time = time.time()

            self.documents = documents
            self.vectors = embeddings

            if self.index_type == "flat":
                self._build_flat_index()
            elif self.index_type == "clustered":
                self._build_clustered_index()
            elif self.index_type == "pq":
                self._build_pq_index()

            self.build_time = time.time() - start_time

        def _build_flat_index(self):
            """Build flat (brute-force) index."""
            self.flat_index = self.vectors

        def _build_clustered_index(self, n_clusters: int = 10):
            """Build clustered index using k-means."""
            from sklearn.cluster import MiniBatchKMeans

            # Cluster vectors
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.vectors)

            # Store clusters
            self.clustered_index = {
                "centroids": kmeans.cluster_centers_,
                "clusters": defaultdict(list)
            }

            for i, label in enumerate(cluster_labels):
                self.clustered_index["clusters"][label].append(i)

        def _build_pq_index(self, n_subvectors: int = 8):
            """Build product quantization index."""
            # Simplified PQ implementation
            subvector_dim = self.dimension // n_subvectors

            self.pq_index = {
                "n_subvectors": n_subvectors,
                "subvector_dim": subvector_dim,
                "codebooks": [],
                "codes": []
            }

            # Create codebooks for each subvector
            for i in range(n_subvectors):
                start_idx = i * subvector_dim
                end_idx = start_idx + subvector_dim

                subvectors = self.vectors[:, start_idx:end_idx]

                # Simple quantization (would use k-means in practice)
                unique_subvectors = np.unique(subvectors, axis=0)[:256]  # Max 256 codes
                self.pq_index["codebooks"].append(unique_subvectors)

                # Encode subvectors
                codes = []
                for sv in subvectors:
                    distances = np.sum((unique_subvectors - sv) ** 2, axis=1)
                    codes.append(np.argmin(distances))
                self.pq_index["codes"].append(codes)

        def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float, str]]:
            """Search for nearest neighbors."""
            self.search_count += 1

            if self.index_type == "flat":
                return self._search_flat(query, k)
            elif self.index_type == "clustered":
                return self._search_clustered(query, k)
            elif self.index_type == "pq":
                return self._search_pq(query, k)

        def _search_flat(self, query: np.ndarray, k: int) -> List[Tuple[int, float, str]]:
            """Flat search (brute-force)."""
            similarities = np.dot(self.flat_index, query)
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                results.append((idx, float(similarities[idx]), self.documents[idx]))

            return results

        def _search_clustered(self, query: np.ndarray, k: int,
                            n_probe: int = 2) -> List[Tuple[int, float, str]]:
            """Search using clustered index."""
            # Find nearest clusters
            centroid_distances = np.sum((self.clustered_index["centroids"] - query) ** 2, axis=1)
            nearest_clusters = np.argsort(centroid_distances)[:n_probe]

            # Search within nearest clusters
            candidates = []
            for cluster_id in nearest_clusters:
                for doc_idx in self.clustered_index["clusters"][cluster_id]:
                    similarity = np.dot(self.vectors[doc_idx], query)
                    candidates.append((doc_idx, similarity))

            # Sort and return top k
            candidates.sort(key=lambda x: x[1], reverse=True)
            results = []
            for idx, sim in candidates[:k]:
                results.append((idx, float(sim), self.documents[idx]))

            return results

        def _search_pq(self, query: np.ndarray, k: int) -> List[Tuple[int, float, str]]:
            """Search using product quantization."""
            # Compute distances to codebooks
            n_docs = len(self.documents)
            distances = np.zeros(n_docs)

            for i in range(self.pq_index["n_subvectors"]):
                start_idx = i * self.pq_index["subvector_dim"]
                end_idx = start_idx + self.pq_index["subvector_dim"]

                query_subvector = query[start_idx:end_idx]
                codebook = self.pq_index["codebooks"][i]

                # Precompute distances to codebook
                codebook_distances = np.sum((codebook - query_subvector) ** 2, axis=1)

                # Add distances for each document
                codes = self.pq_index["codes"][i]
                for doc_idx, code in enumerate(codes):
                    if code < len(codebook_distances):
                        distances[doc_idx] += codebook_distances[code]

            # Get top k (lower distance = higher similarity)
            top_k_indices = np.argsort(distances)[:k]

            results = []
            for idx in top_k_indices:
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distances[idx])
                results.append((idx, float(similarity), self.documents[idx]))

            return results

        def get_statistics(self) -> Dict:
            """Get index statistics."""
            stats = {
                "index_type": self.index_type,
                "dimension": self.dimension,
                "num_documents": len(self.documents),
                "build_time": self.build_time,
                "search_count": self.search_count
            }

            if self.index_type == "flat":
                stats["index_size_mb"] = self.vectors.nbytes / (1024 * 1024)
            elif self.index_type == "clustered":
                stats["num_clusters"] = len(self.clustered_index["centroids"])
            elif self.index_type == "pq":
                stats["n_subvectors"] = self.pq_index["n_subvectors"]
                # Calculate compression ratio
                original_size = self.vectors.nbytes
                compressed_size = sum(len(codes) for codes in self.pq_index["codes"])
                stats["compression_ratio"] = original_size / compressed_size

            return stats

    # Test documents
    documents = [f"Document about {topic} number {i}"
                for i in range(100)
                for topic in ["science", "technology", "art", "history"]]

    # Create embeddings
    dimension = 128
    embeddings = np.random.randn(len(documents), dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Test different index types
    index_types = ["flat", "clustered", "pq"]
    query = np.random.randn(dimension).astype(np.float32)
    query = query / np.linalg.norm(query)

    print("Index Comparison:")
    print("-" * 50)

    for index_type in index_types:
        print(f"\n{index_type.upper()} Index:")

        index = OptimizedVectorIndex(dimension, index_type)
        index.add_documents(documents, embeddings)

        # Measure search time
        start_time = time.time()
        results = index.search(query, k=5)
        search_time = time.time() - start_time

        # Display statistics
        stats = index.get_statistics()
        print(f"  Build time: {stats['build_time']:.4f}s")
        print(f"  Search time: {search_time:.4f}s")

        if "index_size_mb" in stats:
            print(f"  Index size: {stats['index_size_mb']:.2f} MB")
        if "compression_ratio" in stats:
            print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")

        # Show top results
        print(f"\n  Top 3 results:")
        for idx, sim, doc in results[:3]:
            print(f"    [{sim:.3f}] {doc[:40]}...")


# Helper function for creating embeddings
def create_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Create pseudo-embedding for demonstration."""
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 10: Retrieval Systems Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_basic_vector_store,
        2: example_2_similarity_metrics,
        3: example_3_hybrid_search,
        4: example_4_metadata_filtering,
        5: example_5_reranking,
        6: example_6_query_expansion,
        7: example_7_index_optimization
    }

    if args.all:
        for example in examples.values():
            example()
            print("\n" + "=" * 70 + "\n")
    elif args.example and args.example in examples:
        examples[args.example]()
    else:
        print("Module 10: Retrieval Systems - Examples")
        print("\nUsage:")
        print("  python retrieval_systems.py --example N  # Run example N")
        print("  python retrieval_systems.py --all         # Run all examples")
        print("\nAvailable examples:")
        print("  1: Basic Vector Store")
        print("  2: Similarity Metrics")
        print("  3: Hybrid Search")
        print("  4: Metadata Filtering")
        print("  5: Re-ranking Strategies")
        print("  6: Query Expansion")
        print("  7: Index Optimization")