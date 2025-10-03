"""
Hybrid retrieval combining BM25 (lexical) and dense (semantic) search.

This module implements production-grade retrieval strategies that combine
multiple search methods for better recall and precision.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "bm25", "dense", or "hybrid"


class BM25Retriever:
    """
    BM25 (Best Match 25) lexical retriever.

    Fast keyword-based search using probabilistic relevance framework.
    Good for exact term matching and proper nouns.
    """

    def __init__(self, documents: List[Dict[str, Any]], content_key: str = "content"):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of document dicts
            content_key: Key containing document text
        """
        self.documents = documents
        self.content_key = content_key

        # Extract and tokenize documents
        self.doc_texts = [doc[content_key] for doc in documents]
        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents by BM25 score.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc.get("id", str(idx)),
                content=doc[self.content_key],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                source="bm25"
            ))

        return results


class DenseRetriever:
    """
    Dense retriever using semantic embeddings.

    Uses cosine similarity between query and document embeddings.
    Good for semantic matching and paraphrases.
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        content_key: str = "content"
    ):
        """
        Initialize dense retriever.

        Args:
            documents: List of document dicts
            embeddings: Document embeddings (n_docs, embedding_dim)
            content_key: Key containing document text
        """
        self.documents = documents
        self.embeddings = embeddings
        self.content_key = content_key

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents by cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute cosine similarities
        similarities = np.dot(doc_norms, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc.get("id", str(idx)),
                content=doc[self.content_key],
                score=float(similarities[idx]),
                metadata=doc.get("metadata", {}),
                source="dense"
            ))

        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval.

    Implements multiple fusion strategies:
    - Weighted fusion (alpha * bm25 + (1-alpha) * dense)
    - Reciprocal Rank Fusion (RRF)
    - Score normalization

    Also supports metadata filtering.
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        content_key: str = "content"
    ):
        """
        Initialize hybrid retriever.

        Args:
            documents: List of document dicts
            embeddings: Optional pre-computed embeddings
            content_key: Key containing document text
        """
        self.documents = documents
        self.content_key = content_key

        # Initialize BM25
        self.bm25_retriever = BM25Retriever(documents, content_key)

        # Initialize dense retriever if embeddings provided
        self.dense_retriever = None
        if embeddings is not None:
            self.dense_retriever = DenseRetriever(
                documents, embeddings, content_key
            )

    def normalize_scores(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Normalize scores to [0, 1] range.

        Args:
            results: List of retrieval results

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for r in results:
                r.score = 1.0
            return results

        # Min-max normalization
        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)

        return results

    def weighted_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Fuse results using weighted combination.

        Score = alpha * bm25_score + (1-alpha) * dense_score

        Args:
            bm25_results: BM25 results
            dense_results: Dense results
            alpha: Weight for BM25 (0-1)

        Returns:
            Fused and sorted results
        """
        # Normalize scores
        bm25_results = self.normalize_scores(bm25_results)
        dense_results = self.normalize_scores(dense_results)

        # Build score dict
        scores = {}

        for result in bm25_results:
            scores[result.doc_id] = {
                "bm25": result.score,
                "dense": 0.0,
                "doc": result
            }

        for result in dense_results:
            if result.doc_id in scores:
                scores[result.doc_id]["dense"] = result.score
            else:
                scores[result.doc_id] = {
                    "bm25": 0.0,
                    "dense": result.score,
                    "doc": result
                }

        # Compute weighted scores
        fused = []
        for doc_id, data in scores.items():
            score = alpha * data["bm25"] + (1 - alpha) * data["dense"]
            result = data["doc"]
            result.score = score
            result.source = "hybrid"
            fused.append(result)

        # Sort by score descending
        fused.sort(key=lambda x: x.score, reverse=True)

        return fused

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank))

        More robust to score scale differences than weighted fusion.

        Args:
            bm25_results: BM25 results
            dense_results: Dense results
            k: RRF constant (default: 60, from paper)

        Returns:
            Fused and sorted results
        """
        # Build rank dicts
        scores = {}

        # Add BM25 ranks
        for rank, result in enumerate(bm25_results):
            scores[result.doc_id] = {
                "rrf_score": 1.0 / (k + rank + 1),
                "doc": result
            }

        # Add dense ranks
        for rank, result in enumerate(dense_results):
            if result.doc_id in scores:
                scores[result.doc_id]["rrf_score"] += 1.0 / (k + rank + 1)
            else:
                scores[result.doc_id] = {
                    "rrf_score": 1.0 / (k + rank + 1),
                    "doc": result
                }

        # Build fused results
        fused = []
        for doc_id, data in scores.items():
            result = data["doc"]
            result.score = data["rrf_score"]
            result.source = "hybrid"
            fused.append(result)

        # Sort by RRF score descending
        fused.sort(key=lambda x: x.score, reverse=True)

        return fused

    def filter_by_metadata(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """
        Filter results by metadata criteria.

        Args:
            results: Retrieval results
            filters: Metadata filters (key: value)

        Returns:
            Filtered results

        Example:
            filters = {"category": "science", "year": 2024}
        """
        filtered = []

        for result in results:
            match = True
            for key, value in filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break

            if match:
                filtered.append(result)

        return filtered

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
        fusion_method: str = "weighted",
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach.

        Args:
            query: Search query
            query_embedding: Optional query embedding for dense retrieval
            top_k: Number of results to return
            fusion_method: "weighted" or "rrf"
            alpha: Weight for weighted fusion (0-1)
            metadata_filters: Optional metadata filters

        Returns:
            List of top-k retrieval results
        """
        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

        # Dense retrieval (if available)
        if self.dense_retriever and query_embedding is not None:
            dense_results = self.dense_retriever.retrieve(
                query_embedding, top_k=top_k * 2
            )
        else:
            dense_results = []

        # Fusion
        if not dense_results:
            # Fall back to BM25 only
            fused = bm25_results
        elif fusion_method == "weighted":
            fused = self.weighted_fusion(bm25_results, dense_results, alpha)
        elif fusion_method == "rrf":
            fused = self.reciprocal_rank_fusion(bm25_results, dense_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Apply metadata filters
        if metadata_filters:
            fused = self.filter_by_metadata(fused, metadata_filters)

        # Return top-k
        return fused[:top_k]


# Example usage
if __name__ == "__main__":
    print("Hybrid Retrieval Examples")
    print("=" * 60)

    # Sample documents
    documents = [
        {
            "id": "1",
            "content": "Machine learning algorithms learn patterns from data",
            "metadata": {"category": "AI", "year": 2024}
        },
        {
            "id": "2",
            "content": "The weather forecast predicts rain tomorrow",
            "metadata": {"category": "weather", "year": 2024}
        },
        {
            "id": "3",
            "content": "Deep learning uses neural networks with many layers",
            "metadata": {"category": "AI", "year": 2024}
        },
        {
            "id": "4",
            "content": "Python is a popular programming language for data science",
            "metadata": {"category": "programming", "year": 2023}
        },
        {
            "id": "5",
            "content": "Natural language processing analyzes text and speech",
            "metadata": {"category": "AI", "year": 2024}
        }
    ]

    # Example 1: BM25 only
    print("\nExample 1: BM25 Retrieval")
    print("-" * 60)

    bm25 = BM25Retriever(documents)
    results = bm25.retrieve("machine learning patterns", top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result.score:.4f}] {result.content}")

    # Example 2: Hybrid with weighted fusion
    print("\n\nExample 2: Hybrid Retrieval (Weighted)")
    print("-" * 60)

    # Create dummy embeddings (normally from embedding model)
    embeddings = np.random.randn(len(documents), 128)

    hybrid = HybridRetriever(documents, embeddings)

    # Create dummy query embedding
    query_embedding = np.random.randn(128)

    results = hybrid.retrieve(
        "deep learning neural networks",
        query_embedding=query_embedding,
        top_k=3,
        fusion_method="weighted",
        alpha=0.5
    )

    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result.score:.4f}] {result.content}")

    # Example 3: With metadata filtering
    print("\n\nExample 3: Hybrid with Metadata Filtering")
    print("-" * 60)

    results = hybrid.retrieve(
        "AI and machine learning",
        query_embedding=query_embedding,
        top_k=5,
        metadata_filters={"category": "AI", "year": 2024}
    )

    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result.score:.4f}]")
        print(f"    {result.content}")
        print(f"    Category: {result.metadata.get('category')}")

    # Example 4: RRF fusion
    print("\n\nExample 4: Reciprocal Rank Fusion")
    print("-" * 60)

    results = hybrid.retrieve(
        "programming language data",
        query_embedding=query_embedding,
        top_k=3,
        fusion_method="rrf"
    )

    for i, result in enumerate(results, 1):
        print(f"{i}. [RRF Score: {result.score:.4f}] {result.content}")
