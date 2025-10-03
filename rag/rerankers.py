"""
Reranking utilities for RAG systems.

This module provides production-grade reranking using HuggingFace cross-encoders,
replacing LLM-based scoring with more efficient and accurate neural reranking.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Production-grade reranker using HuggingFace cross-encoder models.

    Cross-encoders provide better accuracy than bi-encoders for reranking
    by jointly encoding query and document, at the cost of slower inference.
    This is acceptable for reranking since we only process top-k candidates.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, lower quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, best quality)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)

        Example:
            reranker = CrossEncoderReranker()
            # Or use a faster model:
            reranker = CrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        """
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        return_scores: bool = False
    ) -> List[int] | List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Return only top k results (None = all)
            return_scores: If True, return (index, score) tuples

        Returns:
            List of document indices sorted by relevance, or
            List of (index, score) tuples if return_scores=True

        Example:
            docs = [
                "Python is a programming language",
                "The weather is nice today",
                "Python programming tutorials"
            ]
            indices = reranker.rerank("python coding", docs, top_k=2)
            # Returns: [2, 0] (most relevant documents)
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Create (index, score) pairs and sort by score descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply top_k filter
        if top_k is not None:
            ranked = ranked[:top_k]

        if return_scores:
            return ranked
        else:
            return [idx for idx, _ in ranked]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents with metadata, preserving all fields.

        Args:
            query: Search query
            documents: List of document dictionaries
            text_key: Key containing document text
            top_k: Return only top k results

        Returns:
            Reranked documents with added 'rerank_score' field

        Example:
            docs = [
                {"id": 1, "text": "Python programming", "source": "wiki"},
                {"id": 2, "text": "Weather forecast", "source": "news"},
                {"id": 3, "text": "Python tutorials", "source": "blog"}
            ]
            ranked = reranker.rerank_with_metadata("python", docs, top_k=2)
            # Returns top 2 documents with rerank_score added
        """
        if not documents:
            return []

        # Extract texts
        texts = [doc[text_key] for doc in documents]

        # Get ranked indices with scores
        ranked = self.rerank(query, texts, top_k=None, return_scores=True)

        # Build result with metadata
        results = []
        for idx, score in ranked[:top_k] if top_k else ranked:
            doc = documents[idx].copy()
            doc['rerank_score'] = float(score)
            results.append(doc)

        return results

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: int = None
    ) -> List[List[int]]:
        """
        Rerank multiple query-document sets efficiently.

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Return only top k per query

        Returns:
            List of ranked indices lists (one per query)

        Example:
            queries = ["python", "weather"]
            docs_list = [
                ["Python code", "Java code", "Python tutorial"],
                ["Sunny weather", "Python programming", "Rain forecast"]
            ]
            results = reranker.batch_rerank(queries, docs_list, top_k=2)
        """
        results = []
        for query, documents in zip(queries, documents_list):
            ranked = self.rerank(query, documents, top_k=top_k)
            results.append(ranked)
        return results

    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Score arbitrary query-document pairs.

        Useful for custom reranking logic or evaluation.

        Args:
            query_doc_pairs: List of (query, document) tuples

        Returns:
            Array of relevance scores

        Example:
            pairs = [
                ("python", "Python is a language"),
                ("weather", "It's sunny today"),
                ("python", "Weather forecast")
            ]
            scores = reranker.score_pairs(pairs)
        """
        return self.model.predict(query_doc_pairs)


# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = CrossEncoderReranker()

    # Example 1: Basic reranking
    query = "machine learning tutorials"
    documents = [
        "Introduction to machine learning with Python",
        "The history of ancient Rome",
        "Deep learning and neural networks tutorial",
        "Cooking recipes for beginners",
        "Machine learning fundamentals course"
    ]

    print("Query:", query)
    print("\nOriginal documents:")
    for i, doc in enumerate(documents):
        print(f"{i}: {doc}")

    # Rerank and get top 3
    top_indices = reranker.rerank(query, documents, top_k=3, return_scores=True)

    print("\nReranked top 3:")
    for idx, score in top_indices:
        print(f"{idx} (score: {score:.4f}): {documents[idx]}")

    # Example 2: Reranking with metadata
    print("\n" + "="*60)
    print("Example 2: Reranking with metadata")
    print("="*60)

    docs_with_metadata = [
        {
            "id": "doc1",
            "text": "Python programming basics",
            "author": "Alice",
            "date": "2024-01-15"
        },
        {
            "id": "doc2",
            "text": "Advanced JavaScript techniques",
            "author": "Bob",
            "date": "2024-02-20"
        },
        {
            "id": "doc3",
            "text": "Python for data science",
            "author": "Carol",
            "date": "2024-03-10"
        }
    ]

    query = "python programming"
    ranked_docs = reranker.rerank_with_metadata(
        query,
        docs_with_metadata,
        text_key="text",
        top_k=2
    )

    print(f"\nQuery: {query}")
    print("\nTop 2 ranked documents:")
    for doc in ranked_docs:
        print(f"  ID: {doc['id']}")
        print(f"  Text: {doc['text']}")
        print(f"  Score: {doc['rerank_score']:.4f}")
        print(f"  Author: {doc['author']}")
        print()
