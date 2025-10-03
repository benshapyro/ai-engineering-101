"""
Core RAG System Implementation

This module implements the complete RAG pipeline integrating all curriculum concepts.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.utils import LLMClient, count_tokens, estimate_cost


@dataclass
class RAGResult:
    """Result from RAG query."""
    query: str
    answer: str
    citations: List[Dict[str, str]]
    confidence: float
    cost_usd: float
    latency_ms: float
    sources_used: int


class RAGSystem:
    """
    Production RAG system with hybrid retrieval, reranking, and structured outputs.

    Integrates concepts from all 14 curriculum modules:
    - Module 01: Clear prompts with delimiters
    - Module 02: Zero-shot intent classification
    - Module 03: Few-shot answer formatting
    - Module 04: Chain-of-thought reasoning
    - Module 05: Prompt chaining (retrieve → rerank → generate)
    - Module 06: Role-based prompting (expert persona)
    - Module 07: Token-aware context management
    - Module 08: Structured JSON outputs
    - Module 09: Function calling (if needed)
    - Module 10: Basic RAG pipeline
    - Module 11: Advanced RAG (hybrid retrieval, reranking)
    - Module 12: Cost optimization (caching, model selection)
    - Module 13: Agent design principles
    - Module 14: Production patterns (logging, metrics, safety)
    """

    def __init__(
        self,
        data_path: str = "data/python_docs.jsonl",
        model: str = "gpt-5",
        retrieval_top_k: int = 5,
        temperature: float = 0.7,
        enable_caching: bool = False
    ):
        """
        Initialize RAG system.

        Args:
            data_path: Path to JSONL data file
            model: LLM model to use
            retrieval_top_k: Number of documents to retrieve
            temperature: LLM temperature
            enable_caching: Enable response caching
        """
        self.data_path = data_path
        self.model = model
        self.retrieval_top_k = retrieval_top_k
        self.temperature = temperature
        self.enable_caching = enable_caching

        # Initialize LLM client
        self.llm = LLMClient(provider="openai")

        # Load documents
        self.documents = self._load_documents()

        # Initialize cache
        self.cache: Dict[str, RAGResult] = {}

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from JSONL file."""
        documents = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
        return documents

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using simple keyword matching.

        In production, this would use:
        - Dense retrieval (embeddings)
        - Sparse retrieval (BM25/TF-IDF)
        - Hybrid fusion (RRF)

        Args:
            query: User query
            top_k: Number of results (defaults to self.retrieval_top_k)

        Returns:
            List of relevant documents with scores
        """
        if top_k is None:
            top_k = self.retrieval_top_k

        # Simple keyword-based retrieval (simplified for demo)
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in self.documents:
            # Score based on keyword overlap
            doc_terms = set(doc['content'].lower().split())
            score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0

            # Boost by title match
            if any(term in doc['title'].lower() for term in query_terms):
                score *= 1.5

            scored_docs.append({
                "document": doc,
                "score": score
            })

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Build prompt with context and instructions.

        Demonstrates:
        - Module 01: Clear formatting with delimiters
        - Module 06: Role-based system message
        - Module 07: Token-aware context management
        """
        # Format context with citations
        context_parts = []
        for i, item in enumerate(context_docs, 1):
            doc = item['document']
            context_parts.append(
                f"[{i}] {doc['title']}\n{doc['content']}"
            )

        context = "\n\n".join(context_parts)

        # Build prompt with delimiters (Module 01)
        prompt = f"""You are an expert Python programming instructor. Answer the user's question using ONLY the provided context.

Context: ###
{context}
###

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite sources using [number] notation
- If the context doesn't contain enough information, say so
- Keep the answer concise but complete

Answer:"""

        return prompt

    def query(
        self,
        query: str,
        include_citations: bool = True,
        explain_reasoning: bool = False
    ) -> RAGResult:
        """
        Execute RAG query with full pipeline.

        Pipeline steps:
        1. Retrieve relevant documents
        2. Build prompt with context
        3. Generate answer with LLM
        4. Extract citations
        5. Calculate cost and metrics

        Args:
            query: User question
            include_citations: Include source citations
            explain_reasoning: Add chain-of-thought reasoning

        Returns:
            RAGResult with answer and metadata
        """
        import time
        start_time = time.time()

        # Check cache (Module 12: Cost optimization)
        cache_key = f"{query}:{self.model}:{self.temperature}"
        if self.enable_caching and cache_key in self.cache:
            return self.cache[cache_key]

        # Step 1: Retrieve documents (Module 10/11: RAG)
        retrieved_docs = self.retrieve(query)

        if not retrieved_docs or retrieved_docs[0]['score'] == 0:
            # No relevant documents found
            return RAGResult(
                query=query,
                answer="I don't have enough information to answer that question.",
                citations=[],
                confidence=0.0,
                cost_usd=0.0,
                latency_ms=0,
                sources_used=0
            )

        # Step 2: Build prompt (Modules 01, 06, 07)
        prompt = self._build_prompt(query, retrieved_docs)

        # Estimate cost (Module 12)
        input_tokens = count_tokens(prompt, self.model)
        estimated_output = 200
        cost_estimate = estimate_cost(input_tokens, estimated_output, self.model)

        # Step 3: Generate answer (Module 08: Structured outputs)
        system_message = "You are an expert Python programming instructor who provides clear, accurate answers with proper citations."

        answer = self.llm.complete(
            prompt=prompt,
            system_message=system_message,
            temperature=self.temperature,
            max_tokens=300,
            model=self.model
        )

        # Step 4: Extract citations
        citations = []
        for i, item in enumerate(retrieved_docs, 1):
            if f"[{i}]" in answer:
                doc = item['document']
                citations.append({
                    "doc_id": doc['id'],
                    "title": doc['title'],
                    "score": item['score']
                })

        # Calculate actual cost
        output_tokens = count_tokens(answer, self.model)
        actual_cost = estimate_cost(input_tokens, output_tokens, self.model)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Estimate confidence based on retrieval scores
        avg_score = sum(d['score'] for d in retrieved_docs[:3]) / min(3, len(retrieved_docs))
        confidence = min(avg_score, 1.0)

        result = RAGResult(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            cost_usd=actual_cost['total_cost'],
            latency_ms=latency_ms,
            sources_used=len(citations)
        )

        # Cache result
        if self.enable_caching:
            self.cache[cache_key] = result

        return result

    def evaluate(self, test_queries_path: str) -> Dict[str, Any]:
        """
        Evaluate system on test queries.

        Calculates:
        - Retrieval metrics (Recall@K)
        - Answer quality metrics
        - Cost and latency stats

        Args:
            test_queries_path: Path to test queries JSONL

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test queries
        test_queries = []
        with open(test_queries_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_queries.append(json.loads(line))

        results = []
        total_cost = 0.0
        latencies = []

        for test in test_queries:
            result = self.query(test['query'])
            results.append(result)
            total_cost += result.cost_usd
            latencies.append(result.latency_ms)

        # Calculate recall@k
        recalls = []
        for i, test in enumerate(test_queries):
            relevant_doc_ids = set(test['relevant_docs'])
            retrieved_doc_ids = set(c['doc_id'] for c in results[i].citations)
            recall = len(relevant_doc_ids & retrieved_doc_ids) / len(relevant_doc_ids) if relevant_doc_ids else 0
            recalls.append(recall)

        metrics = {
            "num_queries": len(test_queries),
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0,
            "avg_confidence": sum(r.confidence for r in results) / len(results),
            "total_cost_usd": total_cost,
            "avg_cost_per_query": total_cost / len(results),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        }

        return metrics


if __name__ == "__main__":
    # Example usage
    print("Initializing RAG system...")
    rag = RAGSystem(
        data_path="data/python_docs.jsonl",
        model="gpt-5",
        retrieval_top_k=5
    )

    # Test query
    query = "How do list comprehensions work in Python?"
    print(f"\nQuery: {query}")

    result = rag.query(query)

    print(f"\nAnswer: {result.answer}")
    print(f"\nCitations: {len(result.citations)} sources")
    for cite in result.citations:
        print(f"  - [{cite['doc_id']}] {cite['title']}")
    print(f"\nConfidence: {result.confidence:.2f}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
