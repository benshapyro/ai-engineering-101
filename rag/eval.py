"""
RAG Evaluation Harness

This module provides metrics and tools for evaluating RAG systems:
- Faithfulness (answer grounded in context?)
- Answer relevancy (answers the query?)
- Retrieval quality (hit rate, MRR, precision@k)

Usage:
    python -m rag.eval --dataset data/rag_eval_min.jsonl
"""

import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path


@dataclass
class EvalResult:
    """Single evaluation result."""
    query: str
    faithfulness: float
    answer_relevancy: float
    hit_rate: float
    precision_at_k: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaithfulnessEvaluator:
    """
    Evaluate whether answer is faithful to retrieved context.

    Two modes:
    - Rule-based: Check for hallucination patterns
    - LLM-judge: Use LLM to assess faithfulness
    """

    def __init__(self, mode: str = "rule"):
        """
        Initialize evaluator.

        Args:
            mode: "rule" or "llm"
        """
        self.mode = mode

    def evaluate_rule_based(
        self,
        answer: str,
        context: List[str]
    ) -> float:
        """
        Rule-based faithfulness check.

        Checks:
        - Are key entities in answer present in context?
        - Are specific claims supported?

        Args:
            answer: Generated answer
            context: Retrieved context snippets

        Returns:
            Faithfulness score (0-1)
        """
        # Combine context
        full_context = " ".join(context).lower()
        answer_lower = answer.lower()

        # Extract potential claims (sentences with factual patterns)
        # This is a simplified heuristic
        claim_patterns = [
            " is ", " are ", " was ", " were ",
            " has ", " have ", " can ", " will "
        ]

        claims = []
        for sentence in answer.split('.'):
            sentence = sentence.strip()
            if any(pattern in sentence.lower() for pattern in claim_patterns):
                claims.append(sentence)

        if not claims:
            # No factual claims, assume faithful
            return 1.0

        # Check what fraction of claims have support in context
        supported = 0
        for claim in claims:
            # Simple overlap check (5+ overlapping words suggests support)
            claim_words = set(claim.lower().split())
            context_words = set(full_context.split())
            overlap = len(claim_words & context_words)

            if overlap >= 5:
                supported += 1

        return supported / len(claims) if claims else 1.0

    def evaluate_llm_judge(
        self,
        answer: str,
        context: List[str],
        client
    ) -> float:
        """
        LLM-as-judge faithfulness evaluation.

        Args:
            answer: Generated answer
            context: Retrieved context
            client: LLMClient instance

        Returns:
            Faithfulness score (0-1)
        """
        context_text = "\n\n".join(context)

        prompt = f"""Evaluate whether the answer is faithful to the context.
A faithful answer only makes claims that are supported by the context.

Context:
{context_text}

Answer:
{answer}

Is this answer faithful to the context? Rate from 0.0 (completely unfaithful) to 1.0 (perfectly faithful).
Provide only the numeric score."""

        response = client.generate(
            input_text=prompt,
            temperature=0.0,
            max_tokens=10
        )

        output = client.get_output_text(response)

        # Extract score
        try:
            score = float(output.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            return 0.5  # Default if parsing fails

    def evaluate(
        self,
        answer: str,
        context: List[str],
        client=None
    ) -> float:
        """
        Evaluate faithfulness using configured mode.

        Args:
            answer: Generated answer
            context: Retrieved context
            client: Optional LLM client (required for llm mode)

        Returns:
            Faithfulness score (0-1)
        """
        if self.mode == "rule":
            return self.evaluate_rule_based(answer, context)
        elif self.mode == "llm":
            if client is None:
                raise ValueError("LLM mode requires client")
            return self.evaluate_llm_judge(answer, context, client)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class RelevancyEvaluator:
    """Evaluate whether answer addresses the query."""

    def evaluate_keyword_overlap(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Simple keyword overlap metric.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        answer_words -= stop_words

        if not query_words:
            return 1.0

        overlap = len(query_words & answer_words)
        return overlap / len(query_words)

    def evaluate_llm_judge(
        self,
        query: str,
        answer: str,
        client
    ) -> float:
        """
        LLM-as-judge relevancy evaluation.

        Args:
            query: User query
            answer: Generated answer
            client: LLMClient instance

        Returns:
            Relevancy score (0-1)
        """
        prompt = f"""Does this answer address the query?

Query: {query}

Answer: {answer}

Rate from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Provide only the numeric score."""

        response = client.generate(
            input_text=prompt,
            temperature=0.0,
            max_tokens=10
        )

        output = client.get_output_text(response)

        try:
            score = float(output.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5


class RetrievalMetrics:
    """Compute retrieval quality metrics."""

    @staticmethod
    def hit_rate(
        retrieved_ids: List[int],
        relevant_ids: List[int]
    ) -> float:
        """
        Hit rate: Is any relevant doc in retrieved set?

        Args:
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        return 1.0 if any(rid in retrieved_ids for rid in relevant_ids) else 0.0

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Precision@k: What fraction of top-k are relevant?

        Args:
            retrieved_ids: Retrieved document IDs (in rank order)
            relevant_ids: Relevant document IDs
            k: Cutoff (default: len(retrieved))

        Returns:
            Precision score (0-1)
        """
        if k is None:
            k = len(retrieved_ids)

        retrieved_k = retrieved_ids[:k]

        if not retrieved_k:
            return 0.0

        relevant_count = sum(1 for rid in retrieved_k if rid in relevant_ids)
        return relevant_count / len(retrieved_k)

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Recall@k: What fraction of relevant docs are in top-k?

        Args:
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs
            k: Cutoff (default: len(retrieved))

        Returns:
            Recall score (0-1)
        """
        if not relevant_ids:
            return 1.0

        if k is None:
            k = len(retrieved_ids)

        retrieved_k = retrieved_ids[:k]
        relevant_count = sum(1 for rid in relevant_ids if rid in retrieved_k)

        return relevant_count / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: List[int],
        relevant_ids: List[int]
    ) -> float:
        """
        MRR: Reciprocal rank of first relevant document.

        Args:
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs

        Returns:
            MRR score
        """
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@k.

        Args:
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs
            k: Cutoff

        Returns:
            nDCG score (0-1)
        """
        if k is None:
            k = len(retrieved_ids)

        # DCG: sum(rel_i / log2(i+1))
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k], 1):
            relevance = 1.0 if rid in relevant_ids else 0.0
            dcg += relevance / np.log2(i + 1)

        # IDCG: perfect ranking
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant_ids)) + 1))

        return dcg / idcg if idcg > 0 else 0.0


class RAGEvaluator:
    """Complete RAG evaluation harness."""

    def __init__(
        self,
        faithfulness_mode: str = "rule",
        relevancy_mode: str = "keyword",
        client=None
    ):
        """
        Initialize evaluator.

        Args:
            faithfulness_mode: "rule" or "llm"
            relevancy_mode: "keyword" or "llm"
            client: Optional LLM client for judge modes
        """
        self.faithfulness_eval = FaithfulnessEvaluator(faithfulness_mode)
        self.relevancy_mode = relevancy_mode
        self.retrieval_metrics = RetrievalMetrics()
        self.client = client

    def evaluate_single(
        self,
        item: Dict[str, Any],
        retrieved_ids: Optional[List[int]] = None
    ) -> EvalResult:
        """
        Evaluate single RAG example.

        Args:
            item: Evaluation item with query, answer, context, relevant_docs
            retrieved_ids: Optional retrieved doc IDs (for retrieval metrics)

        Returns:
            EvalResult
        """
        # Faithfulness
        faithfulness = self.faithfulness_eval.evaluate(
            item['answer'],
            item['context'],
            self.client
        )

        # Answer relevancy
        if self.relevancy_mode == "keyword":
            relevancy_eval = RelevancyEvaluator()
            answer_relevancy = relevancy_eval.evaluate_keyword_overlap(
                item['query'],
                item['answer']
            )
        elif self.relevancy_mode == "llm":
            relevancy_eval = RelevancyEvaluator()
            answer_relevancy = relevancy_eval.evaluate_llm_judge(
                item['query'],
                item['answer'],
                self.client
            )
        else:
            answer_relevancy = 0.0

        # Retrieval metrics
        if retrieved_ids is None:
            # Assume retrieved = context indices
            retrieved_ids = list(range(len(item['context'])))

        hit_rate = self.retrieval_metrics.hit_rate(
            retrieved_ids,
            item['relevant_docs']
        )

        precision = self.retrieval_metrics.precision_at_k(
            retrieved_ids,
            item['relevant_docs'],
            k=5
        )

        return EvalResult(
            query=item['query'],
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            hit_rate=hit_rate,
            precision_at_k=precision
        )

    def evaluate_dataset(
        self,
        dataset_path: str
    ) -> Tuple[List[EvalResult], Dict[str, float]]:
        """
        Evaluate entire dataset.

        Args:
            dataset_path: Path to JSONL file

        Returns:
            Tuple of (results list, aggregated metrics)
        """
        results = []

        # Load dataset
        with open(dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                result = self.evaluate_single(item)
                results.append(result)

        # Aggregate metrics
        metrics = {
            "faithfulness_avg": np.mean([r.faithfulness for r in results]),
            "answer_relevancy_avg": np.mean([r.answer_relevancy for r in results]),
            "hit_rate_avg": np.mean([r.hit_rate for r in results]),
            "precision@5_avg": np.mean([r.precision_at_k for r in results]),
            "num_examples": len(results)
        }

        return results, metrics


def print_report(results: List[EvalResult], metrics: Dict[str, float]):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)

    print(f"\nDataset: {metrics['num_examples']} examples")

    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)
    print(f"Faithfulness:      {metrics['faithfulness_avg']:.3f}")
    print(f"Answer Relevancy:  {metrics['answer_relevancy_avg']:.3f}")
    print(f"Hit Rate:          {metrics['hit_rate_avg']:.3f}")
    print(f"Precision@5:       {metrics['precision@5_avg']:.3f}")

    print("\n" + "-" * 70)
    print("PER-QUERY BREAKDOWN")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.query}")
        print(f"   Faithfulness: {result.faithfulness:.3f} | "
              f"Relevancy: {result.answer_relevancy:.3f} | "
              f"Hit: {result.hit_rate:.0f} | "
              f"P@5: {result.precision_at_k:.3f}")

    print("\n" + "=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rag_eval_min.jsonl",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--faithfulness-mode",
        type=str,
        default="rule",
        choices=["rule", "llm"],
        help="Faithfulness evaluation mode"
    )
    parser.add_argument(
        "--relevancy-mode",
        type=str,
        default="keyword",
        choices=["keyword", "llm"],
        help="Relevancy evaluation mode"
    )

    args = parser.parse_args()

    # Initialize evaluator
    client = None
    if args.faithfulness_mode == "llm" or args.relevancy_mode == "llm":
        from llm.client import LLMClient
        client = LLMClient()

    evaluator = RAGEvaluator(
        faithfulness_mode=args.faithfulness_mode,
        relevancy_mode=args.relevancy_mode,
        client=client
    )

    # Run evaluation
    print(f"Evaluating dataset: {args.dataset}")
    results, metrics = evaluator.evaluate_dataset(args.dataset)

    # Print report
    print_report(results, metrics)


if __name__ == "__main__":
    main()
