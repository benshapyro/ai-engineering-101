"""
Module 11: Advanced RAG - Query Understanding Pipeline

This example demonstrates advanced query processing techniques:
- Intent classification
- Query rewriting
- Query decomposition

These techniques improve retrieval quality by transforming user queries
into optimal search queries.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llm.client import LLMClient
from shared.structured import create_json_schema, ask_json

load_dotenv()


# ================================
# Example 1: Intent Classification
# ================================
print("=" * 60)
print("Example 1: Intent Classification")
print("=" * 60)

@dataclass
class QueryIntent:
    """Classified query intent."""
    intent: str  # factual, comparative, exploratory, procedural
    confidence: float
    reasoning: str


def classify_intent(query: str, client: LLMClient) -> Dict[str, Any]:
    """
    Classify the intent of a user query.

    Intent types:
    - factual: Looking for specific facts/data
    - comparative: Comparing multiple things
    - exploratory: Open-ended research
    - procedural: How-to questions

    Args:
        query: User query
        client: LLM client

    Returns:
        Intent classification dict
    """
    schema = create_json_schema(
        name="IntentClassification",
        description="Query intent classification",
        properties={
            "intent": {
                "type": "string",
                "enum": ["factual", "comparative", "exploratory", "procedural"],
                "description": "Primary intent type"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in classification"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of classification"
            }
        }
    )

    instructions = """You are a query intent classifier.
Classify queries into: factual, comparative, exploratory, or procedural."""

    result = ask_json(
        client,
        f"Classify this query: '{query}'",
        schema,
        instructions=instructions,
        temperature=0.0
    )

    return result


# Test intent classification
client = LLMClient()

test_queries = [
    "What is the capital of France?",
    "Compare Python and Java for machine learning",
    "Tell me about quantum computing",
    "How do I train a neural network?"
]

print("\nClassifying query intents:")
for query in test_queries:
    result = classify_intent(query, client)
    print(f"\nQuery: {query}")
    print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
    print(f"Reasoning: {result['reasoning']}")


# ================================
# Example 2: Query Rewriting
# ================================
print("\n\n" + "=" * 60)
print("Example 2: Query Rewriting")
print("=" * 60)


def rewrite_query(query: str, client: LLMClient) -> Dict[str, Any]:
    """
    Rewrite query for better retrieval.

    Improvements:
    - Expand acronyms
    - Add context
    - Fix ambiguity
    - Add synonyms

    Args:
        query: Original query
        client: LLM client

    Returns:
        Rewritten query dict
    """
    schema = create_json_schema(
        name="QueryRewrite",
        description="Rewritten query for better retrieval",
        properties={
            "original": {
                "type": "string",
                "description": "Original query"
            },
            "rewritten": {
                "type": "string",
                "description": "Improved query"
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of improvements made"
            }
        }
    )

    instructions = """You are a query optimization expert.
Rewrite queries to improve retrieval by:
- Expanding acronyms
- Adding context
- Fixing ambiguity
- Including synonyms"""

    result = ask_json(
        client,
        f"Rewrite this query for better search: '{query}'",
        schema,
        instructions=instructions,
        temperature=0.3
    )

    return result


# Test query rewriting
test_queries_rewrite = [
    "ML model training",
    "API docs",
    "Fix bug in code",
    "LLM performance"
]

print("\nRewriting queries:")
for query in test_queries_rewrite:
    result = rewrite_query(query, client)
    print(f"\nOriginal: {result['original']}")
    print(f"Rewritten: {result['rewritten']}")
    print(f"Improvements: {', '.join(result['improvements'])}")


# ================================
# Example 3: Query Decomposition
# ================================
print("\n\n" + "=" * 60)
print("Example 3: Query Decomposition")
print("=" * 60)


def decompose_query(query: str, client: LLMClient) -> Dict[str, Any]:
    """
    Break complex query into sub-queries.

    Useful for multi-hop reasoning and complex questions.

    Args:
        query: Complex query
        client: LLM client

    Returns:
        Decomposed sub-queries dict
    """
    schema = create_json_schema(
        name="QueryDecomposition",
        description="Complex query broken into sub-queries",
        properties={
            "original": {
                "type": "string",
                "description": "Original complex query"
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of simpler sub-queries"
            },
            "execution_order": {
                "type": "string",
                "enum": ["sequential", "parallel"],
                "description": "How sub-queries should be executed"
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of decomposition"
            }
        }
    )

    instructions = """You are a query decomposition expert.
Break complex queries into simpler sub-queries that can be:
- Executed in parallel (independent)
- Executed sequentially (dependent)"""

    result = ask_json(
        client,
        f"Decompose this complex query: '{query}'",
        schema,
        instructions=instructions,
        temperature=0.3
    )

    return result


# Test query decomposition
complex_queries = [
    "Compare the performance of Python and JavaScript for machine learning, considering both speed and ecosystem",
    "What are the differences between GPT-4 and Claude 3, and which one is better for code generation?",
    "How do I set up a RAG system with vector database, reranking, and monitoring?"
]

print("\nDecomposing complex queries:")
for query in complex_queries:
    result = decompose_query(query, client)
    print(f"\nOriginal: {result['original']}")
    print(f"Execution: {result['execution_order']}")
    print("Sub-queries:")
    for i, sub_q in enumerate(result['sub_queries'], 1):
        print(f"  {i}. {sub_q}")
    print(f"Reasoning: {result['reasoning']}")


# ================================
# Example 4: Complete Pipeline
# ================================
print("\n\n" + "=" * 60)
print("Example 4: Complete Query Understanding Pipeline")
print("=" * 60)


class QueryPipeline:
    """
    Complete query understanding pipeline.

    Combines intent classification, rewriting, and decomposition
    based on query complexity.
    """

    def __init__(self, client: LLMClient):
        """Initialize pipeline with LLM client."""
        self.client = client

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process query through complete pipeline.

        Args:
            query: User query

        Returns:
            Processed query info
        """
        # Step 1: Classify intent
        intent_result = classify_intent(query, self.client)

        # Step 2: Rewrite for better retrieval
        rewrite_result = rewrite_query(query, self.client)

        # Step 3: Decompose if complex (exploratory or comparative)
        sub_queries = None
        if intent_result['intent'] in ['exploratory', 'comparative']:
            decomp_result = decompose_query(query, self.client)
            sub_queries = decomp_result['sub_queries']

        return {
            "original_query": query,
            "intent": intent_result['intent'],
            "confidence": intent_result['confidence'],
            "rewritten_query": rewrite_result['rewritten'],
            "improvements": rewrite_result['improvements'],
            "sub_queries": sub_queries,
            "execution_strategy": "decomposed" if sub_queries else "direct"
        }


# Test complete pipeline
pipeline = QueryPipeline(client)

test_queries_pipeline = [
    "What is prompt engineering?",  # Factual - direct
    "Compare RAG approaches",  # Comparative - decompose
    "How do I optimize LLM costs?"  # Procedural - direct
]

print("\nProcessing queries through pipeline:")
for query in test_queries_pipeline:
    print(f"\n{'=' * 60}")
    result = pipeline.process(query)

    print(f"Original: {result['original_query']}")
    print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print(f"Strategy: {result['execution_strategy']}")
    print(f"Rewritten: {result['rewritten_query']}")

    if result['sub_queries']:
        print("Sub-queries:")
        for i, sq in enumerate(result['sub_queries'], 1):
            print(f"  {i}. {sq}")


# ================================
# Example 5: A/B Comparison
# ================================
print("\n\n" + "=" * 60)
print("Example 5: Query Pipeline Impact on Retrieval")
print("=" * 60)

print("""
Demonstrating how query processing improves retrieval:

BEFORE (naive query):
Query: "ML performance"
Retrieved docs might include:
- Machine learning model performance metrics
- ML system performance optimization
- Performance testing in general
- Machine learning hardware requirements

AFTER (processed query):
Original: "ML performance"
Rewritten: "Machine learning model performance optimization techniques,
           including training speed, inference latency, and accuracy metrics"

Retrieved docs now better match intent:
- Model optimization techniques
- Performance benchmarking
- Speed vs accuracy trade-offs
- Inference optimization strategies

KEY IMPROVEMENTS:
1. Acronym expansion (ML → machine learning)
2. Context addition (what kind of performance?)
3. Synonym inclusion (speed, latency, metrics)
4. Intent clarification (optimization focus)

RESULT: Higher precision and recall in retrieval
""")


# ================================
# Rubric and Best Practices
# ================================
print("\n\n" + "=" * 60)
print("Query Understanding Best Practices")
print("=" * 60)

best_practices = """
1. INTENT CLASSIFICATION
   ✓ Use it to: Route to different retrieval strategies
   ✓ Factual → Precise search, low top-k
   ✓ Exploratory → Broader search, high top-k
   ✓ Comparative → Ensure multiple perspectives retrieved

2. QUERY REWRITING
   ✓ Always expand acronyms
   ✓ Add domain context when ambiguous
   ✓ Include synonyms for key terms
   ✗ Don't over-expand (keep it focused)
   ✗ Don't change user's core intent

3. QUERY DECOMPOSITION
   ✓ Use for multi-hop questions
   ✓ Execute sub-queries in parallel when independent
   ✓ Execute sequentially when later queries depend on earlier results
   ✗ Don't decompose simple queries (adds latency)
   ✗ Don't create more than 3-4 sub-queries (diminishing returns)

4. PIPELINE INTEGRATION
   ✓ Cache query rewrites for common patterns
   ✓ Log pipeline decisions for debugging
   ✓ Monitor impact on retrieval quality
   ✓ A/B test with/without pipeline
   ✗ Don't apply all steps blindly (use intent to decide)

5. PERFORMANCE CONSIDERATIONS
   ✓ Query processing adds latency (budget ~200-500ms)
   ✓ Consider skipping for simple factual queries
   ✓ Use lower temperature (0.0-0.3) for consistency
   ✓ Cache results for repeated queries
"""

print(best_practices)


# ================================
# Exercise: Implement Your Own
# ================================
print("\n\n" + "=" * 60)
print("EXERCISE: Build Your Query Pipeline")
print("=" * 60)

exercise = """
TASK: Enhance the QueryPipeline class with these features:

1. Add query expansion using synonyms
   - Input: "car"
   - Output: "car OR automobile OR vehicle"

2. Add spelling correction
   - Input: "machne lerning"
   - Output: "machine learning"

3. Add entity extraction
   - Input: "Claude vs GPT-4"
   - Entities: ["Claude", "GPT-4"]
   - Type: "comparison"

4. Add caching to avoid re-processing
   - Store results by query hash
   - Invalidate after 24 hours

5. Add A/B testing metrics
   - Track retrieval quality with/without pipeline
   - Measure: precision@k, recall@k, latency

BONUS: Implement adaptive pipeline
- Simple queries → fast path (no processing)
- Complex queries → full pipeline
- Use query length + complexity score to decide

EVALUATION RUBRIC:
□ Query expansion works correctly
□ Spelling correction handles common mistakes
□ Entity extraction is accurate
□ Caching reduces latency for repeated queries
□ A/B metrics show improvement
□ Code is well-documented and tested
"""

print(exercise)

print("\n" + "=" * 60)
print("Query Understanding Pipeline Complete!")
print("=" * 60)
