"""
Module 11: Advanced RAG - Advanced Reranking Examples

This file demonstrates sophisticated reranking techniques including
cross-encoder models, diversity-aware ranking, and personalization.

Author: Claude
Date: 2024
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Example 1: Cross-Encoder Reranking
# ================================
print("=" * 50)
print("Example 1: Cross-Encoder Reranking")
print("=" * 50)

@dataclass
class RankedDocument:
    """Document with ranking scores."""
    id: str
    content: str
    initial_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossEncoderReranker:
    """Rerank documents using cross-encoder approach."""

    def __init__(self, model_type: str = "llm"):
        """
        Initialize reranker.

        Args:
            model_type: Type of cross-encoder ('llm' or 'simulated')
        """
        self.model_type = model_type
        self.cache = {}

    def rerank(
        self,
        query: str,
        documents: List[RankedDocument],
        top_k: int = 5
    ) -> List[RankedDocument]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked documents
        """
        # Score each document
        for doc in documents:
            score = self._score_pair(query, doc.content)
            doc.rerank_score = score

        # Sort by rerank score
        documents.sort(key=lambda x: x.rerank_score, reverse=True)

        # Combine with initial scores
        for doc in documents:
            doc.final_score = 0.7 * doc.rerank_score + 0.3 * doc.initial_score

        # Final sort
        documents.sort(key=lambda x: x.final_score, reverse=True)

        return documents[:top_k]

    def _score_pair(self, query: str, document: str) -> float:
        """Score query-document pair."""
        # Check cache
        cache_key = f"{query[:50]}_{document[:50]}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.model_type == "llm":
            score = self._llm_scoring(query, document)
        else:
            score = self._simulated_scoring(query, document)

        self.cache[cache_key] = score
        return score

    def _llm_scoring(self, query: str, document: str) -> float:
        """Score using LLM."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Rate the relevance of the document to the query from 0 to 10. Respond with just a number."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocument: {document[:500]}"
                }
            ],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return score / 10.0
        except:
            return 0.5

    def _simulated_scoring(self, query: str, document: str) -> float:
        """Simulated cross-encoder scoring."""
        # Simple word overlap + length penalty
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        overlap = len(query_words & doc_words)
        query_coverage = overlap / len(query_words) if query_words else 0

        # Length penalty
        doc_length = len(document.split())
        length_penalty = min(1.0, 100 / doc_length) if doc_length > 100 else 1.0

        return query_coverage * length_penalty

    def batch_rerank(
        self,
        queries: List[str],
        document_sets: List[List[RankedDocument]],
        top_k: int = 5
    ) -> List[List[RankedDocument]]:
        """Batch reranking for multiple queries."""
        results = []

        for query, docs in zip(queries, document_sets):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)

        return results

# Test cross-encoder reranking
reranker = CrossEncoderReranker(model_type="simulated")

# Create sample documents
documents = [
    RankedDocument(
        id="doc1",
        content="Machine learning is a subset of AI that enables systems to learn from data.",
        initial_score=0.8
    ),
    RankedDocument(
        id="doc2",
        content="Deep learning uses neural networks with multiple layers.",
        initial_score=0.7
    ),
    RankedDocument(
        id="doc3",
        content="Natural language processing helps computers understand human language.",
        initial_score=0.6
    ),
    RankedDocument(
        id="doc4",
        content="Computer vision enables machines to interpret visual information.",
        initial_score=0.5
    ),
    RankedDocument(
        id="doc5",
        content="Reinforcement learning trains agents through rewards and penalties.",
        initial_score=0.4
    )
]

query = "How do neural networks learn from data?"

# Rerank documents
reranked = reranker.rerank(query, documents.copy(), top_k=3)

print(f"Query: {query}\n")
print("Reranked Results:")
for i, doc in enumerate(reranked, 1):
    print(f"{i}. {doc.id}")
    print(f"   Initial: {doc.initial_score:.3f}, Rerank: {doc.rerank_score:.3f}, Final: {doc.final_score:.3f}")
    print(f"   Content: {doc.content[:60]}...")

# ================================
# Example 2: Maximal Marginal Relevance (MMR)
# ================================
print("\n" + "=" * 50)
print("Example 2: Maximal Marginal Relevance (MMR)")
print("=" * 50)

class MMRReranker:
    """Rerank for diversity using Maximal Marginal Relevance."""

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
                         Higher values favor relevance, lower favor diversity
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        documents: List[RankedDocument],
        k: int = 5
    ) -> List[RankedDocument]:
        """
        Rerank documents using MMR algorithm.

        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding vectors
            documents: Documents to rerank
            k: Number of documents to select

        Returns:
            Reranked documents with diversity
        """
        selected = []
        selected_embeddings = []
        remaining_indices = list(range(len(documents)))

        # Select first document (most relevant)
        relevance_scores = self._calculate_relevance(query_embedding, doc_embeddings)
        best_idx = np.argmax(relevance_scores)
        selected.append(documents[best_idx])
        selected_embeddings.append(doc_embeddings[best_idx])
        remaining_indices.remove(best_idx)

        # Iteratively select diverse relevant documents
        while len(selected) < k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]

                # Maximum similarity to selected documents
                if selected_embeddings:
                    similarities = [
                        self._cosine_similarity(doc_embeddings[idx], sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append((idx, mmr))

            # Select document with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = mmr_scores[0][0]

            selected.append(documents[best_idx])
            selected_embeddings.append(doc_embeddings[best_idx])
            remaining_indices.remove(best_idx)

        return selected

    def _calculate_relevance(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate relevance scores between query and documents."""
        scores = []
        for doc_emb in doc_embeddings:
            score = self._cosine_similarity(query_embedding, doc_emb)
            scores.append(score)
        return np.array(scores)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def adaptive_mmr(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        documents: List[RankedDocument],
        k: int = 5,
        auto_lambda: bool = True
    ) -> List[RankedDocument]:
        """
        MMR with adaptive lambda based on result set characteristics.

        Args:
            auto_lambda: Automatically adjust lambda based on diversity needs
        """
        if auto_lambda:
            # Calculate average pairwise similarity
            avg_similarity = self._calculate_avg_similarity(doc_embeddings)

            # Adjust lambda based on similarity
            if avg_similarity > 0.8:
                # Documents are very similar, favor diversity
                self.lambda_param = 0.3
            elif avg_similarity > 0.6:
                # Moderate similarity
                self.lambda_param = 0.5
            else:
                # Documents are diverse, favor relevance
                self.lambda_param = 0.7

            print(f"Auto-adjusted lambda to {self.lambda_param} (avg similarity: {avg_similarity:.3f})")

        return self.rerank(query_embedding, doc_embeddings, documents, k)

    def _calculate_avg_similarity(self, embeddings: List[np.ndarray]) -> float:
        """Calculate average pairwise similarity."""
        if len(embeddings) < 2:
            return 0.0

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

# Test MMR reranking
mmr_reranker = MMRReranker(lambda_param=0.5)

# Generate embeddings for documents
doc_embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc.content
    )
    doc_embeddings.append(np.array(response.data[0].embedding))

# Generate query embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
query_embedding = np.array(response.data[0].embedding)

# Rerank with MMR
mmr_results = mmr_reranker.rerank(query_embedding, doc_embeddings, documents.copy(), k=3)

print(f"\nMMR Reranking (lambda={mmr_reranker.lambda_param}):")
for i, doc in enumerate(mmr_results, 1):
    print(f"{i}. {doc.id}: {doc.content[:60]}...")

# Test adaptive MMR
adaptive_results = mmr_reranker.adaptive_mmr(
    query_embedding,
    doc_embeddings,
    documents.copy(),
    k=3,
    auto_lambda=True
)

print(f"\nAdaptive MMR Results:")
for i, doc in enumerate(adaptive_results, 1):
    print(f"{i}. {doc.id}: {doc.content[:60]}...")

# ================================
# Example 3: Personalized Reranking
# ================================
print("\n" + "=" * 50)
print("Example 3: Personalized Reranking")
print("=" * 50)

@dataclass
class UserProfile:
    """User profile for personalization."""
    user_id: str
    interests: List[str] = field(default_factory=list)
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    preferences: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)

class PersonalizedReranker:
    """Rerank based on user preferences and history."""

    def __init__(self):
        self.user_profiles = {}

    def add_user_profile(self, profile: UserProfile):
        """Add or update user profile."""
        self.user_profiles[profile.user_id] = profile

    def rerank(
        self,
        user_id: str,
        documents: List[RankedDocument],
        k: int = 5
    ) -> List[RankedDocument]:
        """
        Rerank documents based on user profile.

        Args:
            user_id: User identifier
            documents: Documents to rerank
            k: Number of documents to return

        Returns:
            Personalized ranking
        """
        if user_id not in self.user_profiles:
            # No personalization, return as-is
            return documents[:k]

        profile = self.user_profiles[user_id]

        # Score each document
        for doc in documents:
            personalization_score = self._calculate_personalization_score(doc, profile)
            doc.final_score = 0.6 * doc.initial_score + 0.4 * personalization_score

        # Sort by personalized score
        documents.sort(key=lambda x: x.final_score, reverse=True)

        return documents[:k]

    def _calculate_personalization_score(
        self,
        document: RankedDocument,
        profile: UserProfile
    ) -> float:
        """Calculate personalization score for document."""
        score = 0.0

        # Interest matching
        interest_score = self._match_interests(document.content, profile.interests)
        score += interest_score * 0.4

        # Expertise level matching
        expertise_score = self._match_expertise(document.content, profile.expertise_level)
        score += expertise_score * 0.3

        # Preference matching
        preference_score = self._match_preferences(document.metadata, profile.preferences)
        score += preference_score * 0.2

        # Novelty based on history
        novelty_score = self._calculate_novelty(document.content, profile.interaction_history)
        score += novelty_score * 0.1

        return min(1.0, score)

    def _match_interests(self, content: str, interests: List[str]) -> float:
        """Match document content with user interests."""
        if not interests:
            return 0.5

        content_lower = content.lower()
        matches = sum(1 for interest in interests if interest.lower() in content_lower)
        return matches / len(interests)

    def _match_expertise(self, content: str, expertise_level: str) -> float:
        """Match document complexity with user expertise."""
        # Estimate document complexity
        complexity = self._estimate_complexity(content)

        expertise_scores = {
            "beginner": {"simple": 1.0, "intermediate": 0.5, "advanced": 0.2},
            "intermediate": {"simple": 0.5, "intermediate": 1.0, "advanced": 0.7},
            "expert": {"simple": 0.3, "intermediate": 0.7, "advanced": 1.0}
        }

        return expertise_scores.get(expertise_level, {}).get(complexity, 0.5)

    def _estimate_complexity(self, content: str) -> str:
        """Estimate document complexity."""
        # Simple heuristic based on sentence length and vocabulary
        words = content.split()
        avg_word_length = np.mean([len(word) for word in words])

        if avg_word_length < 5:
            return "simple"
        elif avg_word_length < 7:
            return "intermediate"
        else:
            return "advanced"

    def _match_preferences(
        self,
        metadata: Dict[str, Any],
        preferences: Dict[str, float]
    ) -> float:
        """Match document metadata with user preferences."""
        if not preferences or not metadata:
            return 0.5

        score = 0.0
        matched = 0

        for key, weight in preferences.items():
            if key in metadata:
                # Simple matching
                score += weight
                matched += 1

        return score / len(preferences) if matched > 0 else 0.5

    def _calculate_novelty(
        self,
        content: str,
        history: List[str]
    ) -> float:
        """Calculate novelty score based on interaction history."""
        if not history:
            return 1.0  # Everything is novel

        # Check for similarity with historical items
        content_words = set(content.lower().split())

        max_similarity = 0.0
        for historical_item in history[-10:]:  # Check last 10 items
            hist_words = set(historical_item.lower().split())
            similarity = len(content_words & hist_words) / max(len(content_words), len(hist_words))
            max_similarity = max(max_similarity, similarity)

        # Higher novelty for less similar content
        return 1.0 - max_similarity

# Test personalized reranking
personalized_reranker = PersonalizedReranker()

# Create user profiles
profile1 = UserProfile(
    user_id="user1",
    interests=["machine learning", "neural networks"],
    expertise_level="expert",
    preferences={"technical": 0.9, "practical": 0.6},
    interaction_history=["Basic ML tutorial", "CNN architecture guide"]
)

profile2 = UserProfile(
    user_id="user2",
    interests=["natural language", "computer vision"],
    expertise_level="beginner",
    preferences={"simple": 0.8, "visual": 0.7},
    interaction_history=[]
)

personalized_reranker.add_user_profile(profile1)
personalized_reranker.add_user_profile(profile2)

# Rerank for different users
print("Personalized Rankings:\n")

for user_id in ["user1", "user2"]:
    docs_copy = [
        RankedDocument(d.id, d.content, d.initial_score, metadata={"technical": True})
        for d in documents
    ]

    personalized = personalized_reranker.rerank(user_id, docs_copy, k=3)

    profile = personalized_reranker.user_profiles[user_id]
    print(f"User {user_id} (expertise: {profile.expertise_level}):")
    for i, doc in enumerate(personalized, 1):
        print(f"  {i}. {doc.id}: Score={doc.final_score:.3f}")
    print()

# ================================
# Example 4: Learning to Rank
# ================================
print("\n" + "=" * 50)
print("Example 4: Learning to Rank")
print("=" * 50)

class LearningToRankReranker:
    """Rerank using learned ranking models."""

    def __init__(self):
        self.feature_weights = None
        self.training_data = []

    def extract_features(
        self,
        query: str,
        document: RankedDocument
    ) -> np.ndarray:
        """Extract ranking features."""
        features = []

        # Text similarity features
        query_words = set(query.lower().split())
        doc_words = set(document.content.lower().split())

        # Exact match
        exact_matches = len(query_words & doc_words)
        features.append(exact_matches / len(query_words) if query_words else 0)

        # Partial match
        partial_matches = sum(
            1 for qw in query_words
            for dw in doc_words
            if qw in dw or dw in qw
        )
        features.append(partial_matches / (len(query_words) * len(doc_words)) if query_words and doc_words else 0)

        # Length features
        features.append(min(1.0, len(document.content) / 500))  # Normalized length

        # Position features (if query terms appear early)
        first_match_pos = self._find_first_match_position(query, document.content)
        features.append(1.0 - first_match_pos)  # Earlier is better

        # Initial score
        features.append(document.initial_score)

        return np.array(features)

    def _find_first_match_position(self, query: str, content: str) -> float:
        """Find normalized position of first query term match."""
        query_words = query.lower().split()
        content_lower = content.lower()

        min_pos = len(content)
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                min_pos = min(min_pos, pos)

        return min_pos / len(content) if content else 1.0

    def train(
        self,
        training_queries: List[str],
        training_documents: List[List[RankedDocument]],
        relevance_labels: List[List[float]]
    ):
        """
        Train ranking model.

        Args:
            training_queries: Training queries
            training_documents: Documents for each query
            relevance_labels: Relevance labels for each document
        """
        # Extract features for all training data
        X = []
        y = []

        for query, docs, labels in zip(training_queries, training_documents, relevance_labels):
            for doc, label in zip(docs, labels):
                features = self.extract_features(query, doc)
                X.append(features)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Simple linear model (could use more sophisticated models)
        # Using least squares to find weights
        self.feature_weights = np.linalg.lstsq(X, y, rcond=None)[0]

        print(f"Trained on {len(X)} examples")
        print(f"Feature weights: {self.feature_weights}")

    def rerank(
        self,
        query: str,
        documents: List[RankedDocument],
        k: int = 5
    ) -> List[RankedDocument]:
        """Rerank using learned model."""
        if self.feature_weights is None:
            # No training, use default weights
            self.feature_weights = np.array([0.3, 0.2, 0.1, 0.2, 0.2])

        # Score each document
        for doc in documents:
            features = self.extract_features(query, doc)
            score = np.dot(features, self.feature_weights)
            doc.final_score = score

        # Sort by learned score
        documents.sort(key=lambda x: x.final_score, reverse=True)

        return documents[:k]

# Test learning to rank
ltr_reranker = LearningToRankReranker()

# Create training data (simulated)
training_queries = [
    "machine learning basics",
    "neural network architecture",
    "natural language processing"
]

training_documents = [documents.copy() for _ in training_queries]

# Simulated relevance labels (0-1)
relevance_labels = [
    [0.9, 0.7, 0.3, 0.2, 0.5],  # Relevance for first query
    [0.6, 0.95, 0.4, 0.3, 0.7],  # Relevance for second query
    [0.4, 0.5, 0.9, 0.6, 0.3]   # Relevance for third query
]

# Train the model
ltr_reranker.train(training_queries, training_documents, relevance_labels)

# Test reranking
ltr_results = ltr_reranker.rerank(query, documents.copy(), k=3)

print(f"\nLearning to Rank Results for: {query}")
for i, doc in enumerate(ltr_results, 1):
    print(f"{i}. {doc.id}: Score={doc.final_score:.3f}")
    print(f"   {doc.content[:60]}...")

# ================================
# Example 5: Multi-Stage Reranking
# ================================
print("\n" + "=" * 50)
print("Example 5: Multi-Stage Reranking")
print("=" * 50)

class MultiStageReranker:
    """Multi-stage reranking pipeline."""

    def __init__(self):
        self.stages = []

    def add_stage(
        self,
        name: str,
        reranker,
        input_size: int,
        output_size: int
    ):
        """Add a reranking stage."""
        self.stages.append({
            "name": name,
            "reranker": reranker,
            "input_size": input_size,
            "output_size": output_size
        })

    def rerank(
        self,
        query: str,
        documents: List[RankedDocument],
        verbose: bool = True
    ) -> List[RankedDocument]:
        """Execute multi-stage reranking pipeline."""
        current_docs = documents.copy()

        for i, stage in enumerate(self.stages):
            if verbose:
                print(f"Stage {i+1}: {stage['name']}")
                print(f"  Input: {len(current_docs)} docs, Output: {stage['output_size']} docs")

            # Take top documents for this stage
            stage_input = current_docs[:stage["input_size"]]

            # Apply reranker
            if hasattr(stage["reranker"], "rerank"):
                if stage["name"] == "MMR":
                    # Special handling for MMR (needs embeddings)
                    doc_embeddings = []
                    for doc in stage_input:
                        response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=doc.content
                        )
                        doc_embeddings.append(np.array(response.data[0].embedding))

                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=query
                    )
                    query_emb = np.array(response.data[0].embedding)

                    current_docs = stage["reranker"].rerank(
                        query_emb,
                        doc_embeddings,
                        stage_input,
                        stage["output_size"]
                    )
                else:
                    current_docs = stage["reranker"].rerank(
                        query,
                        stage_input,
                        stage["output_size"]
                    )

        return current_docs

    def parallel_stages(
        self,
        query: str,
        documents: List[RankedDocument],
        aggregation: str = "voting"
    ) -> List[RankedDocument]:
        """Execute stages in parallel and aggregate results."""
        import concurrent.futures

        results = []

        # Run stages in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for stage in self.stages:
                future = executor.submit(
                    self._execute_stage,
                    stage,
                    query,
                    documents.copy()
                )
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                stage_results = future.result()
                results.append(stage_results)

        # Aggregate results
        if aggregation == "voting":
            return self._aggregate_by_voting(results)
        elif aggregation == "score":
            return self._aggregate_by_score(results)
        else:
            return results[0]  # Default to first stage

    def _execute_stage(
        self,
        stage: Dict,
        query: str,
        documents: List[RankedDocument]
    ) -> List[RankedDocument]:
        """Execute a single reranking stage."""
        stage_input = documents[:stage["input_size"]]
        return stage["reranker"].rerank(query, stage_input, stage["output_size"])

    def _aggregate_by_voting(
        self,
        results: List[List[RankedDocument]]
    ) -> List[RankedDocument]:
        """Aggregate results by voting."""
        vote_counts = defaultdict(int)
        doc_map = {}

        for stage_results in results:
            for rank, doc in enumerate(stage_results):
                vote_counts[doc.id] += 1.0 / (rank + 1)  # Higher rank = more votes
                doc_map[doc.id] = doc

        # Sort by votes
        sorted_docs = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def _aggregate_by_score(
        self,
        results: List[List[RankedDocument]]
    ) -> List[RankedDocument]:
        """Aggregate results by average score."""
        score_sums = defaultdict(float)
        score_counts = defaultdict(int)
        doc_map = {}

        for stage_results in results:
            for doc in stage_results:
                score_sums[doc.id] += doc.final_score
                score_counts[doc.id] += 1
                doc_map[doc.id] = doc

        # Calculate average scores
        avg_scores = {
            doc_id: score_sums[doc_id] / score_counts[doc_id]
            for doc_id in score_sums
        }

        # Sort by average score
        sorted_docs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

# Create multi-stage pipeline
pipeline = MultiStageReranker()

# Add stages
pipeline.add_stage(
    "Initial Filter",
    CrossEncoderReranker(model_type="simulated"),
    input_size=5,
    output_size=4
)

pipeline.add_stage(
    "MMR",
    MMRReranker(lambda_param=0.5),
    input_size=4,
    output_size=3
)

pipeline.add_stage(
    "Final Ranking",
    LearningToRankReranker(),
    input_size=3,
    output_size=2
)

# Test multi-stage pipeline
print(f"\nMulti-Stage Reranking Pipeline:")
multi_stage_results = pipeline.rerank(query, documents.copy(), verbose=True)

print(f"\nFinal Results:")
for i, doc in enumerate(multi_stage_results, 1):
    print(f"{i}. {doc.id}: {doc.content[:60]}...")

# ================================
# Example 6: Performance Optimization
# ================================
print("\n" + "=" * 50)
print("Example 6: Performance Optimization")
print("=" * 50)

class OptimizedReranker:
    """Optimized reranker with caching and batching."""

    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    async def async_rerank(
        self,
        queries: List[str],
        document_sets: List[List[RankedDocument]],
        top_k: int = 5
    ) -> List[List[RankedDocument]]:
        """Asynchronous batch reranking."""
        tasks = []

        for query, docs in zip(queries, document_sets):
            task = self._async_score_documents(query, docs, top_k)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def _async_score_documents(
        self,
        query: str,
        documents: List[RankedDocument],
        top_k: int
    ) -> List[RankedDocument]:
        """Score documents asynchronously."""
        # Simulate async scoring
        await asyncio.sleep(0.01)

        # Check cache
        cache_key = f"{query[:30]}_{len(documents)}"
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            return cached_result[:top_k]

        self.cache_misses += 1

        # Score documents
        for doc in documents:
            doc.final_score = random.random()  # Simulated scoring

        # Sort and cache
        documents.sort(key=lambda x: x.final_score, reverse=True)

        # Update cache
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = documents.copy()

        return documents[:top_k]

    def batch_score_with_early_stopping(
        self,
        query: str,
        documents: List[RankedDocument],
        top_k: int = 5,
        confidence_threshold: float = 0.9
    ) -> List[RankedDocument]:
        """
        Batch scoring with early stopping when confidence is high.

        Args:
            confidence_threshold: Stop scoring if top-k confidence exceeds this
        """
        scored_docs = []
        batch_size = 10

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            # Score batch
            for doc in batch:
                doc.final_score = random.random()  # Simulated

            scored_docs.extend(batch)

            # Check if we can stop early
            if len(scored_docs) >= top_k * 2:
                # Sort current results
                scored_docs.sort(key=lambda x: x.final_score, reverse=True)

                # Check confidence of top-k
                if scored_docs[top_k-1].final_score > confidence_threshold:
                    print(f"Early stopping at {len(scored_docs)} documents")
                    break

        # Final sort
        scored_docs.sort(key=lambda x: x.final_score, reverse=True)
        return scored_docs[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

# Test optimized reranking
optimized_reranker = OptimizedReranker()

# Test batch scoring with early stopping
print("Testing Early Stopping:")
large_doc_set = documents * 10  # Create larger set
early_stop_results = optimized_reranker.batch_score_with_early_stopping(
    query,
    large_doc_set.copy(),
    top_k=3,
    confidence_threshold=0.8
)

print(f"Selected top {len(early_stop_results)} documents")

# Test async reranking
async def test_async_reranking():
    """Test asynchronous reranking."""
    queries = [query] * 3
    doc_sets = [documents.copy() for _ in queries]

    start_time = time.time()
    results = await optimized_reranker.async_rerank(queries, doc_sets, top_k=3)
    elapsed = time.time() - start_time

    print(f"\nAsync Reranking:")
    print(f"Processed {len(queries)} queries in {elapsed:.3f}s")
    print(f"Cache stats: {optimized_reranker.get_stats()}")

# Run async test
asyncio.run(test_async_reranking())

# ================================
# Example 7: Production Reranking System
# ================================
print("\n" + "=" * 50)
print("Example 7: Production Reranking System")
print("=" * 50)

class ProductionReranker:
    """Production-ready reranking system with monitoring."""

    def __init__(self):
        self.rerankers = {}
        self.metrics = defaultdict(lambda: {"count": 0, "total_time": 0})
        self.error_count = 0

    def register_reranker(self, name: str, reranker):
        """Register a reranker."""
        self.rerankers[name] = reranker

    def rerank(
        self,
        query: str,
        documents: List[RankedDocument],
        strategy: str = "hybrid",
        top_k: int = 5,
        timeout: float = 1.0
    ) -> Dict[str, Any]:
        """
        Production reranking with monitoring and fallback.

        Args:
            strategy: Reranking strategy to use
            timeout: Maximum time for reranking

        Returns:
            Reranked documents with metadata
        """
        start_time = time.time()

        try:
            # Select reranker
            if strategy == "hybrid":
                results = self._hybrid_reranking(query, documents, top_k)
            elif strategy in self.rerankers:
                results = self._single_reranker(query, documents, strategy, top_k)
            else:
                # Fallback to simple scoring
                results = self._fallback_reranking(documents, top_k)

            # Record metrics
            elapsed = time.time() - start_time
            self.metrics[strategy]["count"] += 1
            self.metrics[strategy]["total_time"] += elapsed

            return {
                "results": results,
                "strategy": strategy,
                "latency": elapsed,
                "success": True
            }

        except Exception as e:
            self.error_count += 1
            print(f"Reranking error: {e}")

            # Fallback
            return {
                "results": documents[:top_k],
                "strategy": "fallback",
                "latency": time.time() - start_time,
                "success": False,
                "error": str(e)
            }

    def _hybrid_reranking(
        self,
        query: str,
        documents: List[RankedDocument],
        top_k: int
    ) -> List[RankedDocument]:
        """Hybrid reranking combining multiple methods."""
        results = []

        # Run multiple rerankers
        for name, reranker in self.rerankers.items():
            try:
                reranked = reranker.rerank(query, documents.copy(), top_k)
                results.append(reranked)
            except:
                continue

        if not results:
            return documents[:top_k]

        # Combine results (simple voting)
        return self._combine_rankings(results, top_k)

    def _single_reranker(
        self,
        query: str,
        documents: List[RankedDocument],
        strategy: str,
        top_k: int
    ) -> List[RankedDocument]:
        """Use single reranker."""
        reranker = self.rerankers[strategy]
        return reranker.rerank(query, documents, top_k)

    def _fallback_reranking(
        self,
        documents: List[RankedDocument],
        top_k: int
    ) -> List[RankedDocument]:
        """Simple fallback reranking."""
        # Just use initial scores
        documents.sort(key=lambda x: x.initial_score, reverse=True)
        return documents[:top_k]

    def _combine_rankings(
        self,
        rankings: List[List[RankedDocument]],
        top_k: int
    ) -> List[RankedDocument]:
        """Combine multiple rankings."""
        doc_scores = defaultdict(float)
        doc_map = {}

        for ranking in rankings:
            for i, doc in enumerate(ranking):
                doc_scores[doc.id] += 1.0 / (i + 1)
                doc_map[doc.id] = doc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        metrics = {}

        for strategy, data in self.metrics.items():
            if data["count"] > 0:
                metrics[strategy] = {
                    "count": data["count"],
                    "avg_latency": data["total_time"] / data["count"],
                    "total_time": data["total_time"]
                }

        metrics["error_count"] = self.error_count
        metrics["total_requests"] = sum(d["count"] for d in self.metrics.values())

        return metrics

# Create production system
production = ProductionReranker()

# Register rerankers
production.register_reranker("cross_encoder", CrossEncoderReranker(model_type="simulated"))
production.register_reranker("mmr", MMRReranker(lambda_param=0.5))
production.register_reranker("personalized", PersonalizedReranker())

# Test different strategies
strategies = ["cross_encoder", "mmr", "hybrid", "invalid_strategy"]

print("Production Reranking Tests:\n")
for strategy in strategies:
    result = production.rerank(
        query,
        documents.copy(),
        strategy=strategy,
        top_k=3
    )

    print(f"Strategy: {strategy}")
    print(f"  Success: {result['success']}")
    print(f"  Latency: {result['latency']:.4f}s")
    print(f"  Results: {len(result['results'])} documents")
    print()

# Show metrics
print("System Metrics:")
for key, value in production.get_metrics().items():
    print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("Advanced Reranking Examples Complete!")
print("=" * 50)