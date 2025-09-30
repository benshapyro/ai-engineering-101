"""
Module 11: Advanced RAG - Solutions

Complete solutions for advanced RAG exercises including hybrid retrieval,
query processing, and sophisticated reranking implementations.

Author: Claude
Date: 2024
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict, Counter
import asyncio
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# ================================
# Solution 1: Hybrid Retriever
# ================================
print("=" * 50)
print("Solution 1: Hybrid Retriever")
print("=" * 50)

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
        self.documents = []
        self.doc_embeddings = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = None
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.index_built = False

    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for both retrieval methods.

        Args:
            documents: List of document strings
        """
        self.documents = documents

        # Build TF-IDF matrix for sparse retrieval
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # Calculate document lengths for BM25
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = np.mean(self.doc_lengths)

        # Generate embeddings for dense retrieval
        self.doc_embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            self.doc_embeddings.append(np.array(response.data[0].embedding))

        self.index_built = True
        print(f"Indexed {len(documents)} documents")

    def bm25_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform BM25 sparse retrieval.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of search results with BM25 scores
        """
        if not self.index_built:
            return []

        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])

        # Calculate BM25 scores
        scores = []
        k1 = 1.2
        b = 0.75

        for i, doc in enumerate(self.documents):
            # Get TF-IDF score
            tfidf_score = self.tfidf_matrix[i].dot(query_vec.T).toarray()[0, 0]

            # Adjust with BM25 formula
            doc_len = self.doc_lengths[i]
            norm_factor = 1 - b + b * (doc_len / self.avg_doc_length)
            bm25_score = tfidf_score * (k1 + 1) / (tfidf_score + k1 * norm_factor)

            scores.append(bm25_score)

        # Get top-k results
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                doc_id=f"doc_{idx}",
                content=self.documents[idx],
                sparse_score=scores[idx]
            ))

        return results

    def dense_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform dense retrieval using embeddings.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of search results with similarity scores
        """
        if not self.index_built:
            return []

        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate cosine similarities
        similarities = []
        for doc_emb in self.doc_embeddings:
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(sim)

        # Get top-k results
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                doc_id=f"doc_{idx}",
                content=self.documents[idx],
                dense_score=similarities[idx]
            ))

        return results

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
        rrf_scores = defaultdict(float)
        doc_map = {}

        # Calculate RRF scores
        for results in results_list:
            for rank, result in enumerate(results, 1):
                rrf_scores[result.doc_id] += 1.0 / (k + rank)
                doc_map[result.doc_id] = result

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Create final results
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_map[doc_id]
            result.final_score = score
            final_results.append(result)

        return final_results

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
        combined_scores = defaultdict(float)
        doc_map = {}

        # Add sparse scores
        for result in sparse_results:
            combined_scores[result.doc_id] += sparse_weight * result.sparse_score
            doc_map[result.doc_id] = result

        # Add dense scores
        for result in dense_results:
            combined_scores[result.doc_id] += dense_weight * result.dense_score
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Create final results
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_map[doc_id]
            result.final_score = score
            final_results.append(result)

        return final_results

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
        # Execute both search methods
        sparse_results = self.bm25_search(query, k=k*2)
        dense_results = self.dense_search(query, k=k*2)

        # Apply chosen fusion method
        if fusion_method == "rrf":
            final_results = self.reciprocal_rank_fusion(
                [sparse_results, dense_results],
                k=60
            )
        else:  # weighted
            final_results = self.weighted_fusion(
                sparse_results,
                dense_results,
                sparse_weight,
                1 - sparse_weight
            )

        return final_results[:k]

# Test implementation
retriever = HybridRetriever()

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing enables computers to understand and generate human text.",
    "Computer vision processes visual information from images and videos.",
    "Reinforcement learning uses reward signals to train intelligent agents."
]

# Index documents
retriever.index_documents(documents)

# Test search
query = "How do neural networks learn from data?"
results = retriever.search(query, k=3, fusion_method="weighted")

print(f"Search results for: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result.doc_id}")
    print(f"   Score: {result.final_score:.3f}")
    print(f"   Content: {result.content[:60]}...")
    print()

# ================================
# Solution 2: Query Understanding Pipeline
# ================================
print("\n" + "=" * 50)
print("Solution 2: Query Understanding Pipeline")
print("=" * 50)

class QueryUnderstandingPipeline:
    """Pipeline for query understanding and processing."""

    def __init__(self):
        """Initialize the pipeline."""
        self.conversation_history = []
        self.entity_cache = {}
        self.query_patterns = {
            "factual": ["what is", "who is", "when did", "where is"],
            "procedural": ["how to", "steps to", "process for"],
            "comparison": ["compare", "difference", "versus", "vs"],
            "causal": ["why", "cause", "reason", "because"]
        }

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query type and intent.

        Args:
            query: Input query

        Returns:
            Classification results with confidence
        """
        query_lower = query.lower()
        classifications = {}

        # Pattern-based classification
        for intent, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                classifications[intent] = score

        # Normalize scores
        total_score = sum(classifications.values())
        if total_score > 0:
            for intent in classifications:
                classifications[intent] /= total_score

        # Get primary intent
        if classifications:
            primary_intent = max(classifications, key=classifications.get)
            confidence = classifications[primary_intent]
        else:
            primary_intent = "general"
            confidence = 0.5

        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "all_intents": classifications
        }

    def extract_entities(self, query: str) -> List[Dict[str, str]]:
        """
        Extract and disambiguate entities.

        Args:
            query: Input query

        Returns:
            List of extracted entities with types
        """
        # Use LLM for entity extraction
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Extract entities from the text. For each entity, provide:
                    - text: the entity text
                    - type: PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, or OTHER
                    Return as JSON array."""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0
        )

        try:
            entities = json.loads(response.choices[0].message.content)

            # Disambiguate entities
            for entity in entities:
                # Check cache for disambiguation
                if entity["text"] in self.entity_cache:
                    entity["disambiguated"] = self.entity_cache[entity["text"]]

            return entities
        except:
            return []

    def expand_query(self, query: str, method: str = "all") -> List[str]:
        """
        Expand query with variations.

        Args:
            query: Original query
            method: Expansion method ('synonyms', 'related', 'all')

        Returns:
            List of expanded queries
        """
        expansions = [query]  # Include original

        if method in ["synonyms", "all"]:
            # Generate synonyms
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 3 synonym variations of the query. Return each on a new line."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.5
            )
            synonyms = response.choices[0].message.content.strip().split('\n')
            expansions.extend([s.strip() for s in synonyms if s.strip()])

        if method in ["related", "all"]:
            # Generate related queries
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate 2 related queries. Return each on a new line."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.5
            )
            related = response.choices[0].message.content.strip().split('\n')
            expansions.extend([r.strip() for r in related if r.strip()])

        return list(set(expansions))  # Remove duplicates

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        # Check for complexity indicators
        if " and " in query.lower() or ", " in query:
            # Multi-part query
            parts = re.split(r'\s+and\s+|,\s+', query)
            return [p.strip() for p in parts if p.strip()]

        # Use LLM for complex decomposition
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Break down the query into 2-4 simpler sub-queries if it's complex. Return each on a new line. If simple, return the original."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        sub_queries = response.choices[0].message.content.strip().split('\n')
        return [q.strip() for q in sub_queries if q.strip()]

    def resolve_coreferences(self, query: str) -> str:
        """
        Resolve pronouns and references using context.

        Args:
            query: Query with potential coreferences

        Returns:
            Resolved query
        """
        pronouns = ["it", "this", "that", "they", "them"]
        query_lower = query.lower()

        # Check for pronouns
        has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)

        if has_pronouns and self.conversation_history:
            # Get last relevant context
            last_context = self.conversation_history[-1] if self.conversation_history else ""

            # Use LLM to resolve
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Resolve pronouns in the query using this context: {last_context}"
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content.strip()

        return query

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query processing pipeline.

        Args:
            query: Input query

        Returns:
            Processed query with all enhancements
        """
        # Resolve coreferences
        resolved_query = self.resolve_coreferences(query)

        # Classify query
        classification = self.classify_query(resolved_query)

        # Extract entities
        entities = self.extract_entities(resolved_query)

        # Decompose if complex
        sub_queries = self.decompose_query(resolved_query)

        # Expand query
        expansions = self.expand_query(resolved_query)

        # Update conversation history
        self.conversation_history.append(query)
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

        return {
            "original": query,
            "resolved": resolved_query,
            "classification": classification,
            "entities": entities,
            "sub_queries": sub_queries,
            "expansions": expansions[:5]  # Limit expansions
        }

# Test implementation
pipeline = QueryUnderstandingPipeline()

# Test queries
test_queries = [
    "Compare BERT and GPT models",
    "How does it work?",  # Coreference
    "What are the steps to train a neural network and evaluate its performance?"
]

for query in test_queries:
    result = pipeline.process_query(query)
    print(f"\nQuery: {query}")
    print(f"Resolved: {result['resolved']}")
    print(f"Intent: {result['classification']['primary_intent']} (confidence: {result['classification']['confidence']:.2f})")
    print(f"Entities: {result['entities']}")
    print(f"Sub-queries: {result['sub_queries']}")
    print(f"Expansions: {result['expansions'][:3]}")

# ================================
# Solution 3: Custom Reranking Algorithm
# ================================
print("\n" + "=" * 50)
print("Solution 3: Custom Reranking Algorithm")
print("=" * 50)

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
        self.cache = {}

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
        # Check cache
        cache_key = f"{query[:30]}_{document.id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Use LLM for scoring
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Rate relevance from 0-10. Just the number."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nDocument: {document.content[:300]}"
                }
            ],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip()) / 10.0
        except:
            score = 0.5

        self.cache[cache_key] = score
        return score

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
        if not documents:
            return []

        selected = []
        remaining = documents.copy()

        # Generate query embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_emb = np.array(response.data[0].embedding)

        # Generate document embeddings
        doc_embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc.content
            )
            doc_embeddings.append(np.array(response.data[0].embedding))

        # Select first document (most relevant)
        relevance_scores = []
        for emb in doc_embeddings:
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            relevance_scores.append(sim)

        best_idx = np.argmax(relevance_scores)
        selected.append(remaining.pop(best_idx))
        selected_embs = [doc_embeddings.pop(best_idx)]

        # Iteratively select diverse documents
        while len(selected) < k and remaining:
            mmr_scores = []

            for i, (doc, emb) in enumerate(zip(remaining, doc_embeddings)):
                # Relevance to query
                relevance = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))

                # Maximum similarity to selected documents
                max_sim = 0
                for sel_emb in selected_embs:
                    sim = np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((i, mmr))

            # Select best MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = mmr_scores[0][0]

            selected.append(remaining.pop(best_idx))
            selected_embs.append(doc_embeddings.pop(best_idx))

        return selected

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
        for doc in documents:
            # Interest matching
            interest_score = 0
            if user_profile.interests:
                doc_lower = doc.content.lower()
                matches = sum(1 for interest in user_profile.interests if interest.lower() in doc_lower)
                interest_score = matches / len(user_profile.interests)

            # Expertise level adjustment
            expertise_scores = {
                "beginner": 0.8 if "basic" in doc.content.lower() or "introduction" in doc.content.lower() else 0.3,
                "intermediate": 0.5,
                "expert": 0.8 if "advanced" in doc.content.lower() or "technical" in doc.content.lower() else 0.3
            }
            expertise_score = expertise_scores.get(user_profile.expertise_level, 0.5)

            # Novelty score (avoid repetition)
            novelty_score = 1.0
            if user_profile.interaction_history:
                for hist in user_profile.interaction_history:
                    if doc.id in hist:
                        novelty_score = 0.3
                        break

            # Combine scores
            doc.rerank_score = (
                0.4 * interest_score +
                0.3 * expertise_score +
                0.2 * novelty_score +
                0.1 * doc.initial_score
            )

        # Sort by personalized score
        documents.sort(key=lambda x: x.rerank_score, reverse=True)
        return documents[:k]

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
        for i, doc in enumerate(documents):
            # Apply position decay
            position_factor = 1.0 / (1 + i * 0.1)  # Decay factor

            # Check if query terms appear early in document
            query_words = query.lower().split()
            doc_words = doc.content.lower().split()

            early_match_bonus = 0
            for qw in query_words:
                for j, dw in enumerate(doc_words[:20]):  # Check first 20 words
                    if qw in dw:
                        early_match_bonus += 1.0 / (j + 1)

            doc.rerank_score = doc.initial_score * position_factor + early_match_bonus * 0.1

        documents.sort(key=lambda x: x.rerank_score, reverse=True)
        return documents[:k]

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
        current_time = time.time()

        for doc in documents:
            # Extract timestamp from metadata if available
            timestamp = doc.metadata.get("timestamp", current_time - 86400)  # Default to 1 day old

            # Calculate freshness score (exponential decay)
            age_days = (current_time - timestamp) / 86400
            freshness_score = np.exp(-age_days / 30)  # 30-day half-life

            # Combine with relevance
            doc.rerank_score = (1 - freshness_weight) * doc.initial_score + freshness_weight * freshness_score

        documents.sort(key=lambda x: x.rerank_score, reverse=True)
        return documents[:k]

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
        if strategies is None:
            strategies = ["cross_encoder", "position"]

        ensemble_scores = defaultdict(float)
        strategy_weights = {
            "cross_encoder": 0.4,
            "mmr": 0.3,
            "personalized": 0.2,
            "position": 0.05,
            "temporal": 0.05
        }

        # Apply selected strategies
        if "cross_encoder" in strategies:
            for doc in documents:
                score = self.cross_encoder_score(query, doc)
                ensemble_scores[doc.id] += strategy_weights["cross_encoder"] * score

        if "mmr" in strategies:
            mmr_docs = self.mmr_rerank(query, documents.copy(), k=len(documents))
            for i, doc in enumerate(mmr_docs):
                ensemble_scores[doc.id] += strategy_weights["mmr"] * (1.0 / (i + 1))

        if "personalized" in strategies and user_profile:
            pers_docs = self.personalized_rerank(documents.copy(), user_profile, k=len(documents))
            for i, doc in enumerate(pers_docs):
                ensemble_scores[doc.id] += strategy_weights["personalized"] * (1.0 / (i + 1))

        if "position" in strategies:
            pos_docs = self.position_aware_rerank(query, documents.copy(), k=len(documents))
            for i, doc in enumerate(pos_docs):
                ensemble_scores[doc.id] += strategy_weights["position"] * (1.0 / (i + 1))

        if "temporal" in strategies:
            temp_docs = self.temporal_rerank(documents.copy(), k=len(documents))
            for i, doc in enumerate(temp_docs):
                ensemble_scores[doc.id] += strategy_weights["temporal"] * (1.0 / (i + 1))

        # Apply ensemble scores to documents
        for doc in documents:
            doc.rerank_score = ensemble_scores[doc.id]

        # Sort by ensemble score
        documents.sort(key=lambda x: x.rerank_score, reverse=True)
        return documents[:k]

# Test implementation
reranker = CustomReranker()

# Create test documents
test_docs = [
    Document("doc1", "Introduction to machine learning basics", 0.8, {"timestamp": time.time()}),
    Document("doc2", "Advanced deep learning techniques and architectures", 0.7, {"timestamp": time.time() - 86400}),
    Document("doc3", "Natural language processing applications", 0.6, {"timestamp": time.time() - 172800}),
    Document("doc4", "Computer vision fundamentals guide", 0.5, {"timestamp": time.time() - 259200}),
    Document("doc5", "Reinforcement learning with reward signals", 0.4, {"timestamp": time.time() - 345600})
]

# Create user profile
user = UserProfile(
    interests=["machine learning", "deep learning"],
    expertise_level="intermediate",
    interaction_history=["doc3"]
)

# Test ensemble reranking
query = "machine learning tutorial for beginners"
reranked = reranker.ensemble_rerank(
    query,
    test_docs.copy(),
    user_profile=user,
    strategies=["cross_encoder", "mmr", "personalized", "position"],
    k=3
)

print(f"\nReranked results for: {query}")
for i, doc in enumerate(reranked, 1):
    print(f"{i}. {doc.id}: Score={doc.rerank_score:.3f}")
    print(f"   {doc.content}")

# ================================
# Solution 4: Production RAG Pipeline
# ================================
print("\n" + "=" * 50)
print("Solution 4: Production RAG Pipeline")
print("=" * 50)

class ProductionRAGPipeline:
    """Production-ready RAG pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.metrics = defaultdict(int)
        self.documents = {}
        self.embeddings = {}
        self.cache = {}
        self.query_processor = QueryUnderstandingPipeline()
        self.retriever = HybridRetriever()
        self.reranker = CustomReranker()

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
        # Generate document ID
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Chunk document
        chunks = self._chunk_document(content)

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embeddings.append(response.data[0].embedding)

        # Store document
        self.documents[doc_id] = {
            "content": content,
            "chunks": chunks,
            "metadata": metadata
        }
        self.embeddings[doc_id] = embeddings

        self.metrics["documents_ingested"] += 1
        return doc_id

    def _chunk_document(self, content: str, chunk_size: int = 500) -> List[str]:
        """Chunk document into smaller pieces."""
        words = content.split()
        chunks = []

        for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

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
        # Check cache
        cache_key = f"{query}_{k}_{str(filters)}"
        if cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            return self.cache[cache_key]

        # Process query
        processed = self.query_processor.process_query(query)

        # Retrieve using hybrid search
        all_docs = list(self.documents.values())
        self.retriever.index_documents([d["content"] for d in all_docs])
        search_results = self.retriever.search(processed["resolved"], k=k)

        # Convert to Document objects
        documents = []
        for result in search_results:
            doc_idx = int(result.doc_id.split("_")[1])
            doc_id = list(self.documents.keys())[doc_idx]
            doc_data = self.documents[doc_id]

            # Apply filters
            if filters:
                match = all(
                    doc_data["metadata"].get(key) == value
                    for key, value in filters.items()
                )
                if not match:
                    continue

            documents.append(Document(
                id=doc_id,
                content=doc_data["content"],
                initial_score=result.final_score,
                metadata=doc_data["metadata"]
            ))

        # Cache results
        self.cache[cache_key] = documents
        self.metrics["cache_misses"] += 1

        return documents

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
        try:
            # Auto-select strategy based on query type
            if strategy == "auto":
                query_info = self.query_processor.classify_query(query)
                if query_info["primary_intent"] == "comparison":
                    strategy = "mmr"
                else:
                    strategy = "cross_encoder"

            # Apply reranking
            if strategy == "mmr":
                reranked = self.reranker.mmr_rerank(query, documents, k=5)
            elif strategy == "cross_encoder":
                for doc in documents:
                    doc.rerank_score = self.reranker.cross_encoder_score(query, doc)
                documents.sort(key=lambda x: x.rerank_score, reverse=True)
                reranked = documents[:5]
            else:
                reranked = documents[:5]

            self.metrics["rerank_success"] += 1
            return reranked

        except Exception as e:
            print(f"Reranking error: {e}")
            self.metrics["rerank_errors"] += 1
            return documents[:5]  # Fallback

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
        # Format context
        context = "\n\n".join([
            f"[{i+1}] {doc.content}"
            for i, doc in enumerate(documents)
        ])

        # Generate response
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on the context. Cite sources using [number] format."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=300
        )

        answer = response.choices[0].message.content

        # Extract citations
        citations = re.findall(r'\[(\d+)\]', answer)
        cited_docs = [documents[int(c)-1].id for c in citations if int(c) <= len(documents)]

        return {
            "answer": answer,
            "citations": cited_docs,
            "sources": [{"id": doc.id, "content": doc.content[:200]} for doc in documents]
        }

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
        start_time = time.time()

        # Retrieve documents
        documents = await self.retrieve(query, k=10)

        # Rerank results
        reranked = await self.rerank(query, documents)

        # Generate response
        response = await self.generate_response(query, reranked)

        # Calculate metrics
        latency = time.time() - start_time
        self.metrics["total_queries"] += 1
        self.metrics["total_latency"] += latency

        return {
            "query": query,
            "response": response,
            "latency": latency,
            "documents_retrieved": len(documents),
            "documents_used": len(reranked)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics.

        Returns:
            Performance metrics
        """
        total_queries = self.metrics["total_queries"]
        avg_latency = self.metrics["total_latency"] / total_queries if total_queries > 0 else 0
        cache_rate = self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"]) if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0

        return {
            "total_queries": total_queries,
            "avg_latency": avg_latency,
            "cache_hit_rate": cache_rate,
            "documents_ingested": self.metrics["documents_ingested"],
            "rerank_errors": self.metrics["rerank_errors"]
        }

    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check.

        Returns:
            Health status of components
        """
        health = {
            "document_store": len(self.documents) > 0,
            "embeddings": len(self.embeddings) > 0,
            "cache": True,  # Always healthy
            "query_processor": True,
            "retriever": True,
            "reranker": True
        }

        # Test LLM availability
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            health["llm"] = True
        except:
            health["llm"] = False

        return health

# Test implementation
config = {
    "retrieval_k": 10,
    "rerank_k": 5,
    "cache_ttl": 3600,
    "max_retries": 3
}

pipeline = ProductionRAGPipeline(config)

# Test pipeline (async)
async def test_pipeline():
    # Ingest documents
    for doc in documents:
        doc_id = await pipeline.ingest_document(
            doc,
            {"source": "test", "timestamp": time.time()}
        )
        print(f"Ingested document: {doc_id}")

    # Query pipeline
    result = await pipeline.query("What is deep learning?")
    print(f"\nQuery: {result['query']}")
    print(f"Response: {result['response']['answer'][:200]}...")
    print(f"Latency: {result['latency']:.3f}s")
    print(f"Documents used: {result['documents_used']}")

    # Check metrics
    metrics = pipeline.get_metrics()
    print(f"\nMetrics: {metrics}")

    # Check health
    health = pipeline.health_check()
    print(f"Health: {health}")

# Run test
asyncio.run(test_pipeline())

# ================================
# Solution 5: Evaluation Metrics
# ================================
print("\n" + "=" * 50)
print("Solution 5: Evaluation Metrics")
print("=" * 50)

class RAGEvaluator:
    """Evaluation system for RAG pipelines."""

    def __init__(self):
        """Initialize evaluator."""
        self.results_cache = {}

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
        if k == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)

        true_positives = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        precision = true_positives / k

        return precision

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
        if len(relevant) == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)

        true_positives = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        recall = true_positives / len(relevant)

        return recall

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
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            rr = 0.0

            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    rr = 1.0 / rank
                    break

            reciprocal_ranks.append(rr)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

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
        if k == 0:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1

        # Calculate IDCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += rel / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ndcg

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
        # Use LLM to evaluate faithfulness
        sources_text = "\n".join(source_documents)

        response_llm = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Rate how well the response is supported by the sources from 0-10. Just the number."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{sources_text[:1000]}\n\nResponse:\n{response}"
                }
            ],
            temperature=0
        )

        try:
            score = float(response_llm.choices[0].message.content.strip()) / 10.0
        except:
            score = 0.5

        return score

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
        # Use LLM to evaluate relevance
        response_llm = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Rate how well the response answers the query from 0-10. Just the number."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nResponse: {response}"
                }
            ],
            temperature=0
        )

        try:
            score = float(response_llm.choices[0].message.content.strip()) / 10.0
        except:
            score = 0.5

        return score

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
        results_a = []
        results_b = []

        for query in test_queries:
            # Run both pipelines
            result_a = asyncio.run(pipeline_a.query(query))
            result_b = asyncio.run(pipeline_b.query(query))

            results_a.append(result_a)
            results_b.append(result_b)

        # Calculate metrics
        comparison = {}

        if "latency" in metrics:
            comparison["latency_a"] = np.mean([r["latency"] for r in results_a])
            comparison["latency_b"] = np.mean([r["latency"] for r in results_b])
            comparison["latency_winner"] = "A" if comparison["latency_a"] < comparison["latency_b"] else "B"

        if "relevance" in metrics:
            relevance_a = []
            relevance_b = []
            for query, res_a, res_b in zip(test_queries, results_a, results_b):
                relevance_a.append(self.evaluate_relevance(query, res_a["response"]["answer"]))
                relevance_b.append(self.evaluate_relevance(query, res_b["response"]["answer"]))

            comparison["relevance_a"] = np.mean(relevance_a)
            comparison["relevance_b"] = np.mean(relevance_b)
            comparison["relevance_winner"] = "A" if comparison["relevance_a"] > comparison["relevance_b"] else "B"

        return comparison

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
        report = []
        report.append("=" * 50)
        report.append("RAG Evaluation Report")
        report.append("=" * 50)
        report.append("")

        # Retrieval metrics
        if "retrieval" in evaluation_results:
            report.append("Retrieval Metrics:")
            for metric, value in evaluation_results["retrieval"].items():
                report.append(f"  {metric}: {value:.3f}")
            report.append("")

        # Generation metrics
        if "generation" in evaluation_results:
            report.append("Generation Metrics:")
            for metric, value in evaluation_results["generation"].items():
                report.append(f"  {metric}: {value:.3f}")
            report.append("")

        # System metrics
        if "system" in evaluation_results:
            report.append("System Metrics:")
            for metric, value in evaluation_results["system"].items():
                report.append(f"  {metric}: {value}")
            report.append("")

        # Recommendations
        report.append("Recommendations:")
        if evaluation_results.get("retrieval", {}).get("precision", 0) < 0.5:
            report.append("  - Improve retrieval precision")
        if evaluation_results.get("generation", {}).get("faithfulness", 0) < 0.7:
            report.append("  - Enhance response faithfulness")
        if evaluation_results.get("system", {}).get("avg_latency", 0) > 1.0:
            report.append("  - Optimize for latency")

        return "\n".join(report)

# Test implementation
evaluator = RAGEvaluator()

# Test retrieval metrics
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = ["doc1", "doc3", "doc6"]

precision = evaluator.precision_at_k(retrieved, relevant, k=3)
recall = evaluator.recall_at_k(retrieved, relevant, k=5)

print(f"Precision@3: {precision:.3f}")
print(f"Recall@5: {recall:.3f}")

# Test MRR
retrieved_lists = [
    ["doc1", "doc2", "doc3"],
    ["doc4", "doc1", "doc5"],
    ["doc2", "doc3", "doc1"]
]
relevant_lists = [
    ["doc1", "doc3"],
    ["doc1", "doc2"],
    ["doc1"]
]

mrr = evaluator.mean_reciprocal_rank(retrieved_lists, relevant_lists)
print(f"MRR: {mrr:.3f}")

# Test NDCG
relevance_scores = {
    "doc1": 3.0,
    "doc2": 1.0,
    "doc3": 2.0,
    "doc4": 0.0,
    "doc5": 1.0
}

ndcg = evaluator.ndcg_at_k(retrieved, relevance_scores, k=3)
print(f"NDCG@3: {ndcg:.3f}")

# Generate report
evaluation_results = {
    "retrieval": {
        "precision@3": precision,
        "recall@5": recall,
        "mrr": mrr,
        "ndcg@3": ndcg
    },
    "generation": {
        "faithfulness": 0.85,
        "relevance": 0.90,
        "coherence": 0.88
    },
    "system": {
        "avg_latency": 0.5,
        "cache_hit_rate": 0.4,
        "throughput": "100 QPS"
    }
}

report = evaluator.generate_report(evaluation_results)
print(f"\n{report}")

print("\n" + "=" * 50)
print("All Solutions Complete!")
print("=" * 50)