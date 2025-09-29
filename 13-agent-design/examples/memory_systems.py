"""
Module 13: Agent Design
Memory Systems Examples

This file demonstrates various memory systems for agents:
1. Working memory implementation
2. Long-term memory with vector store
3. Episodic memory for experiences
4. Semantic memory for facts
5. Procedural memory for skills
6. Memory consolidation
7. Complete memory system

Each example shows progressively more sophisticated memory patterns.
"""

import os
import json
import time
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from dotenv import load_dotenv

load_dotenv()


# Example 1: Working Memory Implementation
print("=" * 50)
print("Example 1: Working Memory Implementation")
print("=" * 50)


@dataclass
class MemoryItem:
    """Single item in working memory."""
    content: Any
    timestamp: datetime
    importance: float = 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class WorkingMemory:
    """Short-term memory for current task context."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # Miller's Law: 7±2 items
        self.memory = deque(maxlen=capacity)
        self.attention_weights = {}
        self.focus_item = None

    def add(self, content: Any, importance: float = 1.0, tags: List[str] = None):
        """Add item to working memory."""
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or []
        )

        # If at capacity, remove least important item
        if len(self.memory) >= self.capacity:
            self._evict_least_important()

        self.memory.append(item)
        self.attention_weights[id(item)] = importance

        print(f"Added to working memory: {str(content)[:50]}...")
        print(f"  Importance: {importance:.2f}")
        print(f"  Current capacity: {len(self.memory)}/{self.capacity}")

    def _evict_least_important(self):
        """Remove least important item when at capacity."""
        if not self.memory:
            return

        # Calculate scores based on importance and recency
        scores = []
        for item in self.memory:
            recency = 1.0 / (1 + (datetime.now() - item.timestamp).seconds)
            score = item.importance * recency * (1 + item.access_count * 0.1)
            scores.append(score)

        # Remove item with lowest score
        min_idx = scores.index(min(scores))
        evicted = self.memory[min_idx]
        del self.memory[min_idx]

        if id(evicted) in self.attention_weights:
            del self.attention_weights[id(evicted)]

        print(f"Evicted from working memory: {str(evicted.content)[:30]}...")

    def retrieve(self, query: str = None, tags: List[str] = None) -> List[MemoryItem]:
        """Retrieve items from working memory."""
        results = []

        for item in self.memory:
            # Update access info
            item.access_count += 1
            item.last_accessed = datetime.now()

            # Filter by tags if specified
            if tags and not any(tag in item.tags for tag in tags):
                continue

            # Simple text matching if query provided
            if query and query.lower() not in str(item.content).lower():
                continue

            results.append(item)

        return sorted(results, key=lambda x: x.importance, reverse=True)

    def focus_on(self, item: MemoryItem):
        """Set focus on a specific item."""
        if item in self.memory:
            self.focus_item = item
            item.importance *= 1.5  # Boost importance of focused item
            print(f"Focused on: {str(item.content)[:50]}...")

    def get_context(self, max_items: int = 3) -> List[MemoryItem]:
        """Get most relevant items for current context."""
        # Sort by importance and recency
        sorted_items = sorted(
            self.memory,
            key=lambda x: x.importance * (1 + x.access_count * 0.1),
            reverse=True
        )

        context_items = sorted_items[:max_items]

        if self.focus_item and self.focus_item not in context_items:
            context_items.insert(0, self.focus_item)

        return context_items

    def decay(self, decay_rate: float = 0.1):
        """Apply decay to importance over time."""
        for item in self.memory:
            age = (datetime.now() - item.timestamp).seconds
            decay_factor = np.exp(-decay_rate * age / 60)  # Decay per minute
            item.importance *= decay_factor

    def clear(self):
        """Clear working memory."""
        self.memory.clear()
        self.attention_weights.clear()
        self.focus_item = None
        print("Working memory cleared")

    def get_summary(self) -> Dict:
        """Get summary of working memory state."""
        if not self.memory:
            return {"items": 0, "total_importance": 0}

        return {
            "items": len(self.memory),
            "capacity_used": f"{len(self.memory)}/{self.capacity}",
            "total_importance": sum(item.importance for item in self.memory),
            "avg_access_count": np.mean([item.access_count for item in self.memory]),
            "focused_item": str(self.focus_item.content)[:30] if self.focus_item else None
        }


# Example usage
wm = WorkingMemory(capacity=5)
wm.add("Task: Analyze customer feedback", importance=2.0, tags=["task"])
wm.add("Context: Q4 2023 data", importance=1.5, tags=["context"])
wm.add("Tool: Sentiment analyzer", importance=1.0, tags=["tool"])
wm.add("Insight: 80% positive feedback", importance=1.8, tags=["insight"])
wm.add("Next step: Generate report", importance=1.2, tags=["task"])
wm.add("New data point", importance=0.5)  # Will trigger eviction

context = wm.get_context()
print(f"\nCurrent context ({len(context)} items):")
for item in context:
    print(f"  - {item.content} (importance: {item.importance:.2f})")

print(f"\nMemory summary: {wm.get_summary()}")


# Example 2: Long-term Memory with Vector Store
print("\n" + "=" * 50)
print("Example 2: Long-term Memory with Vector Store")
print("=" * 50)


class VectorMemoryStore:
    """Vector-based memory store for semantic search."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.memories = []
        self.embeddings = []
        self.index = {}

    def _generate_embedding(self, content: str) -> np.ndarray:
        """Generate embedding for content (simplified)."""
        # In practice, would use actual embedding model
        # This is a deterministic pseudo-embedding
        hash_val = hashlib.md5(content.encode()).hexdigest()

        # Convert hash to vector
        embedding = []
        for i in range(0, len(hash_val), 2):
            val = int(hash_val[i:i+2], 16) / 255.0
            embedding.append(val)

        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding.extend([0.5] * (self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        return np.array(embedding)

    def store(self, content: str, metadata: Dict = None) -> str:
        """Store content with embedding."""
        memory_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]

        # Generate embedding
        embedding = self._generate_embedding(content)

        # Store memory
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(),
            "access_count": 0
        }

        self.memories.append(memory)
        self.embeddings.append(embedding)
        self.index[memory_id] = len(self.memories) - 1

        print(f"Stored in long-term memory: {content[:50]}...")
        print(f"  Memory ID: {memory_id}")

        return memory_id

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve k most similar memories."""
        if not self.embeddings:
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.array(self.embeddings)
        )[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Retrieve memories
        results = []
        for idx in top_indices:
            if idx < len(self.memories):
                memory = self.memories[idx].copy()
                memory["similarity"] = similarities[idx]
                memory["access_count"] += 1
                results.append(memory)

        print(f"Retrieved {len(results)} memories for query: {query[:50]}...")

        return results

    def update(self, memory_id: str, updates: Dict):
        """Update existing memory."""
        if memory_id in self.index:
            idx = self.index[memory_id]
            self.memories[idx].update(updates)
            print(f"Updated memory: {memory_id}")
        else:
            print(f"Memory not found: {memory_id}")

    def forget(self, memory_id: str):
        """Remove a memory."""
        if memory_id in self.index:
            idx = self.index[memory_id]
            del self.memories[idx]
            del self.embeddings[idx]
            del self.index[memory_id]

            # Rebuild index
            self.index = {m["id"]: i for i, m in enumerate(self.memories)}
            print(f"Forgot memory: {memory_id}")
        else:
            print(f"Memory not found: {memory_id}")

    def consolidate(self, similarity_threshold: float = 0.9):
        """Consolidate similar memories."""
        if len(self.embeddings) < 2:
            return

        embeddings_array = np.array(self.embeddings)
        similarities = cosine_similarity(embeddings_array)

        consolidated = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i, j] > similarity_threshold:
                    consolidated.append((i, j, similarities[i, j]))

        print(f"Found {len(consolidated)} similar memory pairs to consolidate")

        # Merge similar memories (keep the more recent one)
        for i, j, sim in consolidated:
            if i < len(self.memories) and j < len(self.memories):
                older = i if self.memories[i]["timestamp"] < self.memories[j]["timestamp"] else j
                newer = j if older == i else i

                # Merge metadata
                self.memories[newer]["metadata"].update(self.memories[older]["metadata"])

                # Remove older memory
                self.forget(self.memories[older]["id"])


# Example usage
ltm = VectorMemoryStore()

# Store various memories
ltm.store("Python is a programming language", {"type": "fact", "domain": "programming"})
ltm.store("Machine learning uses Python extensively", {"type": "fact", "domain": "ML"})
ltm.store("Completed data analysis task", {"type": "experience", "success": True})
ltm.store("Python is widely used in data science", {"type": "fact", "domain": "data"})

# Retrieve similar memories
results = ltm.retrieve("Python programming", k=3)
print("\nRetrieved memories:")
for mem in results:
    print(f"  - {mem['content']} (similarity: {mem['similarity']:.3f})")

# Consolidate similar memories
ltm.consolidate(similarity_threshold=0.8)


# Example 3: Episodic Memory for Experiences
print("\n" + "=" * 50)
print("Example 3: Episodic Memory for Experiences")
print("=" * 50)


@dataclass
class Episode:
    """Represents a single episodic memory."""
    context: str
    action: str
    outcome: str
    success: bool
    timestamp: datetime
    emotional_valence: float  # -1 (negative) to 1 (positive)
    importance: float
    learned_lessons: List[str] = field(default_factory=list)


class EpisodicMemory:
    """Memory system for storing and learning from experiences."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)

    def record_episode(self,
                       context: str,
                       action: str,
                       outcome: str,
                       success: bool,
                       emotional_valence: float = 0.0,
                       importance: float = 1.0) -> Episode:
        """Record a new episodic memory."""
        episode = Episode(
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            timestamp=datetime.now(),
            emotional_valence=emotional_valence,
            importance=importance
        )

        # Extract lessons
        lessons = self._extract_lessons(episode)
        episode.learned_lessons = lessons

        # Store episode
        self.episodes.append(episode)

        # Update patterns
        pattern_key = self._get_pattern_key(context, action)
        if success:
            self.success_patterns[pattern_key].append(episode)
        else:
            self.failure_patterns[pattern_key].append(episode)

        print(f"Recorded episode: {action[:50]}...")
        print(f"  Success: {success}")
        print(f"  Emotional valence: {emotional_valence:.2f}")
        print(f"  Lessons learned: {len(lessons)}")

        return episode

    def _extract_lessons(self, episode: Episode) -> List[str]:
        """Extract lessons from an episode."""
        lessons = []

        if episode.success:
            lessons.append(f"Action '{episode.action}' works in context '{episode.context}'")
        else:
            lessons.append(f"Avoid '{episode.action}' in context '{episode.context}'")

        # Emotional learning
        if abs(episode.emotional_valence) > 0.5:
            emotion = "positive" if episode.emotional_valence > 0 else "negative"
            lessons.append(f"This type of action has {emotion} emotional impact")

        return lessons

    def _get_pattern_key(self, context: str, action: str) -> str:
        """Generate pattern key for indexing."""
        # Simplified pattern extraction
        context_key = context.split()[0] if context else "unknown"
        action_key = action.split()[0] if action else "unknown"
        return f"{context_key}:{action_key}"

    def recall_similar_episodes(self,
                               context: str,
                               action: str = None,
                               k: int = 5) -> List[Episode]:
        """Recall episodes similar to current situation."""
        scores = []

        for episode in self.episodes:
            score = 0

            # Context similarity (simplified)
            if context.lower() in episode.context.lower():
                score += 2

            # Action similarity
            if action and action.lower() in episode.action.lower():
                score += 1

            # Boost recent episodes
            age = (datetime.now() - episode.timestamp).days
            recency_bonus = 1.0 / (1 + age)
            score += recency_bonus

            # Weight by importance
            score *= episode.importance

            scores.append((score, episode))

        # Sort and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [episode for _, episode in scores[:k]]

        print(f"Recalled {len(results)} similar episodes")

        return results

    def predict_outcome(self, context: str, action: str) -> Dict:
        """Predict likely outcome based on past episodes."""
        pattern_key = self._get_pattern_key(context, action)

        successes = self.success_patterns.get(pattern_key, [])
        failures = self.failure_patterns.get(pattern_key, [])

        total = len(successes) + len(failures)

        if total == 0:
            # No direct experience, check similar episodes
            similar = self.recall_similar_episodes(context, action, k=10)
            successes = [ep for ep in similar if ep.success]
            failures = [ep for ep in similar if not ep.success]
            total = len(similar)

        if total == 0:
            return {
                "prediction": "unknown",
                "confidence": 0,
                "success_rate": 0.5
            }

        success_rate = len(successes) / total

        # Calculate average emotional valence
        all_episodes = successes + failures
        avg_valence = np.mean([ep.emotional_valence for ep in all_episodes])

        prediction = {
            "prediction": "success" if success_rate > 0.5 else "failure",
            "confidence": abs(success_rate - 0.5) * 2,  # 0 to 1
            "success_rate": success_rate,
            "expected_emotional_valence": avg_valence,
            "based_on_episodes": total
        }

        print(f"Prediction for '{action}' in '{context}':")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Success rate: {prediction['success_rate']:.2%}")

        return prediction

    def get_learned_strategies(self) -> List[Dict]:
        """Extract learned strategies from episodic memory."""
        strategies = []

        # Analyze success patterns
        for pattern_key, episodes in self.success_patterns.items():
            if len(episodes) >= 3:  # Need multiple successes to form strategy
                context, action = pattern_key.split(":")
                strategy = {
                    "context": context,
                    "action": action,
                    "success_count": len(episodes),
                    "confidence": len(episodes) / (len(episodes) +
                                 len(self.failure_patterns.get(pattern_key, [])))
                }
                strategies.append(strategy)

        # Sort by confidence
        strategies.sort(key=lambda x: x["confidence"], reverse=True)

        return strategies

    def emotional_summary(self) -> Dict:
        """Get emotional summary of experiences."""
        if not self.episodes:
            return {"average_valence": 0, "emotional_range": 0}

        valences = [ep.emotional_valence for ep in self.episodes]

        return {
            "average_valence": np.mean(valences),
            "emotional_range": np.max(valences) - np.min(valences),
            "positive_experiences": sum(1 for v in valences if v > 0.3),
            "negative_experiences": sum(1 for v in valences if v < -0.3),
            "neutral_experiences": sum(1 for v in valences if abs(v) <= 0.3)
        }


# Example usage
episodic = EpisodicMemory()

# Record various episodes
episodic.record_episode(
    context="customer complaint",
    action="apologize and offer refund",
    outcome="customer satisfied",
    success=True,
    emotional_valence=0.8,
    importance=2.0
)

episodic.record_episode(
    context="customer complaint",
    action="explain policy without empathy",
    outcome="customer escalated",
    success=False,
    emotional_valence=-0.9,
    importance=2.5
)

episodic.record_episode(
    context="technical issue",
    action="restart system",
    outcome="problem resolved",
    success=True,
    emotional_valence=0.5,
    importance=1.0
)

episodic.record_episode(
    context="customer complaint",
    action="apologize and offer refund",
    outcome="customer satisfied",
    success=True,
    emotional_valence=0.7,
    importance=1.8
)

# Predict outcome for new situation
prediction = episodic.predict_outcome("customer complaint", "apologize and offer refund")

# Get learned strategies
strategies = episodic.get_learned_strategies()
print("\nLearned strategies:")
for strategy in strategies:
    print(f"  Context: {strategy['context']}, Action: {strategy['action']}")
    print(f"    Confidence: {strategy['confidence']:.2%}")

# Emotional summary
print(f"\nEmotional summary: {episodic.emotional_summary()}")


# Example 4: Semantic Memory for Facts
print("\n" + "=" * 50)
print("Example 4: Semantic Memory for Facts")
print("=" * 50)


class SemanticMemory:
    """Memory system for facts and general knowledge."""

    def __init__(self):
        self.facts = {}
        self.concepts = defaultdict(dict)
        self.relationships = defaultdict(list)
        self.confidence_scores = {}

    def store_fact(self,
                   fact: str,
                   category: str = "general",
                   confidence: float = 1.0,
                   source: str = None):
        """Store a fact in semantic memory."""
        fact_id = hashlib.md5(fact.encode()).hexdigest()[:12]

        self.facts[fact_id] = {
            "content": fact,
            "category": category,
            "confidence": confidence,
            "source": source,
            "stored_at": datetime.now(),
            "access_count": 0,
            "last_accessed": None
        }

        self.confidence_scores[fact_id] = confidence

        # Extract and store concepts
        concepts = self._extract_concepts(fact)
        for concept in concepts:
            self.concepts[concept][fact_id] = fact

        print(f"Stored fact: {fact[:50]}...")
        print(f"  Category: {category}")
        print(f"  Confidence: {confidence:.2f}")

        return fact_id

    def _extract_concepts(self, fact: str) -> List[str]:
        """Extract key concepts from fact (simplified)."""
        # In practice, would use NER or more sophisticated extraction
        words = fact.lower().split()

        # Simple heuristic: words longer than 4 chars that aren't common
        common_words = {"about", "there", "which", "where", "these", "those"}
        concepts = [w for w in words if len(w) > 4 and w not in common_words]

        return concepts[:5]  # Limit to 5 concepts

    def store_relationship(self,
                          concept1: str,
                          relationship: str,
                          concept2: str,
                          confidence: float = 1.0):
        """Store a relationship between concepts."""
        rel_data = {
            "from": concept1,
            "type": relationship,
            "to": concept2,
            "confidence": confidence,
            "stored_at": datetime.now()
        }

        self.relationships[concept1].append(rel_data)

        print(f"Stored relationship: {concept1} --[{relationship}]--> {concept2}")

    def query_fact(self, query: str, threshold: float = 0.5) -> List[Dict]:
        """Query facts by content or concepts."""
        results = []

        query_lower = query.lower()

        for fact_id, fact_data in self.facts.items():
            score = 0

            # Check if query appears in fact
            if query_lower in fact_data["content"].lower():
                score = 1.0

            # Check concepts
            for concept in self._extract_concepts(query):
                if concept in self.concepts and fact_id in self.concepts[concept]:
                    score = max(score, 0.7)

            # Apply confidence
            score *= fact_data["confidence"]

            if score >= threshold:
                result = fact_data.copy()
                result["relevance_score"] = score
                result["id"] = fact_id
                results.append(result)

                # Update access info
                self.facts[fact_id]["access_count"] += 1
                self.facts[fact_id]["last_accessed"] = datetime.now()

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        print(f"Found {len(results)} facts for query: {query}")

        return results

    def get_related_facts(self, fact_id: str, max_depth: int = 2) -> List[Dict]:
        """Get facts related to a given fact."""
        if fact_id not in self.facts:
            return []

        related = set()
        base_fact = self.facts[fact_id]
        base_concepts = self._extract_concepts(base_fact["content"])

        # Find facts sharing concepts
        for concept in base_concepts:
            if concept in self.concepts:
                for related_id in self.concepts[concept]:
                    if related_id != fact_id:
                        related.add(related_id)

        # Convert to list of fact data
        results = []
        for rel_id in related:
            fact = self.facts[rel_id].copy()
            fact["id"] = rel_id
            results.append(fact)

        print(f"Found {len(results)} related facts")

        return results

    def update_confidence(self, fact_id: str, new_confidence: float):
        """Update confidence in a fact."""
        if fact_id in self.facts:
            old_confidence = self.facts[fact_id]["confidence"]
            self.facts[fact_id]["confidence"] = new_confidence
            self.confidence_scores[fact_id] = new_confidence

            print(f"Updated confidence: {old_confidence:.2f} -> {new_confidence:.2f}")
        else:
            print(f"Fact not found: {fact_id}")

    def get_knowledge_graph(self) -> Dict:
        """Get a summary of the knowledge graph."""
        return {
            "total_facts": len(self.facts),
            "total_concepts": len(self.concepts),
            "total_relationships": sum(len(rels) for rels in self.relationships.values()),
            "categories": list(set(f["category"] for f in self.facts.values())),
            "average_confidence": np.mean(list(self.confidence_scores.values()))
                                if self.confidence_scores else 0,
            "most_connected_concepts": self._get_most_connected_concepts()
        }

    def _get_most_connected_concepts(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get concepts with most connections."""
        connection_counts = {}

        for concept, facts in self.concepts.items():
            connection_counts[concept] = len(facts)

        for concept, rels in self.relationships.items():
            if concept not in connection_counts:
                connection_counts[concept] = 0
            connection_counts[concept] += len(rels)

        # Sort and return top N
        sorted_concepts = sorted(connection_counts.items(),
                               key=lambda x: x[1], reverse=True)

        return sorted_concepts[:top_n]


# Example usage
semantic = SemanticMemory()

# Store various facts
semantic.store_fact(
    "Python was created by Guido van Rossum in 1991",
    category="programming",
    confidence=1.0,
    source="documentation"
)

semantic.store_fact(
    "Python is widely used for machine learning and data science",
    category="programming",
    confidence=0.95,
    source="industry reports"
)

semantic.store_fact(
    "Machine learning requires large datasets for training",
    category="ML",
    confidence=0.9,
    source="research"
)

# Store relationships
semantic.store_relationship("Python", "used_for", "machine learning", 0.95)
semantic.store_relationship("machine learning", "requires", "datasets", 0.9)

# Query facts
results = semantic.query_fact("Python")
print("\nFacts about Python:")
for fact in results:
    print(f"  - {fact['content'][:60]}...")
    print(f"    Relevance: {fact['relevance_score']:.2f}")

# Get knowledge graph summary
print(f"\nKnowledge graph: {semantic.get_knowledge_graph()}")


# Example 5: Procedural Memory for Skills
print("\n" + "=" * 50)
print("Example 5: Procedural Memory for Skills")
print("=" * 50)


@dataclass
class Procedure:
    """Represents a procedural skill or routine."""
    name: str
    steps: List[str]
    preconditions: List[str]
    postconditions: List[str]
    success_rate: float = 0.0
    execution_count: int = 0
    last_executed: Optional[datetime] = None


class ProceduralMemory:
    """Memory system for skills and procedures."""

    def __init__(self):
        self.procedures = {}
        self.skill_hierarchy = defaultdict(list)
        self.execution_history = []

    def learn_procedure(self,
                       name: str,
                       steps: List[str],
                       preconditions: List[str] = None,
                       postconditions: List[str] = None,
                       parent_skill: str = None):
        """Learn a new procedure or skill."""
        procedure = Procedure(
            name=name,
            steps=steps,
            preconditions=preconditions or [],
            postconditions=postconditions or []
        )

        self.procedures[name] = procedure

        # Add to skill hierarchy
        if parent_skill:
            self.skill_hierarchy[parent_skill].append(name)

        print(f"Learned procedure: {name}")
        print(f"  Steps: {len(steps)}")
        print(f"  Preconditions: {len(preconditions or [])}")

        return procedure

    def execute_procedure(self, name: str, context: Dict = None) -> Dict:
        """Execute a learned procedure."""
        if name not in self.procedures:
            return {"success": False, "error": "Procedure not found"}

        procedure = self.procedures[name]

        print(f"Executing procedure: {name}")

        # Check preconditions
        if not self._check_preconditions(procedure, context):
            return {"success": False, "error": "Preconditions not met"}

        # Execute steps
        results = []
        success = True

        for i, step in enumerate(procedure.steps):
            print(f"  Step {i+1}: {step}")

            # Simulate step execution
            step_result = self._execute_step(step, context)
            results.append(step_result)

            if not step_result["success"]:
                success = False
                print(f"    Failed at step {i+1}")
                break

        # Update procedure statistics
        procedure.execution_count += 1
        procedure.last_executed = datetime.now()

        if success:
            procedure.success_rate = (
                (procedure.success_rate * (procedure.execution_count - 1) + 1)
                / procedure.execution_count
            )
        else:
            procedure.success_rate = (
                (procedure.success_rate * (procedure.execution_count - 1))
                / procedure.execution_count
            )

        # Record execution
        execution_record = {
            "procedure": name,
            "timestamp": datetime.now(),
            "success": success,
            "steps_completed": len(results),
            "total_steps": len(procedure.steps)
        }

        self.execution_history.append(execution_record)

        return {
            "success": success,
            "results": results,
            "steps_completed": len(results),
            "success_rate": procedure.success_rate
        }

    def _check_preconditions(self, procedure: Procedure, context: Dict = None) -> bool:
        """Check if preconditions are met."""
        if not procedure.preconditions:
            return True

        if not context:
            return False

        # Simple check: all preconditions should be in context
        for condition in procedure.preconditions:
            if condition not in context or not context[condition]:
                print(f"  Precondition not met: {condition}")
                return False

        return True

    def _execute_step(self, step: str, context: Dict = None) -> Dict:
        """Execute a single step (simulated)."""
        # Simulate execution with 85% success rate
        success = np.random.random() > 0.15

        return {
            "step": step,
            "success": success,
            "output": f"Result of {step}" if success else "Step failed"
        }

    def compose_procedures(self,
                          name: str,
                          component_procedures: List[str]) -> Procedure:
        """Compose multiple procedures into a new one."""
        if not all(p in self.procedures for p in component_procedures):
            print("Not all component procedures exist")
            return None

        # Combine steps from component procedures
        combined_steps = []
        combined_preconditions = []
        combined_postconditions = []

        for proc_name in component_procedures:
            proc = self.procedures[proc_name]
            combined_steps.extend([f"{proc_name}: {step}" for step in proc.steps])
            combined_preconditions.extend(proc.preconditions)
            combined_postconditions.extend(proc.postconditions)

        # Remove duplicate conditions
        combined_preconditions = list(set(combined_preconditions))
        combined_postconditions = list(set(combined_postconditions))

        # Create composed procedure
        composed = self.learn_procedure(
            name=name,
            steps=combined_steps,
            preconditions=combined_preconditions,
            postconditions=combined_postconditions
        )

        print(f"Composed procedure '{name}' from {len(component_procedures)} components")

        return composed

    def get_skill_recommendations(self, context: Dict = None) -> List[str]:
        """Recommend procedures based on context."""
        recommendations = []

        for name, procedure in self.procedures.items():
            # Check if preconditions match context
            if context and self._check_preconditions(procedure, context):
                score = procedure.success_rate * (1 + procedure.execution_count * 0.1)
                recommendations.append((name, score))

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in recommendations[:5]]

    def get_mastery_level(self, procedure_name: str) -> Dict:
        """Get mastery level for a procedure."""
        if procedure_name not in self.procedures:
            return {"mastery": "unknown"}

        procedure = self.procedures[procedure_name]

        # Calculate mastery based on executions and success rate
        if procedure.execution_count == 0:
            mastery = "unlearned"
            level = 0
        elif procedure.execution_count < 5:
            mastery = "novice"
            level = procedure.success_rate * 0.3
        elif procedure.execution_count < 20:
            mastery = "intermediate"
            level = 0.3 + procedure.success_rate * 0.4
        else:
            mastery = "expert"
            level = 0.7 + procedure.success_rate * 0.3

        return {
            "procedure": procedure_name,
            "mastery": mastery,
            "level": level,
            "executions": procedure.execution_count,
            "success_rate": procedure.success_rate
        }


# Example usage
procedural = ProceduralMemory()

# Learn basic procedures
procedural.learn_procedure(
    name="data_cleaning",
    steps=[
        "Load data",
        "Check for missing values",
        "Handle missing values",
        "Remove duplicates",
        "Validate data types"
    ],
    preconditions=["data_available"],
    postconditions=["data_cleaned"]
)

procedural.learn_procedure(
    name="feature_engineering",
    steps=[
        "Analyze features",
        "Create new features",
        "Scale features",
        "Select important features"
    ],
    preconditions=["data_cleaned"],
    postconditions=["features_ready"]
)

# Compose procedures
procedural.compose_procedures(
    name="data_preparation",
    component_procedures=["data_cleaning", "feature_engineering"]
)

# Execute procedure
context = {"data_available": True, "data_cleaned": False}
result = procedural.execute_procedure("data_cleaning", context)
print(f"\nExecution result: {result['success']}")
print(f"Success rate: {result['success_rate']:.2%}")

# Get mastery level
mastery = procedural.get_mastery_level("data_cleaning")
print(f"\nMastery level: {mastery}")


# Example 6: Memory Consolidation
print("\n" + "=" * 50)
print("Example 6: Memory Consolidation")
print("=" * 50)


class MemoryConsolidation:
    """System for consolidating and organizing memories."""

    def __init__(self):
        self.short_term_buffer = deque(maxlen=100)
        self.consolidated_memories = []
        self.memory_patterns = defaultdict(list)
        self.consolidation_threshold = 0.7

    def add_to_buffer(self, memory: Dict):
        """Add memory to short-term buffer."""
        memory["timestamp"] = datetime.now()
        memory["consolidated"] = False
        self.short_term_buffer.append(memory)

        print(f"Added to buffer: {str(memory.get('content', ''))[:50]}...")

    def consolidate(self, method: str = "similarity") -> int:
        """Consolidate memories from buffer to long-term storage."""
        print(f"Starting consolidation using method: {method}")

        if method == "similarity":
            return self._consolidate_by_similarity()
        elif method == "temporal":
            return self._consolidate_temporal()
        elif method == "importance":
            return self._consolidate_by_importance()
        else:
            return 0

    def _consolidate_by_similarity(self) -> int:
        """Consolidate similar memories together."""
        consolidated_count = 0
        groups = defaultdict(list)

        # Group similar memories
        for memory in self.short_term_buffer:
            if memory.get("consolidated"):
                continue

            # Find similar group (simplified)
            group_key = self._get_similarity_key(memory)
            groups[group_key].append(memory)

        # Consolidate groups
        for group_key, memories in groups.items():
            if len(memories) >= 2:  # Need at least 2 memories to consolidate
                consolidated = self._merge_memories(memories)
                self.consolidated_memories.append(consolidated)

                # Mark as consolidated
                for mem in memories:
                    mem["consolidated"] = True

                consolidated_count += len(memories)

                print(f"  Consolidated {len(memories)} memories into: {group_key}")

        return consolidated_count

    def _consolidate_temporal(self) -> int:
        """Consolidate memories by temporal proximity."""
        consolidated_count = 0
        time_window = timedelta(minutes=5)

        # Sort by timestamp
        sorted_memories = sorted(self.short_term_buffer, key=lambda x: x["timestamp"])

        current_group = []
        last_time = None

        for memory in sorted_memories:
            if memory.get("consolidated"):
                continue

            if last_time and memory["timestamp"] - last_time <= time_window:
                current_group.append(memory)
            else:
                # Consolidate previous group
                if len(current_group) >= 2:
                    consolidated = self._merge_memories(current_group)
                    self.consolidated_memories.append(consolidated)
                    consolidated_count += len(current_group)

                    for mem in current_group:
                        mem["consolidated"] = True

                # Start new group
                current_group = [memory]

            last_time = memory["timestamp"]

        # Handle last group
        if len(current_group) >= 2:
            consolidated = self._merge_memories(current_group)
            self.consolidated_memories.append(consolidated)
            consolidated_count += len(current_group)

        print(f"  Temporally consolidated {consolidated_count} memories")

        return consolidated_count

    def _consolidate_by_importance(self) -> int:
        """Consolidate only important memories."""
        consolidated_count = 0
        importance_threshold = 0.7

        for memory in self.short_term_buffer:
            if memory.get("consolidated"):
                continue

            importance = memory.get("importance", 0.5)

            if importance >= importance_threshold:
                self.consolidated_memories.append(memory)
                memory["consolidated"] = True
                consolidated_count += 1

                print(f"  Consolidated important memory (importance: {importance:.2f})")

        return consolidated_count

    def _get_similarity_key(self, memory: Dict) -> str:
        """Get similarity key for grouping (simplified)."""
        content = str(memory.get("content", ""))
        # Use first word as simple key
        return content.split()[0] if content else "unknown"

    def _merge_memories(self, memories: List[Dict]) -> Dict:
        """Merge multiple memories into one consolidated memory."""
        # Combine contents
        contents = [m.get("content", "") for m in memories]
        combined_content = " | ".join(contents)

        # Average importance
        importances = [m.get("importance", 0.5) for m in memories]
        avg_importance = np.mean(importances)

        # Earliest timestamp
        timestamps = [m["timestamp"] for m in memories]
        earliest = min(timestamps)

        consolidated = {
            "content": combined_content,
            "importance": avg_importance,
            "timestamp": earliest,
            "source_count": len(memories),
            "consolidated": True,
            "consolidation_time": datetime.now()
        }

        return consolidated

    def extract_patterns(self) -> List[Dict]:
        """Extract patterns from consolidated memories."""
        patterns = []

        # Analyze consolidated memories
        for memory in self.consolidated_memories:
            pattern_key = self._extract_pattern_key(memory)
            self.memory_patterns[pattern_key].append(memory)

        # Identify strong patterns
        for pattern_key, memories in self.memory_patterns.items():
            if len(memories) >= 3:  # Pattern needs multiple instances
                pattern = {
                    "pattern": pattern_key,
                    "frequency": len(memories),
                    "average_importance": np.mean([m.get("importance", 0.5)
                                                  for m in memories]),
                    "examples": memories[:3]  # Keep few examples
                }
                patterns.append(pattern)

        print(f"Extracted {len(patterns)} patterns from consolidated memories")

        return patterns

    def _extract_pattern_key(self, memory: Dict) -> str:
        """Extract pattern from memory (simplified)."""
        content = str(memory.get("content", ""))
        # Use first two words as pattern
        words = content.split()[:2]
        return " ".join(words) if words else "unknown"

    def get_consolidation_stats(self) -> Dict:
        """Get statistics about memory consolidation."""
        total_in_buffer = len(self.short_term_buffer)
        consolidated_count = sum(1 for m in self.short_term_buffer
                               if m.get("consolidated"))

        return {
            "buffer_size": total_in_buffer,
            "consolidated": consolidated_count,
            "pending": total_in_buffer - consolidated_count,
            "long_term_memories": len(self.consolidated_memories),
            "patterns_identified": len(self.memory_patterns),
            "consolidation_rate": consolidated_count / total_in_buffer
                                 if total_in_buffer > 0 else 0
        }


# Example usage
consolidator = MemoryConsolidation()

# Add various memories to buffer
memories = [
    {"content": "Python is a programming language", "importance": 0.9},
    {"content": "Python is used for data science", "importance": 0.8},
    {"content": "Machine learning uses Python", "importance": 0.7},
    {"content": "Data analysis with pandas", "importance": 0.6},
    {"content": "Data visualization with matplotlib", "importance": 0.5},
    {"content": "Python has many libraries", "importance": 0.8},
]

for memory in memories:
    consolidator.add_to_buffer(memory)
    time.sleep(0.1)  # Small delay for temporal consolidation

# Consolidate memories
consolidated = consolidator.consolidate(method="similarity")
print(f"\nConsolidated {consolidated} memories")

# Extract patterns
patterns = consolidator.extract_patterns()
print("\nIdentified patterns:")
for pattern in patterns:
    print(f"  - {pattern['pattern']}: frequency={pattern['frequency']}")

# Get stats
stats = consolidator.get_consolidation_stats()
print(f"\nConsolidation stats: {stats}")


# Example 7: Complete Memory System
print("\n" + "=" * 50)
print("Example 7: Complete Memory System")
print("=" * 50)


class CompleteMemorySystem:
    """Integrated memory system combining all memory types."""

    def __init__(self):
        self.working = WorkingMemory(capacity=7)
        self.episodic = EpisodicMemory(capacity=1000)
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.long_term = VectorMemoryStore()
        self.consolidator = MemoryConsolidation()
        self.memory_stats = defaultdict(int)

    def process_experience(self,
                          context: str,
                          action: str,
                          outcome: str,
                          success: bool) -> Dict:
        """Process a complete experience through all memory systems."""
        print(f"Processing experience: {action[:50]}...")

        # Add to working memory
        self.working.add(
            f"Experience: {action}",
            importance=2.0 if success else 1.0,
            tags=["experience"]
        )
        self.memory_stats["working_memories"] += 1

        # Record in episodic memory
        episode = self.episodic.record_episode(
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            emotional_valence=0.5 if success else -0.5,
            importance=1.5
        )
        self.memory_stats["episodes"] += 1

        # Extract and store facts
        if success:
            fact = f"{action} leads to {outcome} in context of {context}"
            self.semantic.store_fact(fact, category="learned", confidence=0.8)
            self.memory_stats["facts"] += 1

        # Learn procedure if successful
        if success and action not in self.procedural.procedures:
            self.procedural.learn_procedure(
                name=action,
                steps=[f"In {context}", f"Do {action}", f"Expect {outcome}"],
                preconditions=[context],
                postconditions=[outcome]
            )
            self.memory_stats["procedures"] += 1

        # Store in long-term memory
        memory_content = f"{context}: {action} -> {outcome}"
        memory_id = self.long_term.store(memory_content, {"success": success})
        self.memory_stats["long_term"] += 1

        # Add to consolidation buffer
        self.consolidator.add_to_buffer({
            "content": memory_content,
            "importance": 1.5 if success else 0.5,
            "type": "experience"
        })

        # Consolidate if buffer is getting full
        if len(self.consolidator.short_term_buffer) >= 50:
            self.consolidator.consolidate(method="similarity")

        return {
            "episode": episode,
            "memory_id": memory_id,
            "working_memory_size": len(self.working.memory),
            "total_memories": sum(self.memory_stats.values())
        }

    def recall(self, query: str, memory_types: List[str] = None) -> Dict:
        """Recall relevant information from specified memory systems."""
        print(f"Recalling: {query}")

        if not memory_types:
            memory_types = ["working", "episodic", "semantic", "long_term"]

        results = {}

        if "working" in memory_types:
            working_items = self.working.retrieve(query)
            results["working"] = [item.content for item in working_items]

        if "episodic" in memory_types:
            episodes = self.episodic.recall_similar_episodes(query, k=3)
            results["episodic"] = [
                {"action": ep.action, "outcome": ep.outcome, "success": ep.success}
                for ep in episodes
            ]

        if "semantic" in memory_types:
            facts = self.semantic.query_fact(query)
            results["semantic"] = [fact["content"] for fact in facts]

        if "long_term" in memory_types:
            memories = self.long_term.retrieve(query, k=3)
            results["long_term"] = [mem["content"] for mem in memories]

        return results

    def execute_skill(self, skill_name: str, context: Dict = None) -> Dict:
        """Execute a learned skill from procedural memory."""
        result = self.procedural.execute_procedure(skill_name, context)

        # Record execution in episodic memory
        self.episodic.record_episode(
            context=str(context),
            action=f"execute_{skill_name}",
            outcome="completed" if result["success"] else "failed",
            success=result["success"],
            emotional_valence=0.3 if result["success"] else -0.3
        )

        return result

    def learn_from_demonstration(self, demonstration: Dict) -> bool:
        """Learn from a demonstrated example."""
        # Extract information
        task = demonstration.get("task")
        steps = demonstration.get("steps", [])
        context = demonstration.get("context", {})

        # Store as semantic knowledge
        for i, step in enumerate(steps):
            fact = f"Step {i+1} of {task}: {step}"
            self.semantic.store_fact(fact, category="demonstration", confidence=0.9)

        # Learn as procedure
        self.procedural.learn_procedure(
            name=task,
            steps=steps,
            preconditions=list(context.keys()),
            postconditions=[f"{task}_completed"]
        )

        # Add to episodic memory
        self.episodic.record_episode(
            context="demonstration",
            action=f"learned_{task}",
            outcome="skill_acquired",
            success=True,
            emotional_valence=0.6,
            importance=2.0
        )

        print(f"Learned from demonstration: {task}")

        return True

    def get_memory_summary(self) -> Dict:
        """Get comprehensive summary of all memory systems."""
        return {
            "working_memory": self.working.get_summary(),
            "episodic_summary": {
                "total_episodes": len(self.episodic.episodes),
                "success_rate": len([e for e in self.episodic.episodes if e.success])
                              / max(len(self.episodic.episodes), 1)
            },
            "semantic_knowledge": self.semantic.get_knowledge_graph(),
            "procedural_skills": {
                "total_procedures": len(self.procedural.procedures),
                "total_executions": len(self.procedural.execution_history)
            },
            "long_term_memories": len(self.long_term.memories),
            "consolidation_stats": self.consolidator.get_consolidation_stats(),
            "total_memories_processed": sum(self.memory_stats.values())
        }


# Example usage
memory_system = CompleteMemorySystem()

# Process various experiences
experiences = [
    {
        "context": "customer_service",
        "action": "greet_politely",
        "outcome": "positive_response",
        "success": True
    },
    {
        "context": "problem_solving",
        "action": "break_down_problem",
        "outcome": "found_solution",
        "success": True
    },
    {
        "context": "customer_service",
        "action": "ignore_complaint",
        "outcome": "escalation",
        "success": False
    }
]

for exp in experiences:
    result = memory_system.process_experience(**exp)
    print(f"  Processed: {result['total_memories']} total memories\n")

# Recall relevant information
recall_results = memory_system.recall("customer_service")
print("\nRecall results for 'customer_service':")
for memory_type, memories in recall_results.items():
    print(f"  {memory_type}: {len(memories)} items")

# Learn from demonstration
demo = {
    "task": "handle_refund",
    "steps": [
        "Verify purchase",
        "Check refund policy",
        "Process refund",
        "Send confirmation"
    ],
    "context": {"purchase_verified": True, "within_policy": True}
}

memory_system.learn_from_demonstration(demo)

# Get comprehensive summary
summary = memory_system.get_memory_summary()
print("\nMemory System Summary:")
print(json.dumps(summary, indent=2, default=str))

print("\n" + "=" * 50)
print("All Memory System Examples Complete!")
print("=" * 50)