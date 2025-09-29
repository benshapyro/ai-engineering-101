"""
Module 11: Advanced RAG - Query Processing Examples

This file demonstrates advanced query understanding, decomposition,
expansion, and routing techniques for improved RAG performance.

Author: Claude
Date: 2024
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from enum import Enum
import spacy
import asyncio

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# Try to load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Note: Install spacy and download en_core_web_sm for full functionality")
    nlp = None

# ================================
# Example 1: Query Decomposition
# ================================
print("=" * 50)
print("Example 1: Query Decomposition")
print("=" * 50)

class QueryDecomposer:
    """Break complex queries into simpler sub-queries."""

    def __init__(self):
        self.decomposition_strategies = {
            "multi_part": self._decompose_multi_part,
            "comparison": self._decompose_comparison,
            "temporal": self._decompose_temporal,
            "causal": self._decompose_causal
        }

    def decompose(self, query: str) -> Dict[str, Any]:
        """
        Decompose a complex query into sub-queries.

        Returns:
            Dictionary with original query, sub-queries, and strategy used
        """
        # Detect query type
        query_type = self._detect_query_type(query)

        # Apply appropriate decomposition strategy
        if query_type in self.decomposition_strategies:
            sub_queries = self.decomposition_strategies[query_type](query)
        else:
            sub_queries = self._default_decomposition(query)

        return {
            "original": query,
            "type": query_type,
            "sub_queries": sub_queries,
            "requires_aggregation": len(sub_queries) > 1
        }

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of complex query."""
        query_lower = query.lower()

        if " and " in query_lower or ", " in query:
            return "multi_part"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        elif any(word in query_lower for word in ["before", "after", "during", "when"]):
            return "temporal"
        elif any(word in query_lower for word in ["because", "why", "cause", "result"]):
            return "causal"
        else:
            return "simple"

    def _decompose_multi_part(self, query: str) -> List[str]:
        """Decompose multi-part queries."""
        # Split on conjunctions and commas
        parts = re.split(r'\s+and\s+|,\s+', query.lower())

        sub_queries = []
        for part in parts:
            # Clean up the part
            part = part.strip()
            if part:
                # Restore question format if needed
                if query.endswith("?") and not part.endswith("?"):
                    part += "?"
                sub_queries.append(part)

        return sub_queries

    def _decompose_comparison(self, query: str) -> List[str]:
        """Decompose comparison queries."""
        # Extract entities being compared
        comparison_patterns = [
            r"compare (.*?) (?:and|with|to) (.*)",
            r"difference between (.*?) and (.*)",
            r"(.*?) versus (.*)",
            r"(.*?) vs\.? (.*)"
        ]

        for pattern in comparison_patterns:
            match = re.search(pattern, query.lower())
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()

                return [
                    f"What is {entity1}?",
                    f"What is {entity2}?",
                    f"What are the characteristics of {entity1}?",
                    f"What are the characteristics of {entity2}?"
                ]

        return [query]  # Return original if no pattern matches

    def _decompose_temporal(self, query: str) -> List[str]:
        """Decompose temporal queries."""
        # Extract time-related components
        temporal_keywords = ["before", "after", "during", "when", "while"]

        sub_queries = []

        # Add query about the main event
        main_query = re.sub(r'\b(before|after|during|when|while)\b.*', '', query)
        if main_query.strip():
            sub_queries.append(main_query.strip())

        # Add query about temporal context
        for keyword in temporal_keywords:
            if keyword in query.lower():
                temporal_context = f"What happened {keyword} this?"
                sub_queries.append(temporal_context)
                break

        return sub_queries if sub_queries else [query]

    def _decompose_causal(self, query: str) -> List[str]:
        """Decompose causal queries."""
        # Split into cause and effect queries
        causal_patterns = [
            r"why (.*)",
            r"what causes (.*)",
            r"(.*) because (.*)"
        ]

        for pattern in causal_patterns:
            match = re.search(pattern, query.lower())
            if match:
                if pattern.startswith("why"):
                    main_topic = match.group(1).strip()
                    return [
                        f"What is {main_topic}?",
                        f"What are the causes of {main_topic}?",
                        f"What factors influence {main_topic}?"
                    ]
                break

        return [query]

    def _default_decomposition(self, query: str) -> List[str]:
        """Default decomposition using LLM."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Break down the complex query into 2-4 simpler sub-queries. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        sub_queries = response.choices[0].message.content.strip().split('\n')
        return [q.strip('- ').strip() for q in sub_queries if q.strip()]

# Test query decomposition
decomposer = QueryDecomposer()

test_queries = [
    "Compare machine learning and deep learning approaches",
    "What is NLP and how does it relate to computer vision?",
    "Why do neural networks require large amounts of data?",
    "What happened before and after the transformer architecture was introduced?"
]

for query in test_queries:
    result = decomposer.decompose(query)
    print(f"\nOriginal: {result['original']}")
    print(f"Type: {result['type']}")
    print("Sub-queries:")
    for i, sq in enumerate(result['sub_queries'], 1):
        print(f"  {i}. {sq}")

# ================================
# Example 2: Query Expansion
# ================================
print("\n" + "=" * 50)
print("Example 2: Query Expansion")
print("=" * 50)

class QueryExpander:
    """Expand queries for better retrieval coverage."""

    def __init__(self):
        self.expansion_methods = {
            "synonyms": self._expand_synonyms,
            "hypernyms": self._expand_hypernyms,
            "related": self._expand_related,
            "contextual": self._expand_contextual
        }

    def expand(
        self,
        query: str,
        methods: List[str] = None,
        max_expansions: int = 5
    ) -> Dict[str, List[str]]:
        """
        Expand query using multiple methods.

        Args:
            query: Original query
            methods: Expansion methods to use
            max_expansions: Maximum expansions per method

        Returns:
            Dictionary of expansions by method
        """
        if methods is None:
            methods = list(self.expansion_methods.keys())

        expansions = {"original": [query]}

        for method in methods:
            if method in self.expansion_methods:
                method_expansions = self.expansion_methods[method](query, max_expansions)
                expansions[method] = method_expansions

        return expansions

    def _expand_synonyms(self, query: str, max_expansions: int) -> List[str]:
        """Expand with synonyms."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {max_expansions} queries using synonyms. Keep the same meaning. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        expansions = response.choices[0].message.content.strip().split('\n')
        return [e.strip() for e in expansions if e.strip()][:max_expansions]

    def _expand_hypernyms(self, query: str, max_expansions: int) -> List[str]:
        """Expand with broader terms."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {max_expansions} queries using broader/general terms. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        expansions = response.choices[0].message.content.strip().split('\n')
        return [e.strip() for e in expansions if e.strip()][:max_expansions]

    def _expand_related(self, query: str, max_expansions: int) -> List[str]:
        """Expand with related queries."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {max_expansions} related queries that might find similar information. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        expansions = response.choices[0].message.content.strip().split('\n')
        return [e.strip() for e in expansions if e.strip()][:max_expansions]

    def _expand_contextual(self, query: str, max_expansions: int) -> List[str]:
        """Expand with contextual variations."""
        # Add context clues
        contexts = [
            "technical explanation",
            "practical application",
            "beginner friendly",
            "advanced details",
            "real-world example"
        ]

        expansions = []
        for context in contexts[:max_expansions]:
            expansions.append(f"{query} ({context})")

        return expansions

    def combine_expansions(
        self,
        expansions: Dict[str, List[str]],
        strategy: str = "union"
    ) -> List[str]:
        """
        Combine expansions from different methods.

        Strategies:
        - 'union': Include all unique expansions
        - 'intersection': Only expansions appearing in multiple methods
        - 'weighted': Weight by method importance
        """
        if strategy == "union":
            all_expansions = set()
            for method_expansions in expansions.values():
                all_expansions.update(method_expansions)
            return list(all_expansions)

        elif strategy == "intersection":
            # Find common expansions
            common = None
            for method_expansions in expansions.values():
                if common is None:
                    common = set(method_expansions)
                else:
                    common &= set(method_expansions)
            return list(common) if common else []

        elif strategy == "weighted":
            # Weight different methods
            weights = {
                "original": 1.0,
                "synonyms": 0.8,
                "hypernyms": 0.6,
                "related": 0.7,
                "contextual": 0.5
            }

            scored_expansions = defaultdict(float)
            for method, method_expansions in expansions.items():
                weight = weights.get(method, 0.5)
                for expansion in method_expansions:
                    scored_expansions[expansion] += weight

            # Sort by score
            sorted_expansions = sorted(
                scored_expansions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return [exp for exp, _ in sorted_expansions]

# Test query expansion
expander = QueryExpander()

test_query = "How do neural networks learn?"

# Expand using all methods
expansions = expander.expand(test_query, max_expansions=3)

print(f"Original Query: {test_query}\n")
for method, expanded in expansions.items():
    if method != "original":
        print(f"{method.capitalize()} Expansions:")
        for i, exp in enumerate(expanded, 1):
            print(f"  {i}. {exp}")
        print()

# Combine expansions
combined = expander.combine_expansions(expansions, strategy="weighted")
print("Combined Expansions (weighted):")
for i, exp in enumerate(combined[:5], 1):
    print(f"  {i}. {exp}")

# ================================
# Example 3: Intent Classification
# ================================
print("\n" + "=" * 50)
print("Example 3: Intent Classification")
print("=" * 50)

class IntentType(Enum):
    """Query intent types."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    OPINION = "opinion"
    CALCULATION = "calculation"
    NAVIGATION = "navigation"

class IntentClassifier:
    """Classify query intent for routing."""

    def __init__(self):
        self.patterns = {
            IntentType.FACTUAL: [
                "what is", "who is", "when did", "where is", "which"
            ],
            IntentType.PROCEDURAL: [
                "how to", "how do", "steps to", "process for", "tutorial"
            ],
            IntentType.COMPARISON: [
                "difference", "compare", "versus", "vs", "better", "pros and cons"
            ],
            IntentType.DEFINITION: [
                "define", "definition", "what does", "meaning of"
            ],
            IntentType.EXPLANATION: [
                "why", "explain", "reason", "cause", "understand"
            ],
            IntentType.OPINION: [
                "should", "best", "recommend", "advice", "thoughts on"
            ],
            IntentType.CALCULATION: [
                "calculate", "compute", "how many", "how much", "total"
            ],
            IntentType.NAVIGATION: [
                "find", "locate", "show me", "where can I", "navigate"
            ]
        }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent.

        Returns:
            Dictionary with primary intent, confidence, and routing info
        """
        query_lower = query.lower()

        # Pattern-based classification
        intent_scores = {}
        for intent, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score

        # LLM-based classification for complex cases
        if not intent_scores:
            intent_scores = self._llm_classification(query)

        # Get primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent] / sum(intent_scores.values())
        else:
            primary_intent = IntentType.FACTUAL
            confidence = 0.5

        # Generate routing information
        routing = self._get_routing_info(primary_intent)

        return {
            "query": query,
            "primary_intent": primary_intent,
            "confidence": confidence,
            "all_intents": intent_scores,
            "routing": routing
        }

    def _llm_classification(self, query: str) -> Dict[IntentType, float]:
        """Use LLM for intent classification."""
        intents_str = ", ".join([i.value for i in IntentType])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Classify the query intent. Choose from: {intents_str}. Respond with just the intent type."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0
        )

        intent_str = response.choices[0].message.content.strip().lower()

        # Map to IntentType
        for intent_type in IntentType:
            if intent_type.value == intent_str:
                return {intent_type: 1.0}

        return {}

    def _get_routing_info(self, intent: IntentType) -> Dict[str, Any]:
        """Get routing information based on intent."""
        routing_config = {
            IntentType.FACTUAL: {
                "retrieval_method": "dense",
                "num_sources": 3,
                "response_style": "concise"
            },
            IntentType.PROCEDURAL: {
                "retrieval_method": "hybrid",
                "num_sources": 5,
                "response_style": "step_by_step"
            },
            IntentType.COMPARISON: {
                "retrieval_method": "multi_query",
                "num_sources": 6,
                "response_style": "comparative"
            },
            IntentType.DEFINITION: {
                "retrieval_method": "dense",
                "num_sources": 2,
                "response_style": "definitional"
            },
            IntentType.EXPLANATION: {
                "retrieval_method": "hybrid",
                "num_sources": 4,
                "response_style": "explanatory"
            },
            IntentType.OPINION: {
                "retrieval_method": "diverse",
                "num_sources": 5,
                "response_style": "balanced"
            },
            IntentType.CALCULATION: {
                "retrieval_method": "sparse",
                "num_sources": 2,
                "response_style": "numerical"
            },
            IntentType.NAVIGATION: {
                "retrieval_method": "metadata",
                "num_sources": 1,
                "response_style": "directional"
            }
        }

        return routing_config.get(intent, {
            "retrieval_method": "hybrid",
            "num_sources": 3,
            "response_style": "general"
        })

# Test intent classification
classifier = IntentClassifier()

test_intents = [
    "What is machine learning?",
    "How to train a neural network?",
    "Compare supervised and unsupervised learning",
    "Why do we need activation functions?",
    "Calculate the number of parameters in a CNN",
    "Should I use TensorFlow or PyTorch?"
]

for query in test_intents:
    result = classifier.classify(query)
    print(f"\nQuery: {query}")
    print(f"Intent: {result['primary_intent'].value}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Routing: {result['routing']}")

# ================================
# Example 4: Contextual Query Enhancement
# ================================
print("\n" + "=" * 50)
print("Example 4: Contextual Query Enhancement")
print("=" * 50)

@dataclass
class ConversationContext:
    """Store conversation context."""
    history: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    topic: str = ""
    user_expertise: str = "intermediate"

class ContextualQueryProcessor:
    """Process queries with conversation context."""

    def __init__(self):
        self.context = ConversationContext()
        self.coreference_resolver = self._init_coreference()

    def _init_coreference(self):
        """Initialize coreference resolution (simplified)."""
        return {
            "pronouns": ["it", "this", "that", "they", "them"],
            "references": ["the model", "the algorithm", "the method"]
        }

    def process_with_context(self, query: str) -> Dict[str, Any]:
        """Process query with conversation context."""
        # Resolve coreferences
        resolved_query = self._resolve_coreferences(query)

        # Add context
        contextualized_query = self._add_context(resolved_query)

        # Enhance based on expertise
        enhanced_query = self._enhance_for_expertise(contextualized_query)

        # Update context
        self._update_context(query)

        return {
            "original": query,
            "resolved": resolved_query,
            "contextualized": contextualized_query,
            "enhanced": enhanced_query,
            "context": {
                "topic": self.context.topic,
                "expertise": self.context.user_expertise,
                "entities": list(self.context.entities.keys())
            }
        }

    def _resolve_coreferences(self, query: str) -> str:
        """Resolve pronouns and references."""
        resolved = query

        # Check for pronouns
        for pronoun in self.coreference_resolver["pronouns"]:
            if pronoun in query.lower():
                # Look for antecedent in history
                if self.context.history:
                    last_message = self.context.history[-1]
                    # Extract main noun from last query (simplified)
                    if "content" in last_message:
                        nouns = self._extract_nouns(last_message["content"])
                        if nouns:
                            resolved = resolved.replace(pronoun, nouns[0])

        return resolved

    def _add_context(self, query: str) -> str:
        """Add conversation context to query."""
        context_parts = []

        # Add topic context
        if self.context.topic:
            context_parts.append(f"In the context of {self.context.topic}")

        # Add entity context
        if self.context.entities:
            entities_str = ", ".join(self.context.entities.keys())
            context_parts.append(f"Related to {entities_str}")

        # Combine with query
        if context_parts:
            context_str = ". ".join(context_parts)
            return f"{context_str}: {query}"

        return query

    def _enhance_for_expertise(self, query: str) -> str:
        """Enhance query based on user expertise level."""
        expertise_modifiers = {
            "beginner": "simple explanation",
            "intermediate": "detailed explanation",
            "expert": "technical details"
        }

        modifier = expertise_modifiers.get(self.context.user_expertise, "")

        if modifier and not any(mod in query.lower() for mod in ["simple", "detailed", "technical"]):
            return f"{query} ({modifier})"

        return query

    def _update_context(self, query: str):
        """Update conversation context."""
        # Add to history
        self.context.history.append({
            "role": "user",
            "content": query
        })

        # Extract and update entities
        entities = self._extract_entities(query)
        for entity in entities:
            self.context.entities[entity] = "mentioned"

        # Update topic if needed
        if not self.context.topic:
            self.context.topic = self._detect_topic(query)

    def _extract_nouns(self, text: str) -> List[str]:
        """Extract nouns from text."""
        if nlp:
            doc = nlp(text)
            return [token.text for token in doc if token.pos_ == "NOUN"]
        else:
            # Simple fallback
            common_nouns = ["model", "algorithm", "network", "data", "system"]
            return [word for word in text.lower().split() if word in common_nouns]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities."""
        if nlp:
            doc = nlp(text)
            return [ent.text for ent in doc.ents]
        else:
            # Simple pattern matching
            entities = []
            capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities.extend(capitalized)
            return entities

    def _detect_topic(self, text: str) -> str:
        """Detect conversation topic."""
        topics = {
            "machine learning": ["learning", "model", "training"],
            "neural networks": ["neural", "network", "layer"],
            "nlp": ["language", "text", "nlp"],
            "computer vision": ["image", "vision", "visual"]
        }

        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic

        return ""

    def set_user_expertise(self, level: str):
        """Set user expertise level."""
        if level in ["beginner", "intermediate", "expert"]:
            self.context.user_expertise = level

# Test contextual processing
context_processor = ContextualQueryProcessor()

# Simulate conversation
conversation = [
    "What is a transformer model?",
    "How does it handle long sequences?",  # "it" refers to transformer
    "What are the limitations?",  # Continues transformer topic
    "Compare this with RNNs"  # "this" refers to transformer
]

print("Contextual Query Processing:\n")
for i, query in enumerate(conversation, 1):
    result = context_processor.process_with_context(query)
    print(f"Turn {i}:")
    print(f"  Original: {result['original']}")
    print(f"  Enhanced: {result['enhanced']}")
    print(f"  Context: {result['context']}")
    print()

# ================================
# Example 5: Multi-Turn Query Handling
# ================================
print("\n" + "=" * 50)
print("Example 5: Multi-Turn Query Handling")
print("=" * 50)

class MultiTurnQueryHandler:
    """Handle multi-turn conversational queries."""

    def __init__(self):
        self.conversation_state = {
            "turns": [],
            "current_topic": None,
            "clarification_needed": False,
            "follow_ups": []
        }

    def handle_turn(self, query: str) -> Dict[str, Any]:
        """Handle a conversation turn."""
        # Analyze query type
        turn_type = self._classify_turn_type(query)

        # Process based on turn type
        if turn_type == "new_topic":
            response = self._handle_new_topic(query)
        elif turn_type == "follow_up":
            response = self._handle_follow_up(query)
        elif turn_type == "clarification":
            response = self._handle_clarification(query)
        elif turn_type == "correction":
            response = self._handle_correction(query)
        else:
            response = self._handle_general(query)

        # Update conversation state
        self._update_state(query, response)

        return response

    def _classify_turn_type(self, query: str) -> str:
        """Classify the type of conversation turn."""
        query_lower = query.lower()

        # Check for follow-up indicators
        follow_up_indicators = ["more about", "tell me more", "what about", "how about"]
        if any(ind in query_lower for ind in follow_up_indicators):
            return "follow_up"

        # Check for clarification
        clarification_indicators = ["what do you mean", "can you clarify", "i meant"]
        if any(ind in query_lower for ind in clarification_indicators):
            return "clarification"

        # Check for correction
        correction_indicators = ["no, i meant", "actually", "correction"]
        if any(ind in query_lower for ind in correction_indicators):
            return "correction"

        # Check if it's a completely new topic
        if not self.conversation_state["turns"] or self._is_topic_change(query):
            return "new_topic"

        return "general"

    def _handle_new_topic(self, query: str) -> Dict[str, Any]:
        """Handle new topic introduction."""
        # Reset conversation state for new topic
        self.conversation_state["current_topic"] = self._extract_topic(query)
        self.conversation_state["follow_ups"] = self._generate_follow_ups(query)

        return {
            "type": "new_topic",
            "query": query,
            "processed_query": query,
            "topic": self.conversation_state["current_topic"],
            "suggested_follow_ups": self.conversation_state["follow_ups"][:3]
        }

    def _handle_follow_up(self, query: str) -> Dict[str, Any]:
        """Handle follow-up questions."""
        # Expand query with context from previous turns
        context = self._get_relevant_context()
        processed_query = f"{context} {query}" if context else query

        return {
            "type": "follow_up",
            "query": query,
            "processed_query": processed_query,
            "topic": self.conversation_state["current_topic"],
            "context_used": bool(context)
        }

    def _handle_clarification(self, query: str) -> Dict[str, Any]:
        """Handle clarification requests."""
        # Identify what needs clarification
        if self.conversation_state["turns"]:
            last_turn = self.conversation_state["turns"][-1]
            ambiguous_terms = self._identify_ambiguous_terms(last_turn)
        else:
            ambiguous_terms = []

        return {
            "type": "clarification",
            "query": query,
            "processed_query": query,
            "ambiguous_terms": ambiguous_terms,
            "needs_user_input": True
        }

    def _handle_correction(self, query: str) -> Dict[str, Any]:
        """Handle corrections to previous queries."""
        # Extract the correction
        correction = query.replace("no, i meant", "").replace("actually", "").strip()

        return {
            "type": "correction",
            "query": query,
            "processed_query": correction,
            "original_interpreted": self.conversation_state["turns"][-1] if self.conversation_state["turns"] else None,
            "corrected_to": correction
        }

    def _handle_general(self, query: str) -> Dict[str, Any]:
        """Handle general queries."""
        return {
            "type": "general",
            "query": query,
            "processed_query": query,
            "topic": self.conversation_state["current_topic"]
        }

    def _update_state(self, query: str, response: Dict):
        """Update conversation state."""
        self.conversation_state["turns"].append(query)

        # Update clarification flag
        self.conversation_state["clarification_needed"] = response.get("needs_user_input", False)

    def _is_topic_change(self, query: str) -> bool:
        """Detect if query represents a topic change."""
        if not self.conversation_state["current_topic"]:
            return True

        # Simple heuristic: check for topic keywords
        current_topic_words = self.conversation_state["current_topic"].lower().split()
        query_words = query.lower().split()

        overlap = len(set(current_topic_words) & set(query_words))
        return overlap == 0

    def _extract_topic(self, query: str) -> str:
        """Extract topic from query."""
        # Simple extraction (could use NER or more sophisticated methods)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the main topic in 2-3 words. Just the topic, nothing else."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def _generate_follow_ups(self, query: str) -> List[str]:
        """Generate potential follow-up questions."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Generate 3 follow-up questions. Return each on a new line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        follow_ups = response.choices[0].message.content.strip().split('\n')
        return [f.strip('- ').strip() for f in follow_ups if f.strip()]

    def _get_relevant_context(self) -> str:
        """Get relevant context from conversation history."""
        if not self.conversation_state["turns"]:
            return ""

        # Use last 2 turns as context (simplified)
        recent_turns = self.conversation_state["turns"][-2:]
        return " ".join(recent_turns)

    def _identify_ambiguous_terms(self, text: str) -> List[str]:
        """Identify potentially ambiguous terms."""
        # Simplified: look for pronouns and vague references
        ambiguous = []
        vague_terms = ["it", "this", "that", "they", "those"]

        for term in vague_terms:
            if term in text.lower():
                ambiguous.append(term)

        return ambiguous

# Test multi-turn handling
multi_turn_handler = MultiTurnQueryHandler()

# Simulate multi-turn conversation
multi_turn_conversation = [
    "What is transfer learning?",  # New topic
    "Tell me more about fine-tuning",  # Follow-up
    "What do you mean by that?",  # Clarification
    "Actually, I meant pre-training",  # Correction
    "How about BERT?"  # Related follow-up
]

print("Multi-Turn Conversation Handling:\n")
for i, query in enumerate(multi_turn_conversation, 1):
    response = multi_turn_handler.handle_turn(query)
    print(f"Turn {i}: {query}")
    print(f"  Type: {response['type']}")
    print(f"  Processed: {response['processed_query']}")
    if 'suggested_follow_ups' in response:
        print(f"  Suggested follow-ups: {response['suggested_follow_ups'][:2]}")
    print()

# ================================
# Example 6: Query Rewriting
# ================================
print("\n" + "=" * 50)
print("Example 6: Query Rewriting")
print("=" * 50)

class QueryRewriter:
    """Rewrite queries for better retrieval."""

    def __init__(self):
        self.rewriting_strategies = {
            "clarify": self._clarify_query,
            "specify": self._specify_query,
            "generalize": self._generalize_query,
            "rephrase": self._rephrase_query,
            "correct": self._correct_query
        }

    def rewrite(
        self,
        query: str,
        strategy: str = "auto",
        context: Dict = None
    ) -> Dict[str, Any]:
        """
        Rewrite query using specified strategy.

        Args:
            query: Original query
            strategy: Rewriting strategy or 'auto'
            context: Additional context

        Returns:
            Rewritten query and metadata
        """
        if strategy == "auto":
            strategy = self._select_strategy(query, context)

        if strategy in self.rewriting_strategies:
            rewritten = self.rewriting_strategies[strategy](query, context)
        else:
            rewritten = query

        return {
            "original": query,
            "rewritten": rewritten,
            "strategy": strategy,
            "changes": self._identify_changes(query, rewritten)
        }

    def _select_strategy(self, query: str, context: Dict) -> str:
        """Automatically select rewriting strategy."""
        # Check for ambiguity
        if self._is_ambiguous(query):
            return "clarify"

        # Check for vagueness
        if len(query.split()) < 3:
            return "specify"

        # Check for overly specific
        if len(query.split()) > 20:
            return "generalize"

        # Check for errors
        if self._has_errors(query):
            return "correct"

        return "rephrase"

    def _clarify_query(self, query: str, context: Dict) -> str:
        """Clarify ambiguous queries."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the query to be more clear and unambiguous. Keep it concise."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def _specify_query(self, query: str, context: Dict) -> str:
        """Make query more specific."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Make the query more specific by adding relevant details. Keep it natural."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def _generalize_query(self, query: str, context: Dict) -> str:
        """Make query more general."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Simplify and generalize the query while keeping the main intent."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def _rephrase_query(self, query: str, context: Dict) -> str:
        """Rephrase query for variety."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Rephrase the query in a different way while keeping the same meaning."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    def _correct_query(self, query: str, context: Dict) -> str:
        """Correct errors in query."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Correct any spelling, grammar, or factual errors in the query."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def _is_ambiguous(self, query: str) -> bool:
        """Check if query is ambiguous."""
        ambiguous_terms = ["it", "this", "that", "thing", "stuff"]
        query_lower = query.lower()
        return any(term in query_lower for term in ambiguous_terms)

    def _has_errors(self, query: str) -> bool:
        """Check if query has obvious errors."""
        # Simple check for repeated words or missing spaces
        words = query.split()
        if len(words) != len(set(words)):
            return True  # Has repeated words

        # Could add spell checking here
        return False

    def _identify_changes(self, original: str, rewritten: str) -> Dict[str, Any]:
        """Identify changes made during rewriting."""
        return {
            "length_change": len(rewritten) - len(original),
            "word_count_change": len(rewritten.split()) - len(original.split()),
            "similarity": self._calculate_similarity(original, rewritten)
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between texts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

# Test query rewriting
rewriter = QueryRewriter()

test_rewrites = [
    ("ML", "specify"),  # Too short
    ("What is it?", "clarify"),  # Ambiguous
    ("How does the transformer architecture work in the context of natural language processing tasks specifically for sequence-to-sequence modeling with attention mechanisms", "generalize"),  # Too long
    ("nueral netwroks", "correct"),  # Has errors
    ("What are neural networks?", "rephrase")  # Normal rephrase
]

print("Query Rewriting Examples:\n")
for original, strategy in test_rewrites:
    result = rewriter.rewrite(original, strategy=strategy)
    print(f"Original: {result['original']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Rewritten: {result['rewritten']}")
    print(f"Changes: {result['changes']}")
    print()

# ================================
# Example 7: Entity Extraction
# ================================
print("\n" + "=" * 50)
print("Example 7: Entity Extraction and Disambiguation")
print("=" * 50)

class EntityExtractor:
    """Extract and disambiguate entities from queries."""

    def __init__(self):
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE",
            "TECHNOLOGY", "CONCEPT", "METRIC", "MODEL"
        ]
        self.entity_cache = {}

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query."""
        # Use LLM for entity extraction
        entities = self._llm_extract(query)

        # Disambiguate entities
        disambiguated = self._disambiguate_entities(entities, query)

        # Link entities to knowledge base
        linked = self._link_entities(disambiguated)

        return {
            "query": query,
            "entities": entities,
            "disambiguated": disambiguated,
            "linked": linked,
            "entity_query": self._create_entity_query(linked)
        }

    def _llm_extract(self, query: str) -> List[Dict]:
        """Extract entities using LLM."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""Extract entities from the text. For each entity, provide:
                    - text: the entity text
                    - type: one of {', '.join(self.entity_types)}
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
            return entities
        except:
            return []

    def _disambiguate_entities(self, entities: List[Dict], context: str) -> List[Dict]:
        """Disambiguate extracted entities."""
        disambiguated = []

        for entity in entities:
            # Check for ambiguous entities
            if self._is_ambiguous_entity(entity["text"]):
                # Resolve ambiguity
                resolved = self._resolve_ambiguity(entity, context)
                disambiguated.append(resolved)
            else:
                disambiguated.append(entity)

        return disambiguated

    def _is_ambiguous_entity(self, entity_text: str) -> bool:
        """Check if entity is ambiguous."""
        ambiguous_terms = {
            "bert": ["BERT model", "Bidirectional Encoder Representations from Transformers"],
            "gpt": ["GPT model", "Generative Pre-trained Transformer"],
            "cnn": ["Convolutional Neural Network", "CNN news"],
            "transformer": ["Transformer architecture", "electrical transformer"]
        }

        return entity_text.lower() in ambiguous_terms

    def _resolve_ambiguity(self, entity: Dict, context: str) -> Dict:
        """Resolve entity ambiguity using context."""
        # Simple resolution based on context
        tech_context = ["model", "algorithm", "neural", "learning", "ai"]

        if any(word in context.lower() for word in tech_context):
            # Assume technical meaning
            if entity["text"].lower() == "cnn":
                entity["resolved"] = "Convolutional Neural Network"
                entity["type"] = "TECHNOLOGY"
            elif entity["text"].lower() == "transformer":
                entity["resolved"] = "Transformer architecture"
                entity["type"] = "TECHNOLOGY"

        return entity

    def _link_entities(self, entities: List[Dict]) -> List[Dict]:
        """Link entities to knowledge base entries."""
        linked = []

        for entity in entities:
            # Simulate knowledge base lookup
            kb_entry = self._lookup_knowledge_base(entity.get("resolved", entity["text"]))

            if kb_entry:
                entity["kb_id"] = kb_entry["id"]
                entity["kb_info"] = kb_entry["info"]

            linked.append(entity)

        return linked

    def _lookup_knowledge_base(self, entity_text: str) -> Dict:
        """Lookup entity in knowledge base."""
        # Simulated knowledge base
        kb = {
            "BERT model": {
                "id": "kb_001",
                "info": "Pre-trained transformer model by Google"
            },
            "Transformer architecture": {
                "id": "kb_002",
                "info": "Attention-based neural architecture"
            },
            "Convolutional Neural Network": {
                "id": "kb_003",
                "info": "Deep learning architecture for image processing"
            }
        }

        return kb.get(entity_text, None)

    def _create_entity_query(self, entities: List[Dict]) -> str:
        """Create enhanced query with entity information."""
        if not entities:
            return ""

        entity_strings = []
        for entity in entities:
            if "kb_info" in entity:
                entity_strings.append(f"{entity['text']} ({entity['kb_info']})")
            else:
                entity_strings.append(entity["text"])

        return " AND ".join(entity_strings)

# Test entity extraction
extractor = EntityExtractor()

test_entity_queries = [
    "How does BERT compare to GPT for NLP tasks?",
    "What is the architecture of CNN models?",
    "Explain transformer attention mechanism"
]

print("Entity Extraction and Disambiguation:\n")
for query in test_entity_queries:
    result = extractor.extract_entities(query)
    print(f"Query: {query}")
    print(f"Entities: {result['entities']}")
    if result['disambiguated']:
        print(f"Disambiguated: {result['disambiguated']}")
    if result['entity_query']:
        print(f"Entity Query: {result['entity_query']}")
    print()

print("\n" + "=" * 50)
print("Query Processing Examples Complete!")
print("=" * 50)