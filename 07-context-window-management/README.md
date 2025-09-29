# Module 07: Context Window Management

## Learning Objectives
By the end of this module, you will:
- Understand context window limitations and their implications
- Master techniques for optimizing context usage
- Implement sliding window and summarization strategies
- Learn to prioritize and compress information effectively
- Build systems that handle long conversations and documents

## Key Concepts

### 1. Understanding Context Windows
Context windows represent the maximum amount of text (measured in tokens) that an LLM can process in a single interaction. Managing this limited resource effectively is crucial for building robust applications.

### 2. Token Economics
```python
# Typical context window sizes (as of 2024)
model_limits = {
    "gpt-4": 8192,          # ~6,000 words
    "gpt-4-32k": 32768,     # ~24,000 words
    "gpt-4-turbo": 128000,  # ~96,000 words
    "claude-3": 200000,     # ~150,000 words
}

# Token usage breakdown
context_usage = {
    "system_prompt": 500,
    "conversation_history": 3000,
    "current_input": 1000,
    "retrieved_context": 2000,
    "remaining_for_output": 1692
}
```

### 3. Context Management Strategies

#### Sliding Window
```python
def sliding_window(messages, max_tokens=4000):
    """Keep most recent messages within token limit."""
    total_tokens = 0
    kept_messages = []

    for message in reversed(messages):
        tokens = count_tokens(message)
        if total_tokens + tokens <= max_tokens:
            kept_messages.insert(0, message)
            total_tokens += tokens
        else:
            break

    return kept_messages
```

#### Summarization
```python
def summarize_old_context(messages, threshold=2000):
    """Summarize older messages to preserve space."""
    if count_tokens(messages) > threshold:
        summary = llm.summarize(messages[:-5])
        return [summary] + messages[-5:]
    return messages
```

#### Selective Inclusion
```python
def select_relevant_context(query, documents, max_tokens=3000):
    """Include only relevant information."""
    relevance_scores = calculate_relevance(query, documents)
    selected = []
    tokens_used = 0

    for doc, score in sorted(relevance_scores.items(), reverse=True):
        if tokens_used + count_tokens(doc) <= max_tokens:
            selected.append(doc)
            tokens_used += count_tokens(doc)

    return selected
```

### 4. Common Challenges
- **Context Overflow**: Exceeding token limits
- **Information Loss**: Important details lost in compression
- **Relevance Decay**: Older context becoming less relevant
- **Cost Implications**: Larger contexts cost more
- **Latency Issues**: Processing time increases with context size

## Module Structure

### Examples
1. `token_counting.py` - Token estimation and measurement
2. `sliding_windows.py` - Implementing sliding window strategies
3. `context_compression.py` - Summarization and compression techniques

### Exercises
Practice problems focusing on:
- Token budget planning
- Context prioritization algorithms
- Compression strategies
- Long conversation handling
- Document chunking optimization

### Project: Context Optimizer
Build a system that:
- Monitors token usage in real-time
- Automatically compresses context when needed
- Prioritizes information based on relevance
- Maintains conversation coherence
- Provides analytics on context efficiency

## Best Practices

### 1. Token Budgeting
```python
class TokenBudget:
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.allocations = {
            "system": 0.1,      # 10% for system prompt
            "history": 0.4,     # 40% for conversation history
            "context": 0.3,     # 30% for retrieved context
            "buffer": 0.2       # 20% for response
        }

    def allocate(self, category):
        return int(self.max_tokens * self.allocations[category])
```

### 2. Context Prioritization
```python
def prioritize_context(elements, priorities):
    """Assign priority scores to context elements."""
    scored = []
    for element in elements:
        score = calculate_priority(element, priorities)
        scored.append((score, element))

    # Sort by priority and include until token limit
    return select_within_budget(sorted(scored, reverse=True))
```

### 3. Chunking Strategies
```python
def intelligent_chunking(document, chunk_size=500):
    """Chunk document preserving semantic boundaries."""
    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in document.split('\n\n'):
        para_tokens = count_tokens(paragraph)

        if current_size + para_tokens > chunk_size:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_size = para_tokens
        else:
            current_chunk.append(paragraph)
            current_size += para_tokens

    return chunks
```

## Production Considerations

### Performance Optimization
- **Caching**: Store compressed versions of common contexts
- **Lazy Loading**: Load context only when needed
- **Streaming**: Process long documents in chunks
- **Parallel Processing**: Handle multiple context operations concurrently

### Cost Management
```python
class CostAwareContextManager:
    def __init__(self, cost_per_1k_tokens=0.03):
        self.cost_per_1k = cost_per_1k_tokens
        self.token_budget = 5000  # Default budget

    def estimate_cost(self, tokens):
        return (tokens / 1000) * self.cost_per_1k

    def optimize_for_cost(self, context, max_cost=1.0):
        max_tokens = int((max_cost / self.cost_per_1k) * 1000)
        return compress_to_fit(context, max_tokens)
```

### Monitoring and Analytics
```python
class ContextMetrics:
    def track_usage(self, interaction):
        return {
            "total_tokens": count_tokens(interaction),
            "context_efficiency": calculate_efficiency(interaction),
            "compression_ratio": calculate_compression(interaction),
            "relevance_score": calculate_relevance(interaction),
            "cost": estimate_cost(interaction)
        }
```

## Advanced Techniques

### 1. Hierarchical Summarization
```python
def hierarchical_summary(documents, levels=3):
    """Multi-level summarization for large document sets."""
    summaries = documents

    for level in range(levels):
        chunk_size = len(summaries) // (2 ** (level + 1))
        new_summaries = []

        for i in range(0, len(summaries), chunk_size):
            chunk = summaries[i:i+chunk_size]
            summary = llm.summarize(chunk, level=level)
            new_summaries.append(summary)

        summaries = new_summaries

    return summaries[0]  # Final summary
```

### 2. Dynamic Context Injection
```python
def dynamic_context(query, context_pool, token_limit):
    """Dynamically select context based on query."""
    # Analyze query intent
    intent = analyze_intent(query)

    # Select relevant context sources
    sources = filter_by_intent(context_pool, intent)

    # Rank by relevance
    ranked = rank_by_relevance(sources, query)

    # Include within token budget
    return fit_to_budget(ranked, token_limit)
```

### 3. Context Compression Algorithms
```python
def semantic_compression(text, target_ratio=0.5):
    """Compress while preserving semantic meaning."""
    # Extract key sentences
    key_sentences = extract_key_sentences(text)

    # Remove redundancy
    unique_info = remove_redundancy(key_sentences)

    # Reconstruct compressed version
    compressed = reconstruct_narrative(unique_info)

    return compressed
```

## Common Patterns

### 1. Rolling Context
Maintain a rolling window of recent interactions while preserving key information from earlier.

### 2. Tiered Storage
Store context in tiers: immediate (full), recent (compressed), historical (summary).

### 3. Query-Driven Loading
Load only context relevant to the current query.

### 4. Checkpoint System
Create context checkpoints for long-running conversations.

## Exercises Overview

1. **Token Calculator**: Build accurate token counting for various models
2. **Window Optimizer**: Implement adaptive sliding window
3. **Compression Challenge**: Achieve 70% compression with 90% information retention
4. **Long Document Handler**: Process documents larger than context window
5. **Context Analytics**: Build dashboard for context usage patterns

## Success Metrics
- **Efficiency Rate**: >80% useful context vs total tokens
- **Compression Ratio**: 3:1 while maintaining coherence
- **Cost Reduction**: 40% lower token costs
- **Quality Preservation**: 95% accuracy after compression
- **Latency**: <10% increase despite optimization

## Next Steps
After mastering context window management, you'll move to Module 08: Structured Outputs, where you'll learn to generate precisely formatted responses including JSON, schemas, and structured data - building on efficient context use for reliable parsing.