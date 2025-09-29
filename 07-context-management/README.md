# Module 07: Context Management

Master the art of managing context windows effectively to build robust LLM applications.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Understand context window limitations and token economics
- Implement token counting and budget management
- Design sliding window and compression strategies
- Build memory systems for long conversations
- Optimize context usage for cost and performance
- Handle context overflow gracefully

## 📚 Core Concepts

### 1. Context Windows

Context windows define the maximum amount of text an LLM can process in a single request:

- **GPT-4**: 8K, 32K, or 128K tokens depending on model
- **Claude**: 100K-200K tokens
- **Trade-offs**: Larger contexts = higher cost + slower response

### 2. Token Economics

Understanding token usage is critical:

```python
# Tokens ≠ Words
"Hello world" = 2 tokens
"Antidisestablishmentarianism" = 6 tokens
"👋🌍" = 4 tokens

# Cost calculation
tokens_used = prompt_tokens + completion_tokens
cost = (tokens_used / 1000) * price_per_1k_tokens
```

### 3. Context Management Strategies

#### Sliding Window
Maintain a fixed-size window of recent context:
```
[Old Message 1] [Old Message 2] [Old Message 3] [New Message]
                 ↓ Window slides →
                 [Old Message 2] [Old Message 3] [New Message] [Response]
```

#### Compression
Summarize older context to preserve information:
```
[Long Conversation] → [Summary] + [Recent Messages]
```

#### Priority-Based Retention
Keep important context, discard less relevant:
```
[System Prompt] + [Key Facts] + [Recent Context] + [Current Query]
```

## 🛠️ Key Techniques

### 1. Token Counting

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

### 2. Context Budgeting

```python
class ContextBudget:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def can_add(self, text):
        tokens = count_tokens(text)
        return self.used_tokens + tokens <= self.max_tokens
```

### 3. Dynamic Summarization

```python
def compress_context(messages, target_tokens):
    if count_tokens(messages) <= target_tokens:
        return messages

    # Summarize older messages
    summary = summarize(messages[:-5])
    return summary + messages[-5:]
```

## 📖 Module Structure

### Examples
1. **token_management.py** - Token counting and budgeting
2. **sliding_window.py** - Implementing context windows
3. **context_compression.py** - Summarization strategies

### Exercises
1. Build a token budget calculator
2. Implement a sliding window system
3. Create context prioritization logic
4. Design dynamic summarization
5. Build a memory management system

### Project
**context_manager.py** - A complete context management system with:
- Automatic token tracking
- Multiple compression strategies
- Priority-based retention
- Conversation memory
- Cost optimization

## 🎯 Best Practices

### Do's ✅
- **Always count tokens** before sending requests
- **Set token budgets** for different parts of your prompt
- **Implement fallbacks** for context overflow
- **Monitor costs** continuously
- **Test with different context sizes**
- **Use system prompts efficiently**

### Don'ts ❌
- **Don't assume** token counts (always measure)
- **Don't ignore** context limits
- **Don't lose critical** information during compression
- **Don't compress** too aggressively
- **Don't forget** to handle edge cases

## 🚀 Practical Applications

### 1. Chatbot Memory
```python
class ChatMemory:
    def __init__(self, max_tokens=2000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim_to_fit()

    def _trim_to_fit(self):
        while count_tokens(self.messages) > self.max_tokens:
            self.messages.pop(0)  # Remove oldest
```

### 2. Document Processing
```python
def process_large_document(document, chunk_size=2000):
    chunks = split_into_chunks(document, chunk_size)
    summaries = []

    for chunk in chunks:
        summary = process_chunk(chunk)
        summaries.append(summary)

    return combine_summaries(summaries)
```

### 3. Multi-Turn Reasoning
```python
class ReasoningChain:
    def __init__(self):
        self.context = ContextManager(max_tokens=4000)
        self.key_findings = []

    def reason_step(self, query):
        # Include only relevant context
        relevant = self.context.get_relevant(query)
        response = llm_call(relevant + query)

        # Extract and store key findings
        self.key_findings.extend(extract_key_points(response))

        return response
```

## 💡 Advanced Patterns

### 1. Hierarchical Memory
```
Long-term Memory (compressed)
    ↓
Medium-term Memory (summarized)
    ↓
Short-term Memory (full detail)
    ↓
Working Memory (current context)
```

### 2. Semantic Chunking
Split context based on meaning, not just size:
- Paragraph boundaries
- Topic changes
- Dialogue turns
- Logical sections

### 3. Attention-Based Retention
Keep context based on relevance scores:
```python
def score_relevance(message, current_query):
    # Use embeddings or keyword matching
    return similarity_score(message, current_query)
```

## 🔧 Debugging Tips

### Common Issues

1. **Token Overflow**
   - Solution: Implement automatic trimming
   - Use try/catch for API calls

2. **Lost Context**
   - Solution: Preserve key information in summaries
   - Maintain "facts" separate from dialogue

3. **Inconsistent Behavior**
   - Solution: Ensure system prompts are always included
   - Test with various context sizes

## 📊 Performance Metrics

Track these metrics in production:

- **Average tokens per request**
- **Context overflow frequency**
- **Compression ratio**
- **Information retention score**
- **Cost per conversation**
- **Response quality vs context size**

## 🎓 Learning Path

1. Start with token counting basics
2. Implement simple sliding windows
3. Add compression techniques
4. Build priority systems
5. Create full memory management
6. Optimize for production

## 🔗 Additional Resources

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Tiktoken Library](https://github.com/openai/tiktoken)
- [Context Length Best Practices](https://platform.openai.com/docs/guides/context)
- [Memory Systems for LLMs](https://arxiv.org/abs/2304.03442)

## ✍️ Exercises

Complete the exercises in order:

1. **Token Calculator** - Build a tool to estimate costs
2. **Sliding Window** - Implement conversation memory
3. **Smart Compression** - Summarize without losing key info
4. **Priority Queue** - Manage context by importance
5. **Memory System** - Build a complete solution

## 🏗️ Module Project

Build a production-ready context management system that:
- Handles conversations of any length
- Optimizes token usage automatically
- Preserves important information
- Provides cost estimates
- Supports multiple compression strategies

Ready to master context management? Let's begin! 🚀