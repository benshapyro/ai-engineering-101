# Capstone Project: Production RAG System

## Overview

This capstone project demonstrates a complete, production-ready Retrieval Augmented Generation (RAG) system that integrates concepts from all 14 modules of the curriculum.

**Objective**: Build a question-answering system over a technical documentation corpus that showcases:
- Advanced RAG techniques (hybrid retrieval, reranking, evaluation)
- Prompt engineering best practices
- Production patterns (API, monitoring, safety)
- Cost optimization
- Comprehensive testing

## System Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Query Processing                │
│  - Query expansion                      │
│  - Spelling correction                   │
│  - Intent classification                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       Hybrid Retrieval                  │
│  - Dense retrieval (embeddings)         │
│  - Sparse retrieval (BM25/TF-IDF)       │
│  - Reciprocal Rank Fusion               │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Reranking                       │
│  - Cross-encoder scoring                │
│  - Diversity-aware selection            │
│  - Top-K filtering                      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Context Generation                 │
│  - Token-aware chunking                 │
│  - Citation formatting                  │
│  - Relevance filtering                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Prompt Engineering                 │
│  - System message (role-based)          │
│  - Chain-of-thought reasoning           │
│  - Structured output format             │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         LLM Generation                  │
│  - Model selection (cost/quality)       │
│  - Temperature optimization             │
│  - Response validation                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Response Post-Processing           │
│  - Citation extraction                  │
│  - Fact verification                    │
│  - Safety filtering                     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
│  + Citations    │
│  + Confidence   │
└─────────────────┘
```

## Features

### Core Functionality
- ✅ **Hybrid Retrieval**: Combines dense (embedding) and sparse (BM25) retrieval
- ✅ **Reranking**: Cross-encoder scoring for relevance
- ✅ **Query Processing**: Expansion, correction, classification
- ✅ **Context Management**: Token-aware chunking and formatting
- ✅ **Citation Tracking**: Source attribution for all claims
- ✅ **Structured Outputs**: JSON-formatted responses with validation

### Production Features
- ✅ **FastAPI REST API**: Async endpoints with proper error handling
- ✅ **Evaluation Harness**: Automated metrics (precision, recall, F1)
- ✅ **Cost Tracking**: Token usage and API cost monitoring
- ✅ **Caching**: Prompt hashing and response caching
- ✅ **Observability**: Request logging, metrics, tracing
- ✅ **Safety Filters**: Content policy enforcement

### Advanced Techniques
- ✅ **Multi-hop Reasoning**: Chain-of-thought for complex queries
- ✅ **Few-shot Learning**: Dynamic example selection
- ✅ **Model Routing**: Cost/quality optimization
- ✅ **Prompt Optimization**: A/B testing framework
- ✅ **Agent Orchestration**: Tool use and policy enforcement

## Dataset

**Domain**: Python Programming Documentation

The system includes a curated corpus of Python programming concepts:
- Language fundamentals (data types, control flow, functions)
- Standard library modules (datetime, collections, itertools)
- Advanced topics (decorators, generators, async)
- Best practices (PEP 8, testing, packaging)

**Format**: JSONL with structure:
```json
{
  "id": "python_001",
  "title": "List Comprehensions",
  "content": "List comprehensions provide a concise way to create lists...",
  "category": "fundamentals",
  "tags": ["lists", "comprehensions", "syntax"],
  "difficulty": "beginner"
}
```

**Size**: 100 documents (~50K tokens total)

## Module Integration

This capstone integrates concepts from all 14 modules:

| Module | Concept Applied |
|--------|----------------|
| 01 | Clear, specific prompts with delimiters |
| 02 | Zero-shot classification for query intent |
| 03 | Few-shot examples for answer formatting |
| 04 | Chain-of-thought reasoning for complex queries |
| 05 | Prompt chaining (retrieve → rerank → generate) |
| 06 | Role-based prompting (Python expert persona) |
| 07 | Token-aware context management |
| 08 | Structured JSON outputs with validation |
| 09 | Function calling for tool use (search, calculator) |
| 10 | Basic RAG pipeline implementation |
| 11 | Advanced RAG (hybrid retrieval, reranking, eval) |
| 12 | Cost optimization (model selection, caching) |
| 13 | Agent design (tool orchestration, policies) |
| 14 | Production patterns (API, metrics, safety) |

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key (or Anthropic)
- 4GB RAM minimum

### Installation

```bash
# From project root
cd capstone

# Install dependencies (uses root requirements.txt)
pip install -r ../requirements.txt

# Load sample data
python src/data_loader.py

# Run tests
pytest tests/ -v
```

### Configuration

Create `capstone/.env`:
```bash
# API Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5

# RAG Configuration
RETRIEVAL_TOP_K=10
RERANKER_TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Cost Control
MAX_COST_PER_QUERY=0.10
ENABLE_CACHING=true
```

## Usage

### CLI Interface

```bash
# Run a single query
python src/main.py --query "How do list comprehensions work in Python?"

# Interactive mode
python src/main.py --interactive

# Batch evaluation
python src/main.py --eval data/test_queries.jsonl
```

### API Server

```bash
# Start FastAPI server
uvicorn src.api:app --reload

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain decorators in Python", "top_k": 3}'

# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

### Python SDK

```python
from capstone import RAGSystem

# Initialize system
rag = RAGSystem(
    data_path="data/python_docs.jsonl",
    model="gpt-5",
    retrieval_top_k=10
)

# Query
result = rag.query(
    "What are Python decorators?",
    include_citations=True,
    explain_reasoning=True
)

print(result.answer)
print(f"Sources: {result.citations}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Cost: ${result.cost_usd:.4f}")
```

## Evaluation

### Metrics

The system tracks multiple evaluation metrics:

- **Retrieval Metrics**:
  - Recall@K: % of relevant documents in top K
  - MRR (Mean Reciprocal Rank): Position of first relevant result
  - NDCG (Normalized Discounted Cumulative Gain): Ranking quality

- **Answer Quality Metrics**:
  - Accuracy: Correct answers / total questions
  - Precision: Relevant information / total information
  - Recall: Relevant information found / relevant information available
  - F1 Score: Harmonic mean of precision and recall
  - Citation Accuracy: % of claims with valid sources

- **Performance Metrics**:
  - Latency P50, P95, P99
  - Token usage (input / output)
  - Cost per query
  - Cache hit rate

### Running Evaluations

```bash
# Full evaluation suite
python src/evaluate.py --test-set data/test_queries.jsonl

# Specific metrics
python src/evaluate.py --metrics retrieval,answer_quality

# Ablation study
python src/evaluate.py --ablation reranker,chain_of_thought

# Generate report
python src/evaluate.py --report-format html
```

### Expected Results

With the provided Python documentation corpus:

| Metric | Expected Value |
|--------|---------------|
| Recall@5 | >0.90 |
| MRR | >0.85 |
| Answer Accuracy | >0.80 |
| F1 Score | >0.75 |
| P95 Latency | <3s |
| Cost per query | <$0.02 |

## Project Structure

```
capstone/
├── README.md                 # This file
├── RUBRIC.md                # Grading criteria
├── .env.example             # Environment template
├── data/
│   ├── python_docs.jsonl    # Documentation corpus
│   ├── test_queries.jsonl   # Evaluation queries
│   └── few_shot_examples.json  # Example answers
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── api.py               # FastAPI server
│   ├── rag_system.py        # Core RAG implementation
│   ├── retriever.py         # Hybrid retrieval
│   ├── reranker.py          # Cross-encoder reranking
│   ├── query_processor.py   # Query expansion/correction
│   ├── prompt_builder.py    # Prompt engineering
│   ├── evaluator.py         # Evaluation harness
│   ├── cache.py             # Response caching
│   ├── metrics.py           # Observability
│   └── data_loader.py       # Dataset utilities
├── tests/
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_rag_system.py
│   ├── test_api.py
│   └── conftest.py
└── docs/
    ├── ARCHITECTURE.md      # System design
    ├── API.md               # API documentation
    └── OPTIMIZATION.md      # Performance tuning guide
```

## Assessment Rubric

See [RUBRIC.md](RUBRIC.md) for detailed grading criteria.

**Summary**:
- **Functionality (40%)**: System works end-to-end with all features
- **Code Quality (20%)**: Clean, documented, tested code
- **RAG Performance (20%)**: Meets evaluation benchmarks
- **Integration (10%)**: Uses concepts from all 14 modules
- **Production Readiness (10%)**: API, monitoring, safety

**Passing Grade**: 70% (must meet minimum functional requirements)

## Learning Outcomes

By completing this capstone, you will demonstrate:

1. **RAG Expertise**: Build and optimize production RAG systems
2. **Prompt Engineering**: Apply techniques across diverse scenarios
3. **System Design**: Architect complex AI-powered applications
4. **Production Skills**: Deploy, monitor, and maintain LLM systems
5. **Cost Awareness**: Optimize for quality and cost simultaneously
6. **Evaluation**: Measure and improve system performance scientifically

## Extensions (Optional)

Advanced students can extend the capstone with:

- [ ] Multi-modal RAG (images, code, tables)
- [ ] Conversational context (multi-turn dialogues)
- [ ] Active learning (user feedback loop)
- [ ] Distributed retrieval (multi-index search)
- [ ] Real-time indexing (streaming updates)
- [ ] Multi-language support
- [ ] Explainability (attention visualization)
- [ ] Adversarial testing (prompt injection defense)

## Resources

- **OpenAI Cookbook**: https://cookbook.openai.com/examples/retrieval_augmented_generation
- **LangChain RAG**: https://python.langchain.com/docs/use_cases/question_answering/
- **FAISS Documentation**: https://faiss.ai
- **Sentence Transformers**: https://www.sbert.net

## Support

For questions or issues:
1. Review module READMEs for relevant concepts
2. Check existing examples in modules 10-14
3. Consult ARCHITECTURE.md for design decisions
4. Ask in course discussion forum

---

**Good luck! This project represents the culmination of your prompt engineering journey.** 🚀
