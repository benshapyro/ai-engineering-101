# Module 11: Advanced RAG

## Learning Objectives
By the end of this module, you will:
- Master hybrid search combining multiple retrieval methods
- Implement advanced reranking and filtering strategies
- Build multi-stage RAG pipelines with query understanding
- Handle complex document types and multimodal content
- Deploy production-scale RAG systems with optimization

## Key Concepts

### 1. Advanced Retrieval Strategies

#### Hybrid Search
```python
class HybridRetriever:
    """Combines dense and sparse retrieval methods."""

    def __init__(self, dense_model, sparse_model):
        self.dense = dense_model  # Semantic search
        self.sparse = sparse_model  # BM25/TF-IDF

    def retrieve(self, query, k=10, alpha=0.5):
        # Dense retrieval (semantic)
        dense_results = self.dense.search(query, k=k*2)

        # Sparse retrieval (keyword)
        sparse_results = self.sparse.search(query, k=k*2)

        # Normalize and combine scores
        combined = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=alpha
        )

        return combined[:k]

    def reciprocal_rank_fusion(self, results1, results2, alpha=0.5):
        """RRF score fusion algorithm."""
        scores = {}

        for rank, doc in enumerate(results1):
            scores[doc.id] = alpha / (rank + 1)

        for rank, doc in enumerate(results2):
            if doc.id in scores:
                scores[doc.id] += (1 - alpha) / (rank + 1)
            else:
                scores[doc.id] = (1 - alpha) / (rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### Multi-Vector Retrieval
```python
class MultiVectorRetriever:
    """Use multiple representations per document."""

    def __init__(self):
        self.doc_embeddings = {}  # Full document
        self.chunk_embeddings = {}  # Chunks
        self.summary_embeddings = {}  # Summaries

    def index_document(self, doc):
        # Create multiple representations
        doc_id = doc['id']

        # Full document embedding
        self.doc_embeddings[doc_id] = embed(doc['full_text'])

        # Chunk embeddings
        chunks = chunk_document(doc['full_text'])
        self.chunk_embeddings[doc_id] = [embed(c) for c in chunks]

        # Summary embedding
        summary = generate_summary(doc['full_text'])
        self.summary_embeddings[doc_id] = embed(summary)

    def retrieve(self, query, strategy='ensemble'):
        if strategy == 'ensemble':
            # Search all representations
            doc_scores = self.search_documents(query)
            chunk_scores = self.search_chunks(query)
            summary_scores = self.search_summaries(query)

            # Combine scores
            return self.ensemble_scores(doc_scores, chunk_scores, summary_scores)
```

### 2. Query Understanding & Expansion

#### Query Decomposition
```python
class QueryProcessor:
    def decompose_query(self, complex_query):
        """Break complex queries into sub-queries."""
        prompt = f"""Decompose this complex query into simpler sub-queries:

        Query: {complex_query}

        Sub-queries (one per line):"""

        sub_queries = self.llm.generate(prompt).split('\n')

        return [q.strip() for q in sub_queries if q.strip()]

    def expand_query(self, query):
        """Generate query variations."""
        expansions = {
            'synonyms': self.generate_synonyms(query),
            'hypernyms': self.generate_broader_terms(query),
            'related': self.generate_related_queries(query)
        }

        return self.combine_expansions(query, expansions)
```

#### Intent Recognition
```python
def classify_query_intent(query):
    """Determine query type for routing."""
    intents = {
        'factual': ['what is', 'when did', 'who was'],
        'comparison': ['difference between', 'compare', 'versus'],
        'procedural': ['how to', 'steps to', 'process for'],
        'analytical': ['why', 'analyze', 'explain the reason']
    }

    query_lower = query.lower()

    for intent, patterns in intents.items():
        if any(pattern in query_lower for pattern in patterns):
            return intent

    return 'general'
```

### 3. Advanced Reranking

#### Cross-Encoder Reranking
```python
class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        """Rerank documents using cross-encoder."""
        # Create query-document pairs
        pairs = [[query, doc['text']] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]
```

#### Diversity-Aware Reranking
```python
def maximal_marginal_relevance(query, documents, lambda_param=0.5):
    """MMR algorithm for diverse results."""
    selected = []
    remaining = documents.copy()

    # Select most relevant first
    scores = [similarity(query, doc) for doc in remaining]
    best_idx = np.argmax(scores)
    selected.append(remaining.pop(best_idx))

    # Iteratively select diverse relevant docs
    while remaining and len(selected) < k:
        mmr_scores = []

        for doc in remaining:
            relevance = similarity(query, doc)
            diversity = max([similarity(doc, sel) for sel in selected])
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append(mmr)

        best_idx = np.argmax(mmr_scores)
        selected.append(remaining.pop(best_idx))

    return selected
```

### 4. Document Processing & Chunking

#### Intelligent Chunking
```python
class IntelligentChunker:
    def chunk_by_structure(self, document):
        """Chunk based on document structure."""
        chunks = []

        # Detect document type
        doc_type = self.detect_type(document)

        if doc_type == 'markdown':
            chunks = self.chunk_markdown(document)
        elif doc_type == 'code':
            chunks = self.chunk_code(document)
        elif doc_type == 'table':
            chunks = self.chunk_table(document)
        else:
            chunks = self.chunk_semantic(document)

        return self.add_context_overlap(chunks)

    def chunk_semantic(self, text, max_chunk_size=512):
        """Chunk based on semantic boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sent_size = len(sentence.split())

            if current_size + sent_size > max_chunk_size:
                # Check semantic similarity
                if self.is_semantic_boundary(current_chunk, sentence):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sent_size
                else:
                    # Keep together if semantically related
                    current_chunk.append(sentence)
                    current_size += sent_size
            else:
                current_chunk.append(sentence)
                current_size += sent_size

        return chunks
```

### 5. Production RAG Architecture

#### Scalable RAG System
```python
class ProductionRAG:
    def __init__(self):
        self.query_cache = RedisCache()
        self.doc_store = ElasticsearchStore()
        self.vector_store = PineconeIndex()
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()

    async def process_query(self, query, user_context=None):
        # Check cache
        cache_key = self.generate_cache_key(query, user_context)
        cached = await self.query_cache.get(cache_key)
        if cached:
            return cached

        # Query understanding
        processed_query = await self.understand_query(query)

        # Multi-stage retrieval
        candidates = await self.retrieve_candidates(processed_query)

        # Rerank and filter
        relevant_docs = await self.rerank_and_filter(
            processed_query,
            candidates,
            user_context
        )

        # Generate response
        response = await self.generate_response(
            query,
            relevant_docs,
            user_context
        )

        # Cache result
        await self.query_cache.set(cache_key, response, ttl=3600)

        return response
```

## Advanced Techniques

### 1. Adaptive Retrieval
```python
class AdaptiveRetriever:
    """Adjusts retrieval strategy based on query characteristics."""

    def retrieve(self, query):
        # Analyze query
        query_length = len(query.split())
        query_complexity = self.assess_complexity(query)
        domain = self.detect_domain(query)

        # Select strategy
        if query_complexity == 'simple' and query_length < 5:
            # Use fast keyword search
            return self.keyword_search(query, k=3)
        elif domain == 'technical':
            # Use specialized embeddings
            return self.technical_search(query, k=7)
        else:
            # Use hybrid approach
            return self.hybrid_search(query, k=5)
```

### 2. Contextual RAG
```python
class ContextualRAG:
    """Maintains conversation context in RAG."""

    def __init__(self):
        self.conversation_history = []
        self.user_profile = {}
        self.session_context = {}

    def contextualize_query(self, query):
        """Add context to query for better retrieval."""
        context_elements = []

        # Add recent conversation
        if self.conversation_history:
            recent = self.conversation_history[-3:]
            context_elements.append(f"Recent context: {' '.join(recent)}")

        # Add user preferences
        if self.user_profile:
            prefs = self.user_profile.get('preferences', {})
            context_elements.append(f"User prefers: {prefs}")

        # Combine with original query
        contextualized = f"{' '.join(context_elements)} Query: {query}"

        return contextualized
```

### 3. Multi-Modal RAG
```python
class MultiModalRAG:
    """Handle text, images, tables, and code."""

    def process_multimodal_query(self, query, modalities):
        results = {}

        # Text retrieval
        if 'text' in modalities:
            results['text'] = self.retrieve_text(query)

        # Image retrieval
        if 'image' in modalities:
            results['images'] = self.retrieve_images(query)

        # Table retrieval
        if 'table' in modalities:
            results['tables'] = self.retrieve_tables(query)

        # Code retrieval
        if 'code' in modalities:
            results['code'] = self.retrieve_code(query)

        # Fuse multimodal results
        return self.fuse_multimodal(results)
```

## Production Optimizations

### 1. Caching Strategy
```python
class RAGCache:
    def __init__(self):
        self.embedding_cache = {}  # Cache embeddings
        self.retrieval_cache = LRUCache(1000)  # Cache retrievals
        self.response_cache = TTLCache(500, ttl=3600)  # Cache responses

    def get_or_compute_embedding(self, text):
        cache_key = hash(text)
        if cache_key not in self.embedding_cache:
            self.embedding_cache[cache_key] = compute_embedding(text)
        return self.embedding_cache[cache_key]
```

### 2. Batch Processing
```python
async def batch_rag_pipeline(queries, batch_size=10):
    """Process multiple queries efficiently."""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]

        # Parallel retrieval
        retrieval_tasks = [retrieve_async(q) for q in batch]
        retrieved = await asyncio.gather(*retrieval_tasks)

        # Batch reranking
        reranked = batch_rerank(batch, retrieved)

        # Batch generation
        generation_tasks = [generate_async(q, docs) for q, docs in zip(batch, reranked)]
        responses = await asyncio.gather(*generation_tasks)

        results.extend(responses)

    return results
```

### 3. Index Optimization
```python
class OptimizedIndex:
    def __init__(self):
        # Use HNSW for fast approximate search
        self.index = faiss.IndexHNSWFlat(dimension, 32)

        # Add IVF for large-scale search
        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimension, n_clusters)

    def optimize_for_production(self):
        # Train index
        self.index.train(training_vectors)

        # Add vectors
        self.index.add(all_vectors)

        # Optimize search parameters
        self.index.nprobe = 10  # Number of clusters to search
```

## Evaluation & Monitoring

### 1. RAG Metrics
```python
class RAGEvaluator:
    def evaluate_retrieval(self, queries, ground_truth):
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': [],
            'ndcg': []
        }

        for query, truth in zip(queries, ground_truth):
            retrieved = self.retrieve(query)
            metrics['precision_at_k'].append(
                self.precision_at_k(retrieved, truth)
            )
            # ... compute other metrics

        return {k: np.mean(v) for k, v in metrics.items()}

    def evaluate_generation(self, responses, references):
        return {
            'faithfulness': self.measure_faithfulness(responses, references),
            'relevance': self.measure_relevance(responses),
            'coherence': self.measure_coherence(responses)
        }
```

### 2. A/B Testing
```python
class RAGAB Testing:
    def __init__(self, variant_a, variant_b):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.results = {'a': [], 'b': []}

    def run_test(self, query):
        # Randomly assign variant
        variant = random.choice(['a', 'b'])

        if variant == 'a':
            response = self.variant_a.process(query)
        else:
            response = self.variant_b.process(query)

        # Track metrics
        self.track_performance(variant, response)

        return response
```

## Exercises Overview

1. **Hybrid Retriever**: Build combined dense/sparse search
2. **Query Processor**: Implement query understanding pipeline
3. **Reranker**: Create custom reranking algorithm
4. **Production Pipeline**: Build scalable RAG system
5. **Evaluation Suite**: Comprehensive RAG metrics

## Success Metrics
- **Retrieval F1**: >0.85
- **Answer Quality**: >4.5/5 human rating
- **Latency**: <500ms p95
- **Throughput**: >100 QPS
- **Cache Hit Rate**: >40%

## Next Steps
After mastering advanced RAG, you'll move to Module 12: Prompt Optimization, where you'll learn techniques for improving prompt effectiveness, reducing costs, and maximizing model performance - critical for production RAG systems.