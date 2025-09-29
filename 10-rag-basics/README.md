# Module 10: RAG Basics (Retrieval Augmented Generation)

## Learning Objectives
By the end of this module, you will:
- Understand the fundamentals of RAG architecture
- Build basic retrieval systems for knowledge augmentation
- Implement vector databases and embedding strategies
- Master prompt engineering for RAG systems
- Create end-to-end RAG pipelines

## Key Concepts

### 1. What is RAG?
Retrieval Augmented Generation combines the generative capabilities of LLMs with the precision of information retrieval systems, enabling models to access and utilize external knowledge sources dynamically.

### 2. RAG Architecture Components

```python
# Basic RAG Pipeline
class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorDatabase()
        self.retriever = Retriever()
        self.generator = LLM()

    def process(self, query):
        # 1. Embed query
        query_embedding = self.embedder.encode(query)

        # 2. Retrieve relevant documents
        documents = self.retriever.search(query_embedding, k=5)

        # 3. Augment prompt with retrieved context
        augmented_prompt = self.build_prompt(query, documents)

        # 4. Generate response
        response = self.generator.complete(augmented_prompt)

        return response
```

### 3. Embedding Strategies

#### Text Embeddings
```python
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        """Convert documents to vector representations."""
        embeddings = self.model.encode(documents, batch_size=32)
        return embeddings

    def embed_query(self, query):
        """Embed query for similarity search."""
        return self.model.encode(query)
```

#### Chunking Strategies
```python
def chunk_document(document, chunk_size=500, overlap=50):
    """Split document into overlapping chunks."""
    chunks = []
    words = document.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append({
            'text': chunk,
            'start': i,
            'end': min(i + chunk_size, len(words))
        })

    return chunks
```

### 4. Vector Databases

#### Basic Vector Store
```python
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_documents(self, embeddings, documents):
        """Add documents and their embeddings."""
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(documents)

    def search(self, query_embedding, k=5):
        """Search for similar documents."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        return [self.documents[i] for i in indices[0]]
```

### 5. Common Challenges
- **Retrieval Quality**: Finding truly relevant documents
- **Context Length**: Managing retrieved content within limits
- **Hallucination**: Model ignoring retrieved facts
- **Latency**: Retrieval adds processing time
- **Scalability**: Handling large document collections

## Module Structure

### Examples
1. `simple_rag.py` - Basic RAG implementation
2. `vector_databases.py` - Working with vector stores
3. `retrieval_strategies.py` - Different retrieval approaches

### Exercises
Practice problems focusing on:
- Building document indices
- Optimizing retrieval quality
- Prompt engineering for RAG
- Handling multiple data sources
- Evaluation metrics for RAG

### Project: Knowledge Base QA System
Build a system that:
- Ingests and indexes documents
- Performs semantic search
- Generates answers with citations
- Handles document updates
- Provides relevance feedback

## Best Practices

### 1. Document Processing
```python
class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, document):
        """Process document for RAG pipeline."""
        # Clean text
        cleaned = self.clean_text(document['text'])

        # Extract metadata
        metadata = self.extract_metadata(document)

        # Create chunks with metadata
        chunks = []
        for chunk_text in self.chunk_text(cleaned):
            chunks.append({
                'text': chunk_text,
                'metadata': metadata,
                'source': document['source']
            })

        return chunks

    def clean_text(self, text):
        """Remove noise and normalize text."""
        # Remove special characters, normalize whitespace
        return text.strip()

    def extract_metadata(self, document):
        """Extract useful metadata."""
        return {
            'title': document.get('title'),
            'date': document.get('date'),
            'author': document.get('author'),
            'type': document.get('type')
        }
```

### 2. Retrieval Optimization
```python
class OptimizedRetriever:
    def __init__(self, vector_store, reranker=None):
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(self, query, k=10, rerank_k=5):
        """Two-stage retrieval with reranking."""
        # Stage 1: Fast vector search
        candidates = self.vector_store.search(query, k=k)

        # Stage 2: Rerank with cross-encoder
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates)
            candidates = candidates[:rerank_k]

        return candidates

    def hybrid_search(self, query, k=5):
        """Combine vector and keyword search."""
        # Vector search
        vector_results = self.vector_store.semantic_search(query, k=k)

        # Keyword search
        keyword_results = self.vector_store.keyword_search(query, k=k)

        # Merge and deduplicate
        return self.merge_results(vector_results, keyword_results)
```

### 3. Prompt Engineering for RAG
```python
def build_rag_prompt(query, documents, instruction=None):
    """Create effective RAG prompt."""
    prompt = []

    # Add instruction if provided
    if instruction:
        prompt.append(f"Instruction: {instruction}\n")

    # Add retrieved context
    prompt.append("Context information:\n")
    prompt.append("-" * 40 + "\n")

    for i, doc in enumerate(documents, 1):
        prompt.append(f"Document {i}:\n{doc['text']}\n")
        if doc.get('source'):
            prompt.append(f"Source: {doc['source']}\n")
        prompt.append("-" * 20 + "\n")

    # Add query
    prompt.append("-" * 40 + "\n")
    prompt.append(f"Question: {query}\n\n")

    # Add generation instructions
    prompt.append("Based on the context above, provide a comprehensive answer. ")
    prompt.append("If the context doesn't contain relevant information, say so.\n")
    prompt.append("Answer:")

    return ''.join(prompt)
```

## Production Considerations

### Indexing Pipeline
```python
class IndexingPipeline:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.processed_docs = set()

    def index_documents(self, documents):
        """Batch index documents."""
        new_docs = []

        for doc in documents:
            # Skip if already processed
            doc_id = self.generate_doc_id(doc)
            if doc_id in self.processed_docs:
                continue

            # Process and chunk
            chunks = self.process_document(doc)
            new_docs.extend(chunks)
            self.processed_docs.add(doc_id)

        if new_docs:
            # Batch embed
            embeddings = self.embedder.embed_batch(
                [d['text'] for d in new_docs]
            )

            # Add to vector store
            self.vector_store.add_batch(embeddings, new_docs)

        return len(new_docs)
```

### Performance Optimization
- **Caching**: Cache embeddings and search results
- **Batch Processing**: Process documents in batches
- **Async Operations**: Use async for I/O operations
- **Index Optimization**: Use appropriate index types (HNSW, IVF)

### Monitoring
```python
class RAGMetrics:
    def __init__(self):
        self.queries = []
        self.retrieval_scores = []
        self.generation_scores = []

    def log_query(self, query, retrieved_docs, response):
        self.queries.append({
            'query': query,
            'num_docs': len(retrieved_docs),
            'response_length': len(response),
            'timestamp': datetime.now()
        })

    def calculate_metrics(self):
        return {
            'avg_docs_retrieved': np.mean([q['num_docs'] for q in self.queries]),
            'avg_response_length': np.mean([q['response_length'] for q in self.queries]),
            'retrieval_precision': self.calculate_precision(),
            'response_quality': self.calculate_quality_score()
        }
```

## Advanced Techniques

### 1. Query Expansion
```python
def expand_query(query, llm):
    """Expand query for better retrieval."""
    prompt = f"""Generate 3 alternative phrasings of this question:
    Original: {query}

    Alternatives:"""

    alternatives = llm.generate(prompt)
    return [query] + parse_alternatives(alternatives)
```

### 2. Document Filtering
```python
def filter_documents(documents, query, metadata_filters=None):
    """Apply metadata and relevance filters."""
    filtered = documents

    # Apply metadata filters
    if metadata_filters:
        filtered = [
            doc for doc in filtered
            if match_metadata(doc, metadata_filters)
        ]

    # Apply relevance threshold
    filtered = [
        doc for doc in filtered
        if doc['score'] > RELEVANCE_THRESHOLD
    ]

    return filtered
```

### 3. Citation Generation
```python
def generate_with_citations(query, documents, llm):
    """Generate response with inline citations."""
    prompt = build_citation_prompt(query, documents)
    response = llm.generate(prompt)

    # Add citation references
    cited_response = add_citation_links(response, documents)

    return {
        'answer': cited_response,
        'sources': extract_used_sources(response, documents)
    }
```

## Common RAG Patterns

### 1. Question Answering
```python
def qa_pipeline(question, knowledge_base):
    # Retrieve relevant passages
    passages = retrieve_relevant(question, knowledge_base)

    # Generate answer
    answer = generate_answer(question, passages)

    return answer
```

### 2. Conversational RAG
```python
def conversational_rag(message, history, knowledge_base):
    # Include conversation context
    context_query = build_context_query(message, history)

    # Retrieve with context awareness
    documents = retrieve_with_context(context_query, knowledge_base)

    # Generate contextual response
    response = generate_contextual_response(message, history, documents)

    return response
```

### 3. Multi-Modal RAG
```python
def multimodal_rag(query, text_docs, images, tables):
    # Retrieve from multiple sources
    text_results = retrieve_text(query, text_docs)
    image_results = retrieve_images(query, images)
    table_results = retrieve_tables(query, tables)

    # Combine and generate
    combined_context = merge_multimodal(text_results, image_results, table_results)
    response = generate_from_multimodal(query, combined_context)

    return response
```

## Evaluation Metrics

### 1. Retrieval Metrics
- **Precision@K**: Relevant docs in top K
- **Recall@K**: Coverage of relevant docs
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### 2. Generation Metrics
- **Faithfulness**: Adherence to retrieved content
- **Relevance**: Answer relevance to query
- **Completeness**: Coverage of query aspects
- **Citation Accuracy**: Correct source attribution

## Exercises Overview

1. **Index Builder**: Create efficient document index
2. **Retriever Optimizer**: Improve retrieval quality
3. **Prompt Engineer**: Design effective RAG prompts
4. **Citation System**: Add source attribution
5. **Evaluation Suite**: Build RAG evaluation metrics

## Success Metrics
- **Retrieval Precision**: >80% relevant documents
- **Answer Accuracy**: >90% factually correct
- **Response Time**: <2s end-to-end
- **Scalability**: Handle 1M+ documents
- **Citation Rate**: >95% answers with sources

## Next Steps
After mastering RAG basics, you'll move to Module 11: Advanced RAG, where you'll learn sophisticated techniques like hybrid search, query understanding, document reranking, and production-scale RAG systems - building on these fundamentals for enterprise applications.