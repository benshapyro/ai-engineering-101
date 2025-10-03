# FastAPI Skeleton for LLM Applications

Production-ready API template for deploying LLM applications.

## Features

- ✅ Text generation endpoint
- ✅ RAG query endpoint
- ✅ Health checks and metrics
- ✅ Request logging
- ✅ Error handling
- ✅ CORS support
- ✅ Prometheus metrics export
- ✅ Pydantic validation

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Run Server

```bash
# Development mode (with auto-reload)
uvicorn templates.api_skeleton.main:app --reload

# Or using Python module syntax
python -m uvicorn templates.api_skeleton.main:app --reload

# Production mode
uvicorn templates.api_skeleton.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Run from Project Root

```bash
# From the prompting-101 directory
cd /path/to/prompting-101
python -m uvicorn templates.api_skeleton.main:app --reload
```

## API Endpoints

### Health & Metrics

#### `GET /api/v1/healthz`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "llm-api",
  "version": "1.0.0"
}
```

#### `GET /api/v1/metrics`

Application metrics summary.

**Response:**
```json
{
  "status": "ok",
  "metrics": {
    "total_calls": 42,
    "total_tokens": 15000,
    "total_cost": 0.0075,
    "avg_latency_ms": 450.5,
    "cache_hit_rate": 0.25
  }
}
```

#### `GET /api/v1/metrics/prometheus`

Prometheus-formatted metrics for monitoring.

#### `GET /api/v1/stats`

Detailed statistics including percentiles.

---

### Text Generation

#### `POST /api/v1/generate`

Generate text from a prompt.

**Request:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "instructions": "You are a helpful science educator",
  "temperature": 0.7,
  "max_tokens": 500,
  "model": "gpt-5"
}
```

**Response:**
```json
{
  "text": "Quantum computing is...",
  "tokens": 234,
  "cost": 0.00234,
  "model": "gpt-5"
}
```

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "What is prompt engineering?",
        "temperature": 0.7,
        "max_tokens": 300
    }
)

print(response.json()["text"])
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain RAG",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

---

### RAG Queries

#### `POST /api/v1/rag/query`

Retrieve documents and generate answer.

**Request:**
```json
{
  "query": "What is prompt engineering?",
  "top_k": 3,
  "use_rerank": true,
  "generate_answer": true
}
```

**Response:**
```json
{
  "query": "What is prompt engineering?",
  "answer": "Prompt engineering is the practice of...",
  "sources": [
    {
      "doc_id": "1",
      "content": "Prompt engineering is...",
      "score": 0.95,
      "metadata": {"category": "prompting"}
    }
  ],
  "metadata": {
    "retrieved_count": 3,
    "reranked": true
  }
}
```

**Example (Python):**
```python
response = requests.post(
    "http://localhost:8000/api/v1/rag/query",
    json={
        "query": "What is few-shot learning?",
        "top_k": 3,
        "use_rerank": True
    }
)

result = response.json()
print(result["answer"])
for source in result["sources"]:
    print(f"Source: {source['content']} (score: {source['score']})")
```

#### `GET /api/v1/rag/documents`

List all available documents.

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Models
OPENAI_MODEL=gpt-5
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Server
API_HOST=0.0.0.0
API_PORT=8000
```

### Customization

The skeleton is designed to be extended:

1. **Add new endpoints**: Create new routers in `routers/`
2. **Modify models**: Update Pydantic models for your use case
3. **Add middleware**: Implement authentication, rate limiting, etc.
4. **Configure CORS**: Update `allow_origins` in `main.py`

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "templates.api_skeleton.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn (for production)

```bash
pip install gunicorn

gunicorn templates.api_skeleton.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Setup

```bash
# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export WORKERS=4
```

## Monitoring

### Metrics Collection

Metrics are automatically collected using the `@trace_call` decorator:

```python
from metrics.tracing import trace_call

@trace_call(model="gpt-5")
def my_llm_function():
    # Your code here
    pass
```

### Prometheus Integration

Expose metrics at `/api/v1/metrics/prometheus` and configure Prometheus:

```yaml
scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics/prometheus'
```

## Error Handling

The API includes global error handling:

```python
# Automatic error responses
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred",
  "path": "/api/v1/generate"
}
```

All errors are logged with full stack traces.

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from templates.api_skeleton.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate():
    response = client.post(
        "/api/v1/generate",
        json={"prompt": "test", "max_tokens": 10}
    )
    assert response.status_code == 200
    assert "text" in response.json()
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/v1/healthz

# Using wrk
wrk -t4 -c100 -d30s http://localhost:8000/api/v1/healthz
```

## Security

### Best Practices

1. **API Keys**: Never commit `.env` files
2. **CORS**: Configure `allow_origins` appropriately
3. **Rate Limiting**: Add rate limiting middleware
4. **Authentication**: Implement API key or OAuth
5. **Input Validation**: Pydantic models validate all inputs
6. **Error Messages**: Don't leak sensitive info in errors

### Rate Limiting Example

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@limiter.limit("10/minute")
@app.post("/api/v1/generate")
async def generate_text(request: GenerateRequest):
    # Your code
    pass
```

## Extending the API

### Add Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/api/v1/generate")
async def generate_text(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    # Your code
    pass
```

### Add Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database setup
engine = create_engine("postgresql://user:pass@localhost/db")
SessionLocal = sessionmaker(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you run from project root
2. **API key errors**: Check `.env` file exists and has correct keys
3. **Port in use**: Change port with `--port 8001`
4. **CORS errors**: Update `allow_origins` in `main.py`

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn templates.api_skeleton.main:app --reload --log-level debug
```

## License

See main project LICENSE file.

## Contributing

See main project CONTRIBUTING.md file.
