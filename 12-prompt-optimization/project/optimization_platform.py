"""
Module 12: Prompt Optimization
Production Project - Comprehensive Prompt Optimization Platform

A production-ready platform that combines all optimization techniques
into a scalable, monitored system with real-time analytics.
"""

import os
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import tiktoken
from dotenv import load_dotenv
import openai
import logging

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///optimization.db"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app
app = FastAPI(title="Prompt Optimization Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for caching and real-time features
redis_client = None


# Database Models
class OptimizationRequest(Base):
    __tablename__ = "optimization_requests"

    id = Column(String, primary_key=True)
    prompt = Column(String)
    model = Column(String)
    optimizations_applied = Column(JSON)
    original_cost = Column(Float)
    optimized_cost = Column(Float)
    tokens_saved = Column(Integer)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class ABTestResult(Base):
    __tablename__ = "ab_test_results"

    id = Column(String, primary_key=True)
    test_name = Column(String)
    variant_a = Column(String)
    variant_b = Column(String)
    winner = Column(String)
    improvement = Column(Float)
    p_value = Column(Float)
    sample_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


# Pydantic Models
class OptimizationRequestModel(BaseModel):
    prompt: str
    quality_requirement: float = Field(default=0.8, ge=0.0, le=1.0)
    max_cost: Optional[float] = None
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")
    optimization_strategies: List[str] = Field(
        default=["cache", "compress", "route", "batch"]
    )


class ABTestRequest(BaseModel):
    test_name: str
    variant_a: str
    variant_b: str
    test_inputs: List[str]
    metrics: List[str] = Field(default=["quality", "cost", "latency"])


class OptimizationResponse(BaseModel):
    request_id: str
    response: str
    model_used: str
    optimizations: List[str]
    cost: float
    tokens: int
    savings: Dict[str, float]
    processing_time: float


# Core Optimization Components
class OptimizationStrategy(Enum):
    CACHE = "cache"
    COMPRESS = "compress"
    ROUTE = "route"
    BATCH = "batch"
    TIER = "tier"


@dataclass
class ModelProfile:
    name: str
    input_cost: float
    output_cost: float
    max_tokens: int
    quality_score: float
    latency: str


# Model configurations
MODELS = {
    "gpt-3.5-turbo": ModelProfile(
        name="gpt-3.5-turbo",
        input_cost=0.0005,
        output_cost=0.0015,
        max_tokens=4096,
        quality_score=0.7,
        latency="fast"
    ),
    "gpt-4-turbo-preview": ModelProfile(
        name="gpt-4-turbo-preview",
        input_cost=0.01,
        output_cost=0.03,
        max_tokens=128000,
        quality_score=0.93,
        latency="medium"
    ),
    "gpt-4": ModelProfile(
        name="gpt-4",
        input_cost=0.03,
        output_cost=0.06,
        max_tokens=8192,
        quality_score=0.95,
        latency="slow"
    )
}


class AdvancedCache:
    """Advanced caching with semantic similarity and TTL."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}
        self.similarity_threshold = 0.9

    async def get(self, prompt: str, model: str) -> Optional[Dict]:
        """Get cached response."""
        # Try exact match
        key = self._generate_key(prompt, model)

        # Check local cache
        if key in self.local_cache:
            if self._is_valid(self.local_cache[key]):
                return self.local_cache[key]["response"]

        # Check Redis
        if self.redis:
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)

        # Try semantic similarity
        similar = await self._find_similar(prompt, model)
        if similar:
            return similar

        return None

    async def set(self, prompt: str, model: str, response: Dict):
        """Cache response."""
        key = self._generate_key(prompt, model)
        entry = {
            "response": response,
            "prompt": prompt,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }

        # Local cache
        self.local_cache[key] = entry

        # Redis cache with TTL
        if self.redis:
            await self.redis.setex(key, 3600, json.dumps(response))

    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate cache key."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_valid(self, entry: Dict) -> bool:
        """Check if cache entry is valid."""
        timestamp = datetime.fromisoformat(entry["timestamp"])
        age = (datetime.now() - timestamp).total_seconds()
        return age < 3600  # 1 hour TTL

    async def _find_similar(self, prompt: str, model: str) -> Optional[Dict]:
        """Find semantically similar cached prompt."""
        # Simplified similarity check
        for key, entry in self.local_cache.items():
            if entry.get("model") == model and self._is_valid(entry):
                similarity = self._calculate_similarity(prompt, entry["prompt"])
                if similarity >= self.similarity_threshold:
                    return entry["response"]
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / len(words1 | words2)


class PromptOptimizer:
    """Main optimization engine."""

    def __init__(self, cache: AdvancedCache):
        self.cache = cache
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.batch_queue = []
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "compressions": 0,
            "batched": 0,
            "total_cost": 0,
            "total_savings": 0
        }

    async def optimize(self,
                       prompt: str,
                       quality_requirement: float,
                       max_cost: Optional[float],
                       strategies: List[str]) -> Dict:
        """Apply optimization strategies to a prompt."""
        start_time = time.time()
        optimizations_applied = []
        original_tokens = len(self.encoder.encode(prompt))

        # Calculate baseline cost (GPT-4)
        baseline_cost = (original_tokens / 1000) * MODELS["gpt-4"].input_cost

        # 1. Cache check
        if "cache" in strategies:
            cached = await self.cache.get(prompt, "gpt-4-turbo-preview")
            if cached:
                self.metrics["cache_hits"] += 1
                return {
                    "response": cached.get("content", "Cached response"),
                    "model": "cached",
                    "optimizations": ["cache_hit"],
                    "cost": 0,
                    "tokens": 0,
                    "savings": baseline_cost,
                    "processing_time": time.time() - start_time
                }

        # 2. Compression
        processed_prompt = prompt
        if "compress" in strategies and len(prompt) > 200:
            compressed = self._compress_prompt(prompt)
            if compressed != prompt:
                self.metrics["compressions"] += 1
                optimizations_applied.append("compressed")
                processed_prompt = compressed

        # 3. Model routing
        model = self._select_model(processed_prompt, quality_requirement, max_cost)
        optimizations_applied.append(f"routed_to_{model.name}")

        # 4. Calculate optimized cost
        optimized_tokens = len(self.encoder.encode(processed_prompt))
        optimized_cost = (optimized_tokens / 1000) * model.input_cost

        # 5. Simulate API call (in production, would actually call)
        response = await self._call_api(processed_prompt, model.name)

        # 6. Cache the result
        await self.cache.set(prompt, model.name, {"content": response})

        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["total_cost"] += optimized_cost
        self.metrics["total_savings"] += (baseline_cost - optimized_cost)

        return {
            "response": response,
            "model": model.name,
            "optimizations": optimizations_applied,
            "cost": optimized_cost,
            "tokens": optimized_tokens,
            "savings": baseline_cost - optimized_cost,
            "processing_time": time.time() - start_time
        }

    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt by removing redundancy."""
        replacements = {
            "please ": "",
            "could you ": "",
            "I would like you to ": "",
            "make sure to ": "",
            "be sure to ": "",
            "in order to ": "to ",
            "at this point in time": "now",
            "due to the fact that": "because"
        }

        compressed = prompt
        for verbose, concise in replacements.items():
            compressed = compressed.replace(verbose, concise)

        return compressed.strip()

    def _select_model(self,
                     prompt: str,
                     quality_req: float,
                     max_cost: Optional[float]) -> ModelProfile:
        """Select optimal model based on requirements."""
        tokens = len(self.encoder.encode(prompt))

        candidates = []
        for name, model in MODELS.items():
            # Check token limit
            if tokens > model.max_tokens:
                continue

            # Check quality requirement
            if model.quality_score < quality_req:
                continue

            # Check cost constraint
            estimated_cost = (tokens / 1000) * model.input_cost
            if max_cost and estimated_cost > max_cost:
                continue

            candidates.append((model, estimated_cost))

        if not candidates:
            # Fallback to cheapest
            return MODELS["gpt-3.5-turbo"]

        # Sort by value (quality/cost)
        candidates.sort(key=lambda x: x[0].quality_score / x[1], reverse=True)
        return candidates[0][0]

    async def _call_api(self, prompt: str, model: str) -> str:
        """Simulate API call (in production, would actually call OpenAI)."""
        # Simulated response
        await asyncio.sleep(0.1)  # Simulate network delay
        return f"Optimized response from {model} for: {prompt[:50]}..."


class ABTester:
    """A/B testing for prompt optimization."""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def run_test(self, request: ABTestRequest) -> Dict:
        """Run A/B test on prompt variants."""
        results_a = []
        results_b = []

        for test_input in request.test_inputs:
            # Test variant A
            metrics_a = await self._test_variant(
                request.variant_a.format(input=test_input)
            )
            results_a.append(metrics_a)

            # Test variant B
            metrics_b = await self._test_variant(
                request.variant_b.format(input=test_input)
            )
            results_b.append(metrics_b)

        # Analyze results
        analysis = self._analyze_results(results_a, results_b)

        # Store in database
        test_id = hashlib.md5(f"{request.test_name}{datetime.now()}".encode()).hexdigest()
        db = SessionLocal()
        db_test = ABTestResult(
            id=test_id,
            test_name=request.test_name,
            variant_a=request.variant_a[:200],
            variant_b=request.variant_b[:200],
            winner=analysis["winner"],
            improvement=analysis["improvement"],
            p_value=analysis["p_value"],
            sample_size=len(request.test_inputs)
        )
        db.add(db_test)
        db.commit()
        db.close()

        return {
            "test_id": test_id,
            **analysis
        }

    async def _test_variant(self, prompt: str) -> Dict:
        """Test a single variant."""
        start_time = time.time()

        # Calculate metrics
        tokens = len(self.encoder.encode(prompt))
        cost = (tokens / 1000) * 0.01  # Assume GPT-4-turbo

        # Simulate quality score
        quality = 0.8 + np.random.normal(0, 0.1)
        quality = max(0, min(1, quality))

        return {
            "tokens": tokens,
            "cost": cost,
            "quality": quality,
            "latency": time.time() - start_time
        }

    def _analyze_results(self, results_a: List[Dict], results_b: List[Dict]) -> Dict:
        """Analyze A/B test results."""
        # Calculate averages
        avg_a = {
            "tokens": np.mean([r["tokens"] for r in results_a]),
            "cost": np.mean([r["cost"] for r in results_a]),
            "quality": np.mean([r["quality"] for r in results_a])
        }

        avg_b = {
            "tokens": np.mean([r["tokens"] for r in results_b]),
            "cost": np.mean([r["cost"] for r in results_b]),
            "quality": np.mean([r["quality"] for r in results_b])
        }

        # Simple statistical test (in production, use proper stats)
        from scipy import stats
        quality_a = [r["quality"] for r in results_a]
        quality_b = [r["quality"] for r in results_b]

        if len(quality_a) > 1 and len(quality_b) > 1:
            t_stat, p_value = stats.ttest_ind(quality_a, quality_b)
        else:
            p_value = 1.0

        # Determine winner
        winner = None
        if p_value < 0.05:
            winner = "B" if avg_b["quality"] > avg_a["quality"] else "A"

        improvement = ((avg_b["quality"] - avg_a["quality"]) / avg_a["quality"]) * 100

        return {
            "variant_a_stats": avg_a,
            "variant_b_stats": avg_b,
            "winner": winner,
            "improvement": improvement,
            "p_value": p_value,
            "statistically_significant": p_value < 0.05
        }


# Initialize components
@app.on_event("startup")
async def startup_event():
    global redis_client
    try:
        redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()


# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "Prompt Optimization Platform",
        "version": "1.0.0",
        "endpoints": [
            "/optimize",
            "/ab-test",
            "/metrics",
            "/history",
            "/ws/realtime"
        ]
    }


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_prompt(
    request: OptimizationRequestModel,
    background_tasks: BackgroundTasks
):
    """Optimize a prompt using selected strategies."""
    cache = AdvancedCache(redis_client)
    optimizer = PromptOptimizer(cache)

    # Generate request ID
    request_id = hashlib.md5(
        f"{request.prompt}{datetime.now()}".encode()
    ).hexdigest()[:12]

    # Optimize
    result = await optimizer.optimize(
        request.prompt,
        request.quality_requirement,
        request.max_cost,
        request.optimization_strategies
    )

    # Store in database (async in background)
    background_tasks.add_task(
        store_optimization_request,
        request_id,
        request.prompt,
        result
    )

    return OptimizationResponse(
        request_id=request_id,
        response=result["response"],
        model_used=result["model"],
        optimizations=result["optimizations"],
        cost=result["cost"],
        tokens=result["tokens"],
        savings={"amount": result["savings"], "percentage": (result["savings"] / (result["cost"] + result["savings"])) * 100 if result["cost"] > 0 else 100},
        processing_time=result["processing_time"]
    )


@app.post("/ab-test")
async def run_ab_test(request: ABTestRequest):
    """Run A/B test on prompt variants."""
    tester = ABTester()
    results = await tester.run_test(request)
    return results


@app.get("/metrics")
async def get_metrics():
    """Get platform metrics."""
    db = SessionLocal()

    # Get recent optimization stats
    recent_optimizations = db.query(OptimizationRequest).filter(
        OptimizationRequest.created_at >= datetime.now() - timedelta(hours=24)
    ).all()

    total_requests = len(recent_optimizations)
    total_savings = sum(r.original_cost - r.optimized_cost for r in recent_optimizations)
    avg_processing_time = np.mean([r.processing_time for r in recent_optimizations]) if recent_optimizations else 0

    # Get cache stats
    cache_hits = sum(1 for r in recent_optimizations if "cache_hit" in (r.optimizations_applied or []))
    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

    db.close()

    return {
        "period": "last_24_hours",
        "total_requests": total_requests,
        "total_savings": total_savings,
        "avg_processing_time": avg_processing_time,
        "cache_hit_rate": cache_hit_rate,
        "top_optimizations": get_top_optimizations(recent_optimizations)
    }


@app.get("/history")
async def get_history(limit: int = 100):
    """Get optimization history."""
    db = SessionLocal()
    history = db.query(OptimizationRequest).order_by(
        OptimizationRequest.created_at.desc()
    ).limit(limit).all()
    db.close()

    return [
        {
            "id": r.id,
            "prompt": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
            "model": r.model,
            "optimizations": r.optimizations_applied,
            "savings": r.original_cost - r.optimized_cost,
            "created_at": r.created_at
        }
        for r in history
    ]


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time optimization monitoring."""
    await websocket.accept()

    try:
        while True:
            # Send metrics every 5 seconds
            metrics = await get_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Helper functions
def store_optimization_request(request_id: str, prompt: str, result: Dict):
    """Store optimization request in database."""
    db = SessionLocal()
    try:
        # Calculate original cost (baseline)
        encoder = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoder.encode(prompt))
        original_cost = (original_tokens / 1000) * MODELS["gpt-4"].input_cost

        db_request = OptimizationRequest(
            id=request_id,
            prompt=prompt,
            model=result["model"],
            optimizations_applied=result["optimizations"],
            original_cost=original_cost,
            optimized_cost=result["cost"],
            tokens_saved=original_tokens - result["tokens"],
            processing_time=result["processing_time"]
        )
        db.add(db_request)
        db.commit()
    except Exception as e:
        logger.error(f"Error storing optimization request: {e}")
    finally:
        db.close()


def get_top_optimizations(requests: List[OptimizationRequest]) -> Dict:
    """Get most frequently used optimizations."""
    optimization_counts = {}

    for request in requests:
        if request.optimizations_applied:
            for opt in request.optimizations_applied:
                optimization_counts[opt] = optimization_counts.get(opt, 0) + 1

    # Sort by count
    sorted_opts = sorted(optimization_counts.items(), key=lambda x: x[1], reverse=True)

    return {opt: count for opt, count in sorted_opts[:5]}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    checks = {
        "database": False,
        "redis": False,
        "api": True
    }

    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        checks["database"] = True
    except:
        pass

    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            checks["redis"] = True
        except:
            pass

    status = "healthy" if all(checks.values()) else "degraded"

    return {
        "status": status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )