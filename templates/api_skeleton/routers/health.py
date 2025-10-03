"""
Health and metrics endpoints.
"""

from fastapi import APIRouter
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from metrics.tracing import get_global_collector

router = APIRouter()


@router.get("/healthz")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "llm-api",
        "version": "1.0.0"
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics.

    Returns:
        Metrics summary
    """
    collector = get_global_collector()
    summary = collector.get_summary()

    return {
        "status": "ok",
        "metrics": summary
    }


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> str:
    """
    Get metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics
    """
    collector = get_global_collector()
    return collector.export_prometheus()


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get detailed statistics.

    Returns:
        Detailed stats including percentiles
    """
    collector = get_global_collector()

    summary = collector.get_summary()
    percentiles = collector.get_percentiles([50, 95, 99])
    recent = collector.get_recent(n=10)

    return {
        "summary": summary,
        "latency_percentiles": percentiles,
        "recent_calls": recent
    }
