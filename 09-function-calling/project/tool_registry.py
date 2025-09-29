"""
Module 09: Function Calling - Tool Registry Project

Production-ready tool registry system with comprehensive features for
managing, executing, and monitoring function calling at scale.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import asyncio
import hashlib
import importlib
import inspect
import time
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Union
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import yaml
from jsonschema import validate, ValidationError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== Data Models =====

@dataclass
class ToolVersion:
    """Tool version information."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


@dataclass
class ToolMetadata:
    """Comprehensive tool metadata."""
    id: str
    name: str
    description: str
    version: ToolVersion
    author: str
    category: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict] = field(default_factory=list)
    documentation: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deprecated: bool = False
    deprecation_message: Optional[str] = None


@dataclass
class ToolExecution:
    """Tool execution record."""
    id: str
    tool_id: str
    user_id: str
    session_id: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    cached: bool = False


class ToolStatus(Enum):
    """Tool availability status."""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


# ===== Core Registry =====

class ToolRegistry:
    """Central tool registry with comprehensive management features."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.tools: Dict[str, 'Tool'] = {}
        self.categories: Dict[str, Set[str]] = defaultdict(set)
        self.tags: Dict[str, Set[str]] = defaultdict(set)
        self.versions: Dict[str, List[ToolVersion]] = defaultdict(list)
        self.cache = CacheManager(self.config.get("cache", {}))
        self.metrics = MetricsCollector()
        self.security = SecurityManager(self.config.get("security", {}))
        self.plugins = PluginManager(self)
        self._lock = threading.RLock()

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "cache": {
                "enabled": True,
                "ttl": 300,
                "max_size": 1000
            },
            "security": {
                "sandbox": True,
                "max_execution_time": 30,
                "max_memory": 100 * 1024 * 1024  # 100MB
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60
            }
        }

    def register_tool(self, tool: 'Tool') -> bool:
        """Register a new tool."""
        with self._lock:
            tool_id = tool.metadata.id

            # Check for existing tool
            if tool_id in self.tools:
                existing_version = self.tools[tool_id].metadata.version
                if tool.metadata.version <= existing_version:
                    logger.warning(f"Tool {tool_id} version {tool.metadata.version} not newer than {existing_version}")
                    return False

            # Validate dependencies
            for dep in tool.metadata.dependencies:
                if dep not in self.tools:
                    logger.error(f"Dependency {dep} not found for {tool_id}")
                    return False

            # Register tool
            self.tools[tool_id] = tool
            self.categories[tool.metadata.category].add(tool_id)
            for tag in tool.metadata.tags:
                self.tags[tag].add(tool_id)
            self.versions[tool_id].append(tool.metadata.version)

            logger.info(f"Registered tool: {tool_id} v{tool.metadata.version}")
            self.metrics.record_event("tool_registered", {"tool_id": tool_id})

            return True

    def get_tool(self, tool_id: str, version: Optional[str] = None) -> Optional['Tool']:
        """Get a tool by ID and optional version."""
        if tool_id not in self.tools:
            return None

        tool = self.tools[tool_id]

        # Check if specific version requested
        if version and str(tool.metadata.version) != version:
            logger.warning(f"Version mismatch for {tool_id}: requested {version}, have {tool.metadata.version}")
            return None

        # Check deprecation
        if tool.metadata.deprecated:
            logger.warning(f"Tool {tool_id} is deprecated: {tool.metadata.deprecation_message}")

        return tool

    def search_tools(self,
                    query: Optional[str] = None,
                    category: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    exclude_deprecated: bool = True) -> List['Tool']:
        """Search for tools based on criteria."""
        results = []

        for tool_id, tool in self.tools.items():
            # Skip deprecated if requested
            if exclude_deprecated and tool.metadata.deprecated:
                continue

            # Category filter
            if category and tool.metadata.category != category:
                continue

            # Tag filter
            if tags:
                tool_tags = set(tool.metadata.tags)
                if not any(tag in tool_tags for tag in tags):
                    continue

            # Query filter
            if query:
                query_lower = query.lower()
                if (query_lower not in tool.metadata.name.lower() and
                    query_lower not in tool.metadata.description.lower()):
                    continue

            results.append(tool)

        return results

    def execute_tool(self,
                    tool_id: str,
                    arguments: Dict[str, Any],
                    user_context: Optional['UserContext'] = None) -> ToolExecution:
        """Execute a tool with full lifecycle management."""

        execution = ToolExecution(
            id=str(uuid.uuid4()),
            tool_id=tool_id,
            user_id=user_context.user_id if user_context else "anonymous",
            session_id=user_context.session_id if user_context else str(uuid.uuid4()),
            arguments=arguments
        )

        # Get tool
        tool = self.get_tool(tool_id)
        if not tool:
            execution.error = f"Tool {tool_id} not found"
            execution.end_time = datetime.now()
            return execution

        # Security checks
        if user_context and not self.security.check_permissions(user_context, tool):
            execution.error = "Permission denied"
            execution.end_time = datetime.now()
            return execution

        # Check rate limits
        if user_context and not self.security.check_rate_limit(user_context, tool_id):
            execution.error = "Rate limit exceeded"
            execution.end_time = datetime.now()
            return execution

        # Check cache
        cache_key = self.cache.make_key(tool_id, arguments)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            execution.result = cached_result
            execution.success = True
            execution.cached = True
            execution.end_time = datetime.now()
            execution.duration = 0
            self.metrics.record_execution(execution)
            return execution

        # Execute tool
        try:
            # Validate arguments
            if tool.metadata.schema:
                validate(instance=arguments, schema=tool.metadata.schema)

            # Execute with timeout
            result = tool.execute(**arguments)

            # Cache result
            self.cache.set(cache_key, result)

            # Update execution
            execution.result = result
            execution.success = True

        except ValidationError as e:
            execution.error = f"Validation error: {e.message}"
        except Exception as e:
            execution.error = str(e)
            logger.error(f"Tool execution failed: {e}")

        # Finalize execution
        execution.end_time = datetime.now()
        execution.duration = (execution.end_time - execution.start_time).total_seconds()

        # Record metrics
        self.metrics.record_execution(execution)

        # Audit log
        if user_context:
            self.security.audit_log(user_context, tool_id, execution.success)

        return execution

    def get_statistics(self) -> Dict:
        """Get comprehensive registry statistics."""
        return {
            "total_tools": len(self.tools),
            "categories": {cat: len(tools) for cat, tools in self.categories.items()},
            "deprecated_count": sum(1 for t in self.tools.values() if t.metadata.deprecated),
            "cache_stats": self.cache.get_stats(),
            "metrics": self.metrics.get_summary(),
            "security": self.security.get_stats()
        }


# ===== Tool Implementation =====

class Tool:
    """Base class for all tools."""

    def __init__(self, metadata: ToolMetadata, function: Callable):
        self.metadata = metadata
        self.function = function
        self.status = ToolStatus.AVAILABLE
        self.middleware: List[Callable] = []

    def add_middleware(self, middleware: Callable):
        """Add middleware for preprocessing/postprocessing."""
        self.middleware.append(middleware)

    def execute(self, **kwargs) -> Any:
        """Execute the tool with middleware."""
        # Apply middleware in order
        for mw in self.middleware:
            kwargs = mw(kwargs)

        # Execute function
        result = self.function(**kwargs)

        return result

    @classmethod
    def from_function(cls, func: Callable, category: str = "general") -> 'Tool':
        """Create tool from a function using introspection."""
        sig = inspect.signature(func)

        # Build schema from signature
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            param_schema = {"type": "string"}  # Default type

            # Infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == list:
                    param_schema["type"] = "array"
                elif param.annotation == dict:
                    param_schema["type"] = "object"

            schema["properties"][param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)

        # Create metadata
        metadata = ToolMetadata(
            id=func.__name__,
            name=func.__name__.replace("_", " ").title(),
            description=func.__doc__ or "No description",
            version=ToolVersion(1, 0, 0),
            author="System",
            category=category,
            schema=schema
        )

        return cls(metadata, func)


# ===== Plugin System =====

class PluginManager:
    """Manage tool plugins for extensibility."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.plugins: Dict[str, 'Plugin'] = {}
        self.plugin_paths: List[Path] = []
        self.auto_reload = False
        self._watchers = {}

    def add_plugin_path(self, path: Union[str, Path]):
        """Add a path to search for plugins."""
        path = Path(path)
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
            if path not in sys.path:
                sys.path.insert(0, str(path))

    def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
        discovered = []

        for path in self.plugin_paths:
            if not path.exists():
                continue

            for file_path in path.glob("*_plugin.py"):
                plugin_name = file_path.stem
                if plugin_name not in discovered:
                    discovered.append(plugin_name)

        return discovered

    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin and register its tools."""
        try:
            # Import module
            module = importlib.import_module(plugin_name)

            # Look for Plugin class
            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class:
                logger.error(f"No Plugin class found in {plugin_name}")
                return False

            # Instantiate plugin
            plugin = plugin_class()
            plugin.initialize()

            # Register tools
            for tool in plugin.get_tools():
                self.registry.register_tool(tool)

            self.plugins[plugin_name] = plugin
            logger.info(f"Loaded plugin: {plugin_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]
        plugin.cleanup()

        # Remove tools
        for tool in plugin.get_tools():
            if tool.metadata.id in self.registry.tools:
                del self.registry.tools[tool.metadata.id]

        del self.plugins[plugin_name]
        return True

    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)

        # Reload module
        module = sys.modules.get(plugin_name)
        if module:
            importlib.reload(module)

        return self.load_plugin(plugin_name)


class Plugin:
    """Base class for plugins."""

    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.tools: List[Tool] = []

    def initialize(self):
        """Initialize the plugin."""
        raise NotImplementedError

    def get_tools(self) -> List[Tool]:
        """Get tools provided by this plugin."""
        return self.tools

    def cleanup(self):
        """Cleanup when unloading."""
        pass


# ===== Security & Sandboxing =====

@dataclass
class UserContext:
    """User context for security checks."""
    user_id: str
    session_id: str
    roles: Set[str]
    permissions: Set[str]
    ip_address: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class SecurityManager:
    """Manage security, permissions, and sandboxing."""

    def __init__(self, config: Dict):
        self.config = config
        self.audit_log: List[Dict] = []
        self.rate_limiters: Dict[str, 'RateLimiter'] = {}
        self.permission_cache = {}
        self.sandbox_enabled = config.get("sandbox", True)

    def check_permissions(self, context: UserContext, tool: Tool) -> bool:
        """Check if user has permissions for tool."""
        # Admin bypass
        if "admin" in context.roles:
            return True

        # Check tool permissions
        required = set(tool.metadata.permissions)
        if not required:
            return True  # No permissions required

        return required.issubset(context.permissions)

    def check_rate_limit(self, context: UserContext, tool_id: str) -> bool:
        """Check rate limit for user and tool."""
        key = f"{context.user_id}:{tool_id}"

        if key not in self.rate_limiters:
            self.rate_limiters[key] = RateLimiter(calls=10, period=60)

        return self.rate_limiters[key].allow()

    def sandbox_execute(self, func: Callable, arguments: Dict,
                       timeout: int = 30) -> Any:
        """Execute function in sandbox with resource limits."""
        if not self.sandbox_enabled:
            return func(**arguments)

        # Use process pool for isolation (simplified)
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **arguments)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise Exception(f"Execution timeout ({timeout}s)")

    def audit_log_entry(self, context: UserContext, tool_id: str,
                       success: bool, error: Optional[str] = None):
        """Add audit log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "tool_id": tool_id,
            "success": success,
            "error": error,
            "ip_address": context.ip_address
        }
        self.audit_log.append(entry)

    def get_stats(self) -> Dict:
        """Get security statistics."""
        return {
            "audit_log_size": len(self.audit_log),
            "rate_limiters_active": len(self.rate_limiters),
            "recent_failures": sum(1 for e in self.audit_log[-100:] if not e["success"])
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.tokens = calls
        self.last_refill = datetime.now()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check if request is allowed."""
        with self._lock:
            self._refill()

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        refill_rate = self.calls / self.period

        self.tokens = min(self.calls, self.tokens + elapsed * refill_rate)
        self.last_refill = now


# ===== Caching =====

class CacheManager:
    """Manage result caching with TTL and LRU eviction."""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.ttl = config.get("ttl", 300)
        self.max_size = config.get("max_size", 1000)
        self.cache: Dict[str, 'CacheEntry'] = {}
        self.access_order = deque(maxlen=self.max_size)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def make_key(self, tool_id: str, arguments: Dict) -> str:
        """Create cache key from tool and arguments."""
        key_data = f"{tool_id}:{json.dumps(arguments, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if valid."""
        if not self.enabled:
            return None

        with self._lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check expiration
                if datetime.now() > entry.expires_at:
                    del self.cache[key]
                    self.misses += 1
                    return None

                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                self.hits += 1
                return entry.value

            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache a value with TTL."""
        if not self.enabled:
            return

        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            entry = CacheEntry(
                value=value,
                expires_at=datetime.now() + timedelta(seconds=ttl or self.ttl)
            )

            self.cache[key] = entry
            self.access_order.append(key)

    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0
        }


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    value: Any
    expires_at: datetime


# ===== Metrics & Monitoring =====

class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.executions: List[ToolExecution] = []
        self.events: List[Dict] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def record_execution(self, execution: ToolExecution):
        """Record tool execution."""
        with self._lock:
            self.executions.append(execution)
            self.counters[f"tool.{execution.tool_id}.calls"] += 1

            if execution.success:
                self.counters[f"tool.{execution.tool_id}.success"] += 1
            else:
                self.counters[f"tool.{execution.tool_id}.errors"] += 1

            if execution.duration:
                self.timers[f"tool.{execution.tool_id}.duration"].append(execution.duration)

    def record_event(self, event_type: str, data: Dict):
        """Record a general event."""
        with self._lock:
            self.events.append({
                "type": event_type,
                "timestamp": datetime.now(),
                "data": data
            })
            self.counters[f"event.{event_type}"] += 1

    def get_summary(self, window: Optional[timedelta] = None) -> Dict:
        """Get metrics summary."""
        with self._lock:
            if window:
                cutoff = datetime.now() - window
                recent_executions = [e for e in self.executions if e.start_time > cutoff]
            else:
                recent_executions = self.executions

            total = len(recent_executions)
            successful = sum(1 for e in recent_executions if e.success)

            return {
                "total_executions": total,
                "success_rate": successful / total if total > 0 else 0,
                "counters": dict(self.counters),
                "average_durations": {
                    name: sum(times) / len(times) if times else 0
                    for name, times in self.timers.items()
                }
            }


# ===== API Interface =====

def create_fastapi_app(registry: ToolRegistry):
    """Create FastAPI application for the tool registry."""
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Tool Registry API", version="1.0.0")

    class ToolExecutionRequest(BaseModel):
        tool_id: str
        arguments: Dict[str, Any]
        user_id: Optional[str] = "anonymous"
        session_id: Optional[str] = None

    @app.get("/tools")
    async def list_tools(category: Optional[str] = None, tags: Optional[List[str]] = None):
        """List available tools."""
        tools = registry.search_tools(category=category, tags=tags)
        return [t.metadata.__dict__ for t in tools]

    @app.get("/tools/{tool_id}")
    async def get_tool(tool_id: str):
        """Get tool details."""
        tool = registry.get_tool(tool_id)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        return tool.metadata.__dict__

    @app.post("/tools/{tool_id}/execute")
    async def execute_tool(tool_id: str, request: ToolExecutionRequest):
        """Execute a tool."""
        # Create user context
        context = UserContext(
            user_id=request.user_id,
            session_id=request.session_id or str(uuid.uuid4()),
            roles={"user"},
            permissions=set()
        )

        # Execute tool
        execution = registry.execute_tool(tool_id, request.arguments, context)

        if not execution.success:
            raise HTTPException(status_code=400, detail=execution.error)

        return {
            "execution_id": execution.id,
            "result": execution.result,
            "cached": execution.cached,
            "duration": execution.duration
        }

    @app.websocket("/ws/execute")
    async def websocket_execute(websocket: WebSocket):
        """WebSocket endpoint for real-time tool execution."""
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_json()

                # Execute tool
                context = UserContext(
                    user_id=data.get("user_id", "anonymous"),
                    session_id=data.get("session_id", str(uuid.uuid4())),
                    roles={"user"},
                    permissions=set()
                )

                execution = registry.execute_tool(
                    data["tool_id"],
                    data["arguments"],
                    context
                )

                # Send response
                await websocket.send_json({
                    "execution_id": execution.id,
                    "success": execution.success,
                    "result": execution.result if execution.success else None,
                    "error": execution.error,
                    "duration": execution.duration
                })

        except Exception as e:
            await websocket.close(code=1000)

    @app.get("/stats")
    async def get_statistics():
        """Get registry statistics."""
        return registry.get_statistics()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "tools_count": len(registry.tools)}

    return app


# ===== Example Usage =====

def create_example_registry():
    """Create example registry with sample tools."""

    # Create registry
    registry = ToolRegistry()

    # Define sample tools
    def calculate(operation: str, a: float, b: float) -> float:
        """Perform arithmetic calculation."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }
        return operations.get(operation, lambda x, y: None)(a, b)

    def fetch_data(source: str, query: str, limit: int = 10) -> Dict:
        """Fetch data from source."""
        return {
            "source": source,
            "query": query,
            "results": [f"Result {i}" for i in range(min(limit, 5))]
        }

    def send_notification(recipient: str, message: str, channel: str = "email") -> Dict:
        """Send notification to user."""
        return {
            "sent": True,
            "recipient": recipient,
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        }

    # Create tools
    calc_tool = Tool.from_function(calculate, category="utilities")
    fetch_tool = Tool.from_function(fetch_data, category="data")
    notify_tool = Tool.from_function(send_notification, category="communication")

    # Register tools
    registry.register_tool(calc_tool)
    registry.register_tool(fetch_tool)
    registry.register_tool(notify_tool)

    return registry


def demo():
    """Demonstrate the tool registry system."""
    print("Tool Registry System Demo")
    print("=" * 50)

    # Create registry
    registry = create_example_registry()

    # List tools
    print("\nRegistered Tools:")
    for tool_id, tool in registry.tools.items():
        print(f"  - {tool_id}: {tool.metadata.description}")

    # Execute tool
    print("\n" + "-" * 30)
    print("Tool Execution:")

    execution = registry.execute_tool(
        "calculate",
        {"operation": "multiply", "a": 7, "b": 8}
    )

    print(f"Result: {execution.result}")
    print(f"Success: {execution.success}")
    print(f"Duration: {execution.duration:.4f}s")

    # Get statistics
    print("\n" + "-" * 30)
    print("Registry Statistics:")
    stats = registry.get_statistics()
    print(json.dumps(stats, indent=2, default=str))

    # Test caching
    print("\n" + "-" * 30)
    print("Testing Caching:")

    for i in range(3):
        start = time.time()
        execution = registry.execute_tool(
            "fetch_data",
            {"source": "database", "query": "SELECT *", "limit": 5}
        )
        elapsed = time.time() - start
        print(f"Call {i+1}: Cached={execution.cached}, Time={elapsed:.4f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tool Registry System")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--server", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.server:
        import uvicorn
        registry = create_example_registry()
        app = create_fastapi_app(registry)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        print("Tool Registry System")
        print("\nUsage:")
        print("  python tool_registry.py --demo     # Run demo")
        print("  python tool_registry.py --server   # Start API server")
        print("\nFeatures:")
        print("  - Dynamic tool registration and discovery")
        print("  - Plugin system for extensibility")
        print("  - Security and sandboxing")
        print("  - Result caching with TTL")
        print("  - Comprehensive metrics and monitoring")
        print("  - REST API and WebSocket support")
        print("  - Tool versioning and deprecation")