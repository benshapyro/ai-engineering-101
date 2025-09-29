"""
Module 05: Workflow Orchestrator

A production-ready system for managing complex prompt chains with
state management, error handling, and performance optimization.
"""

import json
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class TaskStatus(Enum):
    """Status of individual tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class Task:
    """Individual task in a workflow."""
    id: str
    name: str
    prompt: str
    dependencies: List[str] = field(default_factory=list)
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_retries: int = 3
    timeout: float = 30.0
    cache_ttl: int = 300  # seconds
    validators: List[Callable] = field(default_factory=list)
    fallback: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    status: TaskStatus
    results: Dict[str, Any]
    errors: List[Dict]
    metrics: Dict
    start_time: datetime
    end_time: datetime
    total_duration: float


class StateManager:
    """Manages state across workflow execution."""

    def __init__(self):
        self.global_state = {}
        self.task_results = {}
        self.execution_history = []
        self.checkpoints = []

    def set(self, key: str, value: Any):
        """Set a value in global state."""
        self.global_state[key] = value
        self.execution_history.append({
            "action": "set",
            "key": key,
            "value": value,
            "timestamp": datetime.now()
        })

    def get(self, key: str, default=None):
        """Get a value from global state."""
        return self.global_state.get(key, default)

    def add_task_result(self, task_id: str, result: Any):
        """Store task result."""
        self.task_results[task_id] = {
            "result": result,
            "timestamp": datetime.now()
        }

    def get_task_result(self, task_id: str):
        """Retrieve task result."""
        return self.task_results.get(task_id, {}).get("result")

    def create_checkpoint(self, name: str):
        """Create a state checkpoint."""
        checkpoint = {
            "name": name,
            "timestamp": datetime.now(),
            "global_state": self.global_state.copy(),
            "task_results": self.task_results.copy()
        }
        self.checkpoints.append(checkpoint)
        return checkpoint

    def restore_checkpoint(self, name: str):
        """Restore from a checkpoint."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint["name"] == name:
                self.global_state = checkpoint["global_state"].copy()
                self.task_results = checkpoint["task_results"].copy()
                return True
        return False

    def get_context(self) -> Dict:
        """Get current execution context."""
        return {
            "global_state": self.global_state,
            "completed_tasks": list(self.task_results.keys()),
            "history_length": len(self.execution_history)
        }


class CacheManager:
    """Manages caching of task results."""

    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, task: Task, context: Dict) -> str:
        """Generate cache key for a task."""
        cache_input = f"{task.id}:{task.prompt}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def get(self, task: Task, context: Dict) -> Optional[Any]:
        """Get cached result if available."""
        key = self._generate_key(task, context)

        if key in self.cache:
            entry = self.cache[key]
            age = (datetime.now() - entry["timestamp"]).seconds

            if age < task.cache_ttl:
                self.hit_count += 1
                logger.debug(f"Cache hit for task {task.id}")
                return entry["result"]
            else:
                # Expired
                del self.cache[key]

        self.miss_count += 1
        return None

    def set(self, task: Task, context: Dict, result: Any):
        """Cache a task result."""
        key = self._generate_key(task, context)
        self.cache[key] = {
            "result": result,
            "timestamp": datetime.now()
        }

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


class ErrorHandler:
    """Handles errors and implements recovery strategies."""

    def __init__(self):
        self.error_log = []
        self.retry_delays = [1, 2, 4, 8]  # Exponential backoff

    def log_error(self, task_id: str, error: Exception, attempt: int):
        """Log an error."""
        self.error_log.append({
            "task_id": task_id,
            "error": str(error),
            "type": type(error).__name__,
            "attempt": attempt,
            "timestamp": datetime.now()
        })

    def should_retry(self, task: Task, attempt: int) -> bool:
        """Determine if task should be retried."""
        return attempt < task.max_retries

    def get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry."""
        if attempt < len(self.retry_delays):
            return self.retry_delays[attempt]
        return self.retry_delays[-1]

    def get_fallback_strategy(self, task: Task) -> Optional[str]:
        """Get fallback strategy for failed task."""
        return task.fallback

    def get_error_summary(self) -> Dict:
        """Get summary of errors."""
        error_counts = defaultdict(int)
        for error in self.error_log:
            error_counts[error["type"]] += 1

        return {
            "total_errors": len(self.error_log),
            "error_types": dict(error_counts),
            "recent_errors": self.error_log[-5:]
        }


class WorkflowOrchestrator:
    """Main orchestrator for managing workflow execution."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.state_manager = StateManager()
        self.cache_manager = CacheManager()
        self.error_handler = ErrorHandler()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics = defaultdict(int)

    def execute_workflow(
        self,
        tasks: List[Task],
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        context: Optional[Dict] = None
    ) -> WorkflowResult:
        """Execute a complete workflow."""
        workflow_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        start_time = datetime.now()

        logger.info(f"Starting workflow {workflow_id} with {len(tasks)} tasks")

        # Initialize context
        if context:
            for key, value in context.items():
                self.state_manager.set(key, value)

        # Choose execution strategy
        if strategy == ExecutionStrategy.ADAPTIVE:
            strategy = self._select_strategy(tasks)

        logger.info(f"Using {strategy.value} execution strategy")

        # Execute based on strategy
        results = {}
        errors = []

        try:
            if strategy == ExecutionStrategy.SEQUENTIAL:
                results = self._execute_sequential(tasks)
            elif strategy == ExecutionStrategy.PARALLEL:
                results = self._execute_parallel(tasks)
            elif strategy == ExecutionStrategy.HYBRID:
                results = self._execute_hybrid(tasks)
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            errors.append({"error": str(e), "timestamp": datetime.now()})

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Create result
        return WorkflowResult(
            workflow_id=workflow_id,
            status=TaskStatus.COMPLETED if not errors else TaskStatus.FAILED,
            results=results,
            errors=errors + self.error_handler.error_log,
            metrics=self._get_metrics(),
            start_time=start_time,
            end_time=end_time,
            total_duration=duration
        )

    def _select_strategy(self, tasks: List[Task]) -> ExecutionStrategy:
        """Adaptively select execution strategy."""
        # Check for dependencies
        has_dependencies = any(task.dependencies for task in tasks)

        # Check task count
        task_count = len(tasks)

        if has_dependencies:
            # Tasks with dependencies need careful orchestration
            return ExecutionStrategy.HYBRID
        elif task_count > 5:
            # Many independent tasks benefit from parallelization
            return ExecutionStrategy.PARALLEL
        else:
            # Small workflows can be sequential
            return ExecutionStrategy.SEQUENTIAL

    def _execute_sequential(self, tasks: List[Task]) -> Dict:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            # Check dependencies
            if not self._dependencies_met(task):
                logger.warning(f"Skipping task {task.id} - dependencies not met")
                results[task.id] = None
                continue

            # Execute task
            result = self._execute_task(task)
            results[task.id] = result

            # Update state
            self.state_manager.add_task_result(task.id, result)

        return results

    def _execute_parallel(self, tasks: List[Task]) -> Dict:
        """Execute tasks in parallel."""
        results = {}
        futures = {}

        # Submit all tasks
        for task in tasks:
            if self._dependencies_met(task):
                future = self.executor.submit(self._execute_task, task)
                futures[future] = task.id
            else:
                results[task.id] = None

        # Collect results
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result(timeout=30)
                results[task_id] = result
                self.state_manager.add_task_result(task_id, result)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results[task_id] = None

        return results

    def _execute_hybrid(self, tasks: List[Task]) -> Dict:
        """Execute with hybrid strategy (parallel where possible)."""
        results = {}
        completed = set()

        # Execute in waves
        while len(completed) < len(tasks):
            # Find tasks that can be executed now
            ready_tasks = [
                task for task in tasks
                if task.id not in completed and
                all(dep in completed for dep in task.dependencies)
            ]

            if not ready_tasks:
                break

            # Execute ready tasks in parallel
            wave_results = self._execute_parallel(ready_tasks)
            results.update(wave_results)
            completed.update(task.id for task in ready_tasks)

        return results

    def _execute_task(self, task: Task) -> Any:
        """Execute a single task with error handling."""
        # Check cache
        context = self.state_manager.get_context()
        cached_result = self.cache_manager.get(task, context)
        if cached_result is not None:
            self.metrics["cache_hits"] += 1
            return cached_result

        # Prepare prompt with context
        prompt = self._prepare_prompt(task, context)

        # Execute with retries
        attempt = 0
        last_error = None

        while attempt < task.max_retries:
            try:
                # Simulate LLM call (replace with actual client)
                if self.llm_client:
                    result = self.llm_client.complete(prompt, temperature=0.3, max_tokens=200)
                else:
                    # Mock result for testing
                    result = f"Result for task {task.id}"

                # Validate result
                if task.validators:
                    for validator in task.validators:
                        if not validator(result):
                            raise ValueError(f"Validation failed for task {task.id}")

                # Cache successful result
                self.cache_manager.set(task, context, result)
                self.metrics["successful_tasks"] += 1

                return result

            except Exception as e:
                attempt += 1
                last_error = e
                self.error_handler.log_error(task.id, e, attempt)

                if self.error_handler.should_retry(task, attempt):
                    delay = self.error_handler.get_retry_delay(attempt)
                    logger.warning(f"Retrying task {task.id} in {delay}s (attempt {attempt})")
                    time.sleep(delay)
                else:
                    break

        # Try fallback
        if task.fallback:
            logger.info(f"Using fallback for task {task.id}")
            return task.fallback

        self.metrics["failed_tasks"] += 1
        raise last_error or Exception(f"Task {task.id} failed")

    def _prepare_prompt(self, task: Task, context: Dict) -> str:
        """Prepare prompt with context injection."""
        prompt_parts = []

        # Add context if available
        if context.get("global_state"):
            context_str = json.dumps(context["global_state"], indent=2)
            prompt_parts.append(f"Context:\n{context_str}\n")

        # Add dependencies results
        for dep_id in task.dependencies:
            dep_result = self.state_manager.get_task_result(dep_id)
            if dep_result:
                prompt_parts.append(f"Result from {dep_id}:\n{dep_result}\n")

        # Add main prompt
        prompt_parts.append(task.prompt)

        return "\n".join(prompt_parts)

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in task.dependencies:
            if self.state_manager.get_task_result(dep_id) is None:
                return False
        return True

    def _get_metrics(self) -> Dict:
        """Get execution metrics."""
        cache_stats = self.cache_manager.get_statistics()
        error_summary = self.error_handler.get_error_summary()

        return {
            "successful_tasks": self.metrics["successful_tasks"],
            "failed_tasks": self.metrics["failed_tasks"],
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "total_errors": error_summary["total_errors"],
            "error_types": error_summary["error_types"]
        }

    def create_checkpoint(self, name: str):
        """Create a workflow checkpoint."""
        return self.state_manager.create_checkpoint(name)

    def restore_checkpoint(self, name: str):
        """Restore from a checkpoint."""
        return self.state_manager.restore_checkpoint(name)


class WorkflowBuilder:
    """Builder for creating workflows."""

    def __init__(self):
        self.tasks = []
        self.global_config = {}

    def add_task(
        self,
        id: str,
        name: str,
        prompt: str,
        dependencies: List[str] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """Add a task to the workflow."""
        task = Task(
            id=id,
            name=name,
            prompt=prompt,
            dependencies=dependencies or [],
            **kwargs
        )
        self.tasks.append(task)
        return self

    def add_sequential_chain(
        self,
        tasks: List[Dict]
    ) -> 'WorkflowBuilder':
        """Add a sequential chain of tasks."""
        prev_id = None
        for task_config in tasks:
            if prev_id:
                task_config["dependencies"] = [prev_id]
            self.add_task(**task_config)
            prev_id = task_config["id"]
        return self

    def add_parallel_tasks(
        self,
        tasks: List[Dict]
    ) -> 'WorkflowBuilder':
        """Add parallel tasks."""
        for task_config in tasks:
            self.add_task(**task_config)
        return self

    def set_global_config(self, **kwargs) -> 'WorkflowBuilder':
        """Set global configuration."""
        self.global_config.update(kwargs)
        return self

    def build(self) -> List[Task]:
        """Build the workflow."""
        # Apply global config to tasks
        for task in self.tasks:
            for key, value in self.global_config.items():
                if not hasattr(task, key) or getattr(task, key) is None:
                    setattr(task, key, value)
        return self.tasks


# Example usage
if __name__ == "__main__":
    # Create workflow builder
    builder = WorkflowBuilder()

    # Build a complex workflow
    workflow = (
        builder
        .set_global_config(max_retries=2, cache_ttl=600)
        .add_task(
            id="extract",
            name="Extract Information",
            prompt="Extract key points from the document"
        )
        .add_task(
            id="analyze",
            name="Analyze Data",
            prompt="Analyze the extracted information",
            dependencies=["extract"]
        )
        .add_parallel_tasks([
            {
                "id": "sentiment",
                "name": "Sentiment Analysis",
                "prompt": "Analyze sentiment",
                "dependencies": ["analyze"]
            },
            {
                "id": "summary",
                "name": "Generate Summary",
                "prompt": "Create summary",
                "dependencies": ["analyze"]
            }
        ])
        .add_task(
            id="report",
            name="Generate Report",
            prompt="Create final report",
            dependencies=["sentiment", "summary"]
        )
        .build()
    )

    # Execute workflow
    orchestrator = WorkflowOrchestrator()
    result = orchestrator.execute_workflow(
        workflow,
        strategy=ExecutionStrategy.HYBRID,
        context={"document": "Sample document content"}
    )

    # Display results
    print(f"\nWorkflow {result.workflow_id} completed")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.total_duration:.2f} seconds")
    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print(f"\nResults:")
    for task_id, task_result in result.results.items():
        print(f"  {task_id}: {task_result[:50] if task_result else 'None'}...")