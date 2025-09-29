"""
Module 09: Function Calling - Function Orchestration

Learn to orchestrate complex multi-function workflows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
import time
from enum import Enum
from dataclasses import dataclass


# ===== Example 1: Function Chaining =====

def example_1_function_chaining():
    """Chain multiple functions together."""
    print("Example 1: Function Chaining")
    print("=" * 50)

    @dataclass
    class FunctionCall:
        """Represents a function call in a chain."""
        name: str
        arguments: Dict[str, Any]
        output_mapping: Optional[Dict[str, str]] = None
        condition: Optional[str] = None

    class FunctionChain:
        """Execute functions in sequence."""

        def __init__(self):
            self.functions = {}
            self.chains = {}

        def register_function(self, name: str, func: Callable):
            """Register a function."""
            self.functions[name] = func

        def create_chain(self, chain_name: str, steps: List[FunctionCall]):
            """Create a named chain of function calls."""
            self.chains[chain_name] = steps

        def execute_chain(self, chain_name: str, initial_input: Dict) -> List[Dict]:
            """Execute a chain of functions."""
            if chain_name not in self.chains:
                raise ValueError(f"Chain {chain_name} not found")

            results = []
            context = initial_input.copy()

            for step in self.chains[chain_name]:
                print(f"\nExecuting: {step.name}")

                # Check condition if specified
                if step.condition and not self._evaluate_condition(step.condition, context):
                    print(f"Skipping due to condition: {step.condition}")
                    continue

                # Prepare arguments from context
                args = self._prepare_arguments(step.arguments, context)
                print(f"Arguments: {json.dumps(args, indent=2)}")

                # Execute function
                try:
                    result = self.functions[step.name](**args)
                    print(f"Result: {json.dumps(result, indent=2)}")

                    # Store result
                    results.append({
                        "function": step.name,
                        "input": args,
                        "output": result
                    })

                    # Update context with output mapping
                    if step.output_mapping:
                        for output_key, context_key in step.output_mapping.items():
                            if isinstance(result, dict) and output_key in result:
                                context[context_key] = result[output_key]
                            else:
                                context[context_key] = result

                except Exception as e:
                    print(f"Error: {e}")
                    results.append({
                        "function": step.name,
                        "input": args,
                        "error": str(e)
                    })
                    break

            return results

        def _prepare_arguments(self, arg_template: Dict, context: Dict) -> Dict:
            """Prepare arguments from template and context."""
            args = {}
            for key, value in arg_template.items():
                if isinstance(value, str) and value.startswith("$"):
                    # Reference to context variable
                    context_key = value[1:]
                    args[key] = context.get(context_key, value)
                else:
                    args[key] = value
            return args

        def _evaluate_condition(self, condition: str, context: Dict) -> bool:
            """Evaluate a simple condition."""
            try:
                # Simple evaluation (in production, use safe eval)
                return eval(condition, {"__builtins__": {}}, context)
            except:
                return True

    # Create chain executor
    chain_executor = FunctionChain()

    # Register functions
    def fetch_user_data(user_id: str) -> Dict:
        """Fetch user data."""
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "subscription": "premium"
        }

    def check_permissions(user_id: str, subscription: str) -> Dict:
        """Check user permissions."""
        permissions = {
            "free": ["read"],
            "premium": ["read", "write", "delete"],
            "enterprise": ["read", "write", "delete", "admin"]
        }
        return {
            "user_id": user_id,
            "permissions": permissions.get(subscription, [])
        }

    def generate_report(user_id: str, name: str, permissions: List[str]) -> Dict:
        """Generate user report."""
        return {
            "report_id": f"RPT-{user_id}",
            "user_name": name,
            "permission_count": len(permissions),
            "has_write_access": "write" in permissions,
            "generated_at": datetime.now().isoformat()
        }

    def send_notification(email: str, report_id: str) -> Dict:
        """Send notification."""
        return {
            "status": "sent",
            "recipient": email,
            "report_id": report_id,
            "sent_at": datetime.now().isoformat()
        }

    chain_executor.register_function("fetch_user", fetch_user_data)
    chain_executor.register_function("check_permissions", check_permissions)
    chain_executor.register_function("generate_report", generate_report)
    chain_executor.register_function("send_notification", send_notification)

    # Define chain
    user_report_chain = [
        FunctionCall(
            name="fetch_user",
            arguments={"user_id": "$user_id"},
            output_mapping={
                "name": "user_name",
                "email": "user_email",
                "subscription": "subscription_type"
            }
        ),
        FunctionCall(
            name="check_permissions",
            arguments={
                "user_id": "$user_id",
                "subscription": "$subscription_type"
            },
            output_mapping={"permissions": "user_permissions"}
        ),
        FunctionCall(
            name="generate_report",
            arguments={
                "user_id": "$user_id",
                "name": "$user_name",
                "permissions": "$user_permissions"
            },
            output_mapping={
                "report_id": "report_id",
                "has_write_access": "can_write"
            }
        ),
        FunctionCall(
            name="send_notification",
            arguments={
                "email": "$user_email",
                "report_id": "$report_id"
            },
            condition="can_write == True"  # Only send if user has write access
        )
    ]

    chain_executor.create_chain("user_report", user_report_chain)

    # Execute chain
    print("\n" + "=" * 50)
    print("Executing User Report Chain")
    print("=" * 50)

    results = chain_executor.execute_chain("user_report", {"user_id": "USER123"})

    print("\n" + "=" * 50)
    print("Chain Execution Summary:")
    print("=" * 50)
    for result in results:
        print(f"\nFunction: {result['function']}")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Success: {json.dumps(result['output'], indent=2)}")


# ===== Example 2: Parallel Function Execution =====

def example_2_parallel_execution():
    """Execute multiple functions in parallel."""
    print("\nExample 2: Parallel Function Execution")
    print("=" * 50)

    class ParallelExecutor:
        """Execute functions in parallel using asyncio."""

        def __init__(self):
            self.async_functions = {}
            self.sync_functions = {}

        def register_async_function(self, name: str, func):
            """Register an async function."""
            self.async_functions[name] = func

        def register_sync_function(self, name: str, func):
            """Register a sync function."""
            self.sync_functions[name] = func

        async def execute_parallel(self, function_calls: List[Dict]) -> List[Dict]:
            """Execute multiple functions in parallel."""
            tasks = []

            for call in function_calls:
                func_name = call["name"]
                args = call.get("arguments", {})

                if func_name in self.async_functions:
                    task = self.async_functions[func_name](**args)
                elif func_name in self.sync_functions:
                    # Wrap sync function in async
                    task = asyncio.to_thread(self.sync_functions[func_name], **args)
                else:
                    task = self._error_task(f"Function {func_name} not found")

                tasks.append(self._wrap_task(func_name, task))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        async def _wrap_task(self, name: str, task):
            """Wrap task with metadata."""
            start_time = time.time()
            try:
                result = await task
                return {
                    "function": name,
                    "status": "success",
                    "result": result,
                    "execution_time": time.time() - start_time
                }
            except Exception as e:
                return {
                    "function": name,
                    "status": "error",
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }

        async def _error_task(self, message: str):
            """Create an error task."""
            raise ValueError(message)

        async def execute_batched(self, function_calls: List[Dict], batch_size: int = 3) -> List[Dict]:
            """Execute functions in batches."""
            results = []

            for i in range(0, len(function_calls), batch_size):
                batch = function_calls[i:i + batch_size]
                print(f"\nExecuting batch {i // batch_size + 1}...")
                batch_results = await self.execute_parallel(batch)
                results.extend(batch_results)

            return results

    # Create parallel executor
    executor = ParallelExecutor()

    # Define async functions
    async def fetch_api_data(api_name: str, delay: float = 1.0) -> Dict:
        """Simulate API call."""
        print(f"Fetching from {api_name}...")
        await asyncio.sleep(delay)
        return {
            "api": api_name,
            "data": f"Data from {api_name}",
            "timestamp": datetime.now().isoformat()
        }

    async def process_data(data_source: str) -> Dict:
        """Process data asynchronously."""
        print(f"Processing {data_source}...")
        await asyncio.sleep(0.5)
        return {
            "source": data_source,
            "processed": True,
            "record_count": 100
        }

    # Define sync functions
    def calculate_metrics(values: List[float]) -> Dict:
        """Calculate metrics synchronously."""
        print("Calculating metrics...")
        time.sleep(0.3)
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

    def validate_data(data: Dict) -> bool:
        """Validate data synchronously."""
        print("Validating data...")
        time.sleep(0.2)
        return bool(data)

    # Register functions
    executor.register_async_function("fetch_api", fetch_api_data)
    executor.register_async_function("process", process_data)
    executor.register_sync_function("calculate", calculate_metrics)
    executor.register_sync_function("validate", validate_data)

    # Define parallel tasks
    parallel_tasks = [
        {"name": "fetch_api", "arguments": {"api_name": "UserAPI", "delay": 1.0}},
        {"name": "fetch_api", "arguments": {"api_name": "ProductAPI", "delay": 1.5}},
        {"name": "fetch_api", "arguments": {"api_name": "OrderAPI", "delay": 0.8}},
        {"name": "process", "arguments": {"data_source": "Database"}},
        {"name": "calculate", "arguments": {"values": [10, 20, 30, 40, 50]}},
        {"name": "validate", "arguments": {"data": {"test": "data"}}}
    ]

    # Execute in parallel
    async def run_parallel_example():
        print("\nExecuting all tasks in parallel:")
        print("=" * 30)
        start_time = time.time()
        results = await executor.execute_parallel(parallel_tasks)
        total_time = time.time() - start_time

        print(f"\nCompleted in {total_time:.2f} seconds")
        print("\nResults:")
        for result in results:
            print(f"\n{result['function']}:")
            print(f"  Status: {result['status']}")
            print(f"  Execution Time: {result['execution_time']:.2f}s")
            if result['status'] == 'success':
                print(f"  Result: {json.dumps(result['result'], indent=4)}")
            else:
                print(f"  Error: {result['error']}")

        # Compare with sequential execution time
        sequential_time = sum(r['execution_time'] for r in results)
        print(f"\nSpeedup: {sequential_time / total_time:.2f}x")

    # Execute batched
    async def run_batched_example():
        print("\n" + "=" * 50)
        print("Executing tasks in batches (size=2):")
        print("=" * 30)
        start_time = time.time()
        results = await executor.execute_batched(parallel_tasks, batch_size=2)
        total_time = time.time() - start_time

        print(f"\nCompleted in {total_time:.2f} seconds")
        print(f"Results: {len(results)} functions executed")

    # Run async examples
    asyncio.run(run_parallel_example())
    asyncio.run(run_batched_example())


# ===== Example 3: Conditional Execution =====

def example_3_conditional_execution():
    """Execute functions based on conditions."""
    print("\nExample 3: Conditional Execution")
    print("=" * 50)

    class ConditionalOrchestrator:
        """Orchestrate functions with conditional logic."""

        def __init__(self):
            self.functions = {}
            self.workflows = {}

        def register_function(self, name: str, func):
            """Register a function."""
            self.functions[name] = func

        def create_workflow(self, name: str, definition: Dict):
            """Create a conditional workflow."""
            self.workflows[name] = definition

        def execute_workflow(self, workflow_name: str, context: Dict) -> Dict:
            """Execute a workflow with conditional logic."""
            if workflow_name not in self.workflows:
                raise ValueError(f"Workflow {workflow_name} not found")

            workflow = self.workflows[workflow_name]
            return self._execute_node(workflow["root"], context)

        def _execute_node(self, node: Dict, context: Dict) -> Dict:
            """Execute a single node in the workflow."""
            node_type = node["type"]

            if node_type == "function":
                return self._execute_function(node, context)
            elif node_type == "condition":
                return self._execute_condition(node, context)
            elif node_type == "switch":
                return self._execute_switch(node, context)
            elif node_type == "loop":
                return self._execute_loop(node, context)
            elif node_type == "parallel":
                return self._execute_parallel_node(node, context)
            else:
                raise ValueError(f"Unknown node type: {node_type}")

        def _execute_function(self, node: Dict, context: Dict) -> Dict:
            """Execute a function node."""
            func_name = node["function"]
            args = self._resolve_arguments(node.get("arguments", {}), context)

            print(f"\nExecuting function: {func_name}")
            print(f"Arguments: {json.dumps(args, indent=2)}")

            if func_name not in self.functions:
                return {"error": f"Function {func_name} not found"}

            try:
                result = self.functions[func_name](**args)
                print(f"Result: {json.dumps(result, indent=2)}")

                # Update context if mapping specified
                if "output" in node:
                    context[node["output"]] = result

                return {"status": "success", "result": result}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        def _execute_condition(self, node: Dict, context: Dict) -> Dict:
            """Execute conditional branch."""
            condition = node["condition"]
            print(f"\nEvaluating condition: {condition}")

            # Evaluate condition
            if self._evaluate_expression(condition, context):
                print("Condition is True")
                return self._execute_node(node["then"], context)
            elif "else" in node:
                print("Condition is False")
                return self._execute_node(node["else"], context)
            else:
                return {"status": "skipped"}

        def _execute_switch(self, node: Dict, context: Dict) -> Dict:
            """Execute switch statement."""
            value = self._resolve_value(node["value"], context)
            print(f"\nSwitch on value: {value}")

            cases = node["cases"]
            if value in cases:
                print(f"Executing case: {value}")
                return self._execute_node(cases[value], context)
            elif "default" in node:
                print("Executing default case")
                return self._execute_node(node["default"], context)
            else:
                return {"status": "no_match"}

        def _execute_loop(self, node: Dict, context: Dict) -> Dict:
            """Execute loop."""
            items = self._resolve_value(node["items"], context)
            var_name = node.get("variable", "item")
            results = []

            print(f"\nLooping over {len(items)} items")
            for i, item in enumerate(items):
                print(f"\nIteration {i + 1}:")
                loop_context = context.copy()
                loop_context[var_name] = item
                loop_context["index"] = i

                result = self._execute_node(node["body"], loop_context)
                results.append(result)

                # Check for break condition
                if "break_if" in node and self._evaluate_expression(node["break_if"], loop_context):
                    print("Breaking loop")
                    break

            return {"status": "success", "results": results}

        def _execute_parallel_node(self, node: Dict, context: Dict) -> Dict:
            """Execute parallel branches."""
            branches = node["branches"]
            results = {}

            print(f"\nExecuting {len(branches)} branches in parallel")
            for branch_name, branch_node in branches.items():
                print(f"\nBranch: {branch_name}")
                results[branch_name] = self._execute_node(branch_node, context.copy())

            return {"status": "success", "branches": results}

        def _resolve_arguments(self, args: Dict, context: Dict) -> Dict:
            """Resolve arguments from context."""
            resolved = {}
            for key, value in args.items():
                resolved[key] = self._resolve_value(value, context)
            return resolved

        def _resolve_value(self, value: Any, context: Dict) -> Any:
            """Resolve a value from context."""
            if isinstance(value, str) and value.startswith("$"):
                return context.get(value[1:], value)
            return value

        def _evaluate_expression(self, expr: str, context: Dict) -> bool:
            """Evaluate an expression."""
            # Simple evaluation (in production, use safe eval)
            try:
                return eval(expr, {"__builtins__": {}}, context)
            except:
                return False

    # Create orchestrator
    orchestrator = ConditionalOrchestrator()

    # Register functions
    def analyze_sentiment(text: str) -> Dict:
        """Analyze text sentiment."""
        # Mock sentiment analysis
        if "happy" in text.lower() or "good" in text.lower():
            sentiment = "positive"
            score = 0.8
        elif "sad" in text.lower() or "bad" in text.lower():
            sentiment = "negative"
            score = -0.7
        else:
            sentiment = "neutral"
            score = 0.1

        return {"sentiment": sentiment, "score": score}

    def escalate_to_human(message: str, priority: str) -> Dict:
        """Escalate to human agent."""
        return {
            "escalated": True,
            "priority": priority,
            "message": message,
            "agent_assigned": "Agent001"
        }

    def send_automated_response(sentiment: str) -> Dict:
        """Send automated response based on sentiment."""
        responses = {
            "positive": "Thank you for your positive feedback!",
            "negative": "We're sorry to hear about your experience. How can we help?",
            "neutral": "Thank you for reaching out. How can we assist you?"
        }
        return {
            "response": responses.get(sentiment, "Thank you for contacting us."),
            "automated": True
        }

    def log_interaction(data: Dict) -> Dict:
        """Log the interaction."""
        return {
            "logged": True,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    orchestrator.register_function("analyze_sentiment", analyze_sentiment)
    orchestrator.register_function("escalate_to_human", escalate_to_human)
    orchestrator.register_function("send_response", send_automated_response)
    orchestrator.register_function("log_interaction", log_interaction)

    # Define complex workflow
    customer_service_workflow = {
        "name": "customer_service",
        "root": {
            "type": "function",
            "function": "analyze_sentiment",
            "arguments": {"text": "$message"},
            "output": "sentiment_result"
        }
    }

    # Simpler workflow with conditions
    sentiment_response_workflow = {
        "name": "sentiment_response",
        "root": {
            "type": "parallel",
            "branches": {
                "analyze": {
                    "type": "function",
                    "function": "analyze_sentiment",
                    "arguments": {"text": "$message"},
                    "output": "sentiment_result"
                },
                "log": {
                    "type": "function",
                    "function": "log_interaction",
                    "arguments": {"data": {"message": "$message"}}
                }
            }
        }
    }

    # Register workflows
    orchestrator.create_workflow("customer_service", customer_service_workflow)
    orchestrator.create_workflow("sentiment_response", sentiment_response_workflow)

    # Test workflows
    test_messages = [
        "This product is really good and I'm happy with it!",
        "This is terrible, I'm very sad and disappointed",
        "Can you provide more information about your services?"
    ]

    for message in test_messages:
        print("\n" + "=" * 50)
        print(f"Processing message: {message}")
        print("=" * 50)

        context = {"message": message}
        result = orchestrator.execute_workflow("sentiment_response", context)

        print("\nWorkflow Result:")
        print(json.dumps(result, indent=2))

        print("\nContext after execution:")
        print(json.dumps(context, indent=2))


# ===== Example 4: State Management =====

def example_4_state_management():
    """Manage state across function calls."""
    print("\nExample 4: State Management")
    print("=" * 50)

    class StatefulOrchestrator:
        """Orchestrator with state management."""

        def __init__(self):
            self.functions = {}
            self.global_state = {}
            self.session_states = {}
            self.state_history = []

        def register_function(self, name: str, func):
            """Register a stateful function."""
            self.functions[name] = func

        def create_session(self, session_id: str, initial_state: Optional[Dict] = None):
            """Create a new session with initial state."""
            self.session_states[session_id] = {
                "id": session_id,
                "state": initial_state or {},
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "function_calls": []
            }
            return session_id

        def execute_in_session(self, session_id: str, function_name: str, arguments: Dict) -> Dict:
            """Execute function within a session context."""
            if session_id not in self.session_states:
                raise ValueError(f"Session {session_id} not found")

            session = self.session_states[session_id]
            state = session["state"]

            # Prepare execution context
            context = {
                "session_id": session_id,
                "state": state,
                "global_state": self.global_state,
                "arguments": arguments
            }

            print(f"\nSession: {session_id}")
            print(f"Function: {function_name}")
            print(f"Current State: {json.dumps(state, indent=2)}")

            # Execute function
            if function_name not in self.functions:
                return {"error": f"Function {function_name} not found"}

            try:
                result = self.functions[function_name](context)

                # Update session
                session["updated_at"] = datetime.now()
                session["function_calls"].append({
                    "function": function_name,
                    "arguments": arguments,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

                # Record state change
                self._record_state_change(session_id, function_name, state.copy())

                print(f"Result: {json.dumps(result, indent=2)}")
                print(f"Updated State: {json.dumps(state, indent=2)}")

                return {"status": "success", "result": result, "state": state}

            except Exception as e:
                return {"status": "error", "error": str(e)}

        def get_session_state(self, session_id: str) -> Dict:
            """Get current session state."""
            if session_id not in self.session_states:
                raise ValueError(f"Session {session_id} not found")
            return self.session_states[session_id]["state"]

        def merge_sessions(self, session_ids: List[str], new_session_id: str) -> str:
            """Merge multiple sessions into one."""
            merged_state = {}
            merged_calls = []

            for sid in session_ids:
                if sid in self.session_states:
                    session = self.session_states[sid]
                    merged_state.update(session["state"])
                    merged_calls.extend(session["function_calls"])

            self.create_session(new_session_id, merged_state)
            self.session_states[new_session_id]["function_calls"] = merged_calls

            return new_session_id

        def _record_state_change(self, session_id: str, function_name: str, state_before: Dict):
            """Record state change for history."""
            self.state_history.append({
                "session_id": session_id,
                "function": function_name,
                "state_before": state_before,
                "state_after": self.session_states[session_id]["state"].copy(),
                "timestamp": datetime.now().isoformat()
            })

        def get_state_history(self, session_id: Optional[str] = None) -> List[Dict]:
            """Get state history, optionally filtered by session."""
            if session_id:
                return [h for h in self.state_history if h["session_id"] == session_id]
            return self.state_history

    # Create stateful orchestrator
    orchestrator = StatefulOrchestrator()

    # Define stateful functions
    def initialize_cart(context: Dict) -> Dict:
        """Initialize shopping cart."""
        state = context["state"]
        state["cart"] = {
            "items": [],
            "total": 0.0,
            "discount": 0.0
        }
        state["user_id"] = context["arguments"].get("user_id", "anonymous")
        return {"initialized": True, "user_id": state["user_id"]}

    def add_item_to_cart(context: Dict) -> Dict:
        """Add item to cart."""
        state = context["state"]
        args = context["arguments"]

        if "cart" not in state:
            return {"error": "Cart not initialized"}

        item = {
            "id": args["item_id"],
            "name": args["name"],
            "price": args["price"],
            "quantity": args.get("quantity", 1)
        }

        state["cart"]["items"].append(item)
        state["cart"]["total"] += item["price"] * item["quantity"]

        return {
            "added": True,
            "item": item,
            "cart_total": state["cart"]["total"],
            "item_count": len(state["cart"]["items"])
        }

    def apply_discount(context: Dict) -> Dict:
        """Apply discount to cart."""
        state = context["state"]
        args = context["arguments"]

        if "cart" not in state:
            return {"error": "Cart not initialized"}

        discount_percent = args["discount_percent"]
        state["cart"]["discount"] = state["cart"]["total"] * (discount_percent / 100)

        return {
            "discount_applied": True,
            "discount_amount": state["cart"]["discount"],
            "final_total": state["cart"]["total"] - state["cart"]["discount"]
        }

    def checkout(context: Dict) -> Dict:
        """Process checkout."""
        state = context["state"]

        if "cart" not in state or not state["cart"]["items"]:
            return {"error": "Cart is empty"}

        final_total = state["cart"]["total"] - state["cart"]["discount"]

        order = {
            "order_id": f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": state.get("user_id", "anonymous"),
            "items": state["cart"]["items"],
            "subtotal": state["cart"]["total"],
            "discount": state["cart"]["discount"],
            "total": final_total,
            "status": "confirmed"
        }

        # Clear cart after checkout
        state["last_order"] = order
        state["cart"] = {"items": [], "total": 0.0, "discount": 0.0}

        return order

    # Register functions
    orchestrator.register_function("init_cart", initialize_cart)
    orchestrator.register_function("add_item", add_item_to_cart)
    orchestrator.register_function("apply_discount", apply_discount)
    orchestrator.register_function("checkout", checkout)

    # Create shopping session
    session_id = orchestrator.create_session("shopping_session_1")

    # Execute shopping workflow
    print("\n" + "=" * 50)
    print("Shopping Cart Workflow")
    print("=" * 50)

    # Initialize cart
    orchestrator.execute_in_session(session_id, "init_cart", {"user_id": "USER123"})

    # Add items
    orchestrator.execute_in_session(session_id, "add_item", {
        "item_id": "PROD001",
        "name": "Laptop",
        "price": 999.99,
        "quantity": 1
    })

    orchestrator.execute_in_session(session_id, "add_item", {
        "item_id": "PROD002",
        "name": "Mouse",
        "price": 29.99,
        "quantity": 2
    })

    # Apply discount
    orchestrator.execute_in_session(session_id, "apply_discount", {
        "discount_percent": 10
    })

    # Checkout
    result = orchestrator.execute_in_session(session_id, "checkout", {})

    # Show final state
    print("\n" + "=" * 50)
    print("Final Session State:")
    print("=" * 50)
    print(json.dumps(orchestrator.get_session_state(session_id), indent=2))

    # Show state history
    print("\n" + "=" * 50)
    print("State History:")
    print("=" * 50)
    history = orchestrator.get_state_history(session_id)
    for entry in history:
        print(f"\nFunction: {entry['function']}")
        print(f"Timestamp: {entry['timestamp']}")
        if "cart" in entry["state_after"]:
            print(f"Cart Total: ${entry['state_after']['cart']['total']:.2f}")


# ===== Example 5: Function Composition =====

def example_5_function_composition():
    """Compose functions to create new functions."""
    print("\nExample 5: Function Composition")
    print("=" * 50)

    class FunctionComposer:
        """Compose multiple functions into new functions."""

        def __init__(self):
            self.functions = {}
            self.compositions = {}

        def register_function(self, name: str, func):
            """Register a base function."""
            self.functions[name] = func

        def compose_sequential(self, name: str, function_names: List[str]):
            """Compose functions sequentially (output of one is input to next)."""
            def composed_function(x):
                result = x
                for func_name in function_names:
                    if func_name in self.functions:
                        result = self.functions[func_name](result)
                    else:
                        raise ValueError(f"Function {func_name} not found")
                return result

            self.compositions[name] = composed_function
            return composed_function

        def compose_parallel(self, name: str, function_names: List[str]):
            """Compose functions in parallel (all receive same input)."""
            def composed_function(x):
                results = {}
                for func_name in function_names:
                    if func_name in self.functions:
                        results[func_name] = self.functions[func_name](x)
                    else:
                        raise ValueError(f"Function {func_name} not found")
                return results

            self.compositions[name] = composed_function
            return composed_function

        def compose_branching(self, name: str, condition_func: str,
                            true_func: str, false_func: str):
            """Compose functions with branching logic."""
            def composed_function(x):
                if self.functions[condition_func](x):
                    return self.functions[true_func](x)
                else:
                    return self.functions[false_func](x)

            self.compositions[name] = composed_function
            return composed_function

        def compose_with_aggregation(self, name: str, function_names: List[str],
                                    aggregator: str):
            """Compose functions with result aggregation."""
            def composed_function(x):
                results = []
                for func_name in function_names:
                    if func_name in self.functions:
                        results.append(self.functions[func_name](x))

                # Apply aggregator
                if aggregator in self.functions:
                    return self.functions[aggregator](results)
                else:
                    return results

            self.compositions[name] = composed_function
            return composed_function

        def create_pipeline(self, name: str, pipeline_def: List[Dict]):
            """Create a complex pipeline from definition."""
            def pipeline_function(x):
                result = x
                context = {"input": x, "intermediate": {}}

                for step in pipeline_def:
                    step_type = step["type"]
                    step_name = step.get("name", "unnamed")

                    print(f"  Step: {step_name}")

                    if step_type == "function":
                        func_name = step["function"]
                        if func_name in self.functions:
                            result = self.functions[func_name](result)
                        elif func_name in self.compositions:
                            result = self.compositions[func_name](result)
                    elif step_type == "transform":
                        transform = step["transform"]
                        result = transform(result)
                    elif step_type == "filter":
                        filter_func = step["filter"]
                        if filter_func(result):
                            result = step.get("value", result)
                        else:
                            result = step.get("default", None)
                    elif step_type == "store":
                        context["intermediate"][step["key"]] = result
                    elif step_type == "load":
                        result = context["intermediate"].get(step["key"], result)

                return result

            self.compositions[name] = pipeline_function
            return pipeline_function

    # Create composer
    composer = FunctionComposer()

    # Register base functions
    def clean_text(text: str) -> str:
        """Clean text by removing extra spaces and lowercasing."""
        return " ".join(text.split()).lower()

    def extract_keywords(text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        stop_words = {"the", "is", "at", "in", "on", "and", "a", "to"}
        words = text.split()
        return [w for w in words if w not in stop_words and len(w) > 2]

    def score_relevance(keywords: List[str]) -> float:
        """Score relevance based on keywords."""
        important_keywords = {"python", "programming", "code", "function", "data"}
        matches = sum(1 for kw in keywords if kw in important_keywords)
        return matches / len(keywords) if keywords else 0

    def format_result(data: Any) -> Dict:
        """Format result for output."""
        if isinstance(data, list):
            return {"type": "list", "count": len(data), "data": data}
        elif isinstance(data, (int, float)):
            return {"type": "number", "value": data}
        else:
            return {"type": "other", "data": str(data)}

    def is_long_text(text: str) -> bool:
        """Check if text is long."""
        return len(text) > 50

    def summarize(text: str) -> str:
        """Create summary of text."""
        words = text.split()
        if len(words) > 10:
            return " ".join(words[:10]) + "..."
        return text

    def expand(text: str) -> str:
        """Expand text with additional context."""
        return f"Expanded content: {text}. Additional context provided."

    def aggregate_scores(scores: List[float]) -> float:
        """Aggregate multiple scores."""
        return sum(scores) / len(scores) if scores else 0

    composer.register_function("clean", clean_text)
    composer.register_function("extract", extract_keywords)
    composer.register_function("score", score_relevance)
    composer.register_function("format", format_result)
    composer.register_function("is_long", is_long_text)
    composer.register_function("summarize", summarize)
    composer.register_function("expand", expand)
    composer.register_function("aggregate", aggregate_scores)

    # Create compositions
    print("Creating Function Compositions:")
    print("-" * 30)

    # Sequential composition
    text_processor = composer.compose_sequential(
        "text_processor",
        ["clean", "extract", "format"]
    )

    # Parallel composition
    multi_analyzer = composer.compose_parallel(
        "multi_analyzer",
        ["clean", "extract", "summarize"]
    )

    # Branching composition
    adaptive_processor = composer.compose_branching(
        "adaptive_processor",
        condition_func="is_long",
        true_func="summarize",
        false_func="expand"
    )

    # Complex pipeline
    analysis_pipeline = composer.create_pipeline(
        "analysis_pipeline",
        [
            {"type": "function", "name": "clean", "function": "clean"},
            {"type": "store", "key": "cleaned"},
            {"type": "function", "name": "extract", "function": "extract"},
            {"type": "store", "key": "keywords"},
            {"type": "function", "name": "score", "function": "score"},
            {"type": "filter", "name": "check_relevance",
             "filter": lambda x: x > 0.3,
             "value": "Relevant",
             "default": "Not relevant"},
            {"type": "transform", "name": "finalize",
             "transform": lambda x: {"relevance": x, "status": "processed"}}
        ]
    )

    # Test compositions
    test_texts = [
        "Python programming is great for data analysis and machine learning",
        "The quick brown fox jumps over the lazy dog",
        "Code functions help organize your programming logic"
    ]

    for text in test_texts:
        print(f"\nInput: {text}")
        print("-" * 50)

        # Sequential
        result = text_processor(text)
        print(f"Sequential: {json.dumps(result, indent=2)}")

        # Parallel
        result = multi_analyzer(text)
        print(f"Parallel: {json.dumps(result, indent=2)}")

        # Branching
        result = adaptive_processor(text)
        print(f"Branching: {result}")

        # Pipeline
        print("Pipeline execution:")
        result = analysis_pipeline(text)
        print(f"Result: {json.dumps(result, indent=2)}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 09: Function Orchestration Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_function_chaining,
        2: example_2_parallel_execution,
        3: example_3_conditional_execution,
        4: example_4_state_management,
        5: example_5_function_composition
    }

    if args.all:
        for example in examples.values():
            example()
            print("\n" + "=" * 70 + "\n")
    elif args.example and args.example in examples:
        examples[args.example]()
    else:
        print("Module 09: Function Orchestration - Examples")
        print("\nUsage:")
        print("  python function_orchestration.py --example N  # Run example N")
        print("  python function_orchestration.py --all         # Run all examples")
        print("\nAvailable examples:")
        print("  1: Function Chaining")
        print("  2: Parallel Function Execution")
        print("  3: Conditional Execution")
        print("  4: State Management")
        print("  5: Function Composition")