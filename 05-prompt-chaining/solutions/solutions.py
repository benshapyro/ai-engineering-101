"""
Module 05: Prompt Chaining - Solutions

Complete solutions for all prompt chaining exercises.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib


# ===== Solution 1: Sequential Pipeline =====

def solution_1_sequential_pipeline():
    """
    Solution 1: Multi-step data processing pipeline with state tracking.
    """
    client = LLMClient("openai")

    print("Solution 1: Sequential Pipeline")
    print("=" * 50)

    feedback = """
    The product quality is excellent but the shipping took forever!
    Customer service was unhelpful when I called. The packaging was
    damaged but the item inside was fine. Price is too high compared
    to competitors. Would maybe recommend to others.
    """

    print(f"Customer Feedback:\n{feedback}\n")

    # Initialize pipeline state
    pipeline_state = {
        "original": feedback,
        "step_results": [],
        "errors": [],
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Step 1: Extract key points
        print("STEP 1: Extracting key points...")
        step1_prompt = f"""Extract the key points from this customer feedback:

        Feedback: {feedback}

        Key Points (bullet list):"""

        key_points = client.complete(step1_prompt, temperature=0.2, max_tokens=150).strip()
        pipeline_state["step_results"].append({
            "step": 1,
            "name": "extract",
            "output": key_points
        })
        print(f"Extracted:\n{key_points}\n")

        # Step 2: Categorize issues
        print("STEP 2: Categorizing issues...")
        step2_prompt = f"""Categorize these key points by area:

        Key Points:
        {key_points}

        Categories (Product, Service, Shipping, Pricing):"""

        categorized = client.complete(step2_prompt, temperature=0.2, max_tokens=200).strip()
        pipeline_state["step_results"].append({
            "step": 2,
            "name": "categorize",
            "output": categorized
        })
        print(f"Categorized:\n{categorized}\n")

        # Step 3: Sentiment analysis per category
        print("STEP 3: Analyzing sentiment per category...")
        step3_prompt = f"""Analyze sentiment for each category:

        Categorized Issues:
        {categorized}

        Sentiment Analysis (category: sentiment score):"""

        sentiment = client.complete(step3_prompt, temperature=0.2, max_tokens=150).strip()
        pipeline_state["step_results"].append({
            "step": 3,
            "name": "sentiment",
            "output": sentiment
        })
        print(f"Sentiment:\n{sentiment}\n")

        # Step 4: Generate action items
        print("STEP 4: Generating action items...")
        step4_prompt = f"""Generate prioritized action items based on this analysis:

        Original Feedback: {feedback[:100]}...
        Key Issues: {key_points[:100]}...
        Sentiment: {sentiment}

        Action Items (prioritized):"""

        actions = client.complete(step4_prompt, temperature=0.3, max_tokens=200).strip()
        pipeline_state["step_results"].append({
            "step": 4,
            "name": "actions",
            "output": actions
        })
        print(f"Action Items:\n{actions}\n")

        # Final summary
        pipeline_state["success"] = True
        pipeline_state["final_output"] = actions

    except Exception as e:
        pipeline_state["errors"].append(str(e))
        pipeline_state["success"] = False
        print(f"Error in pipeline: {e}")

    print("-" * 40)
    print("Pipeline State Summary:")
    print(f"  Steps Completed: {len(pipeline_state['step_results'])}/4")
    print(f"  Success: {pipeline_state.get('success', False)}")
    print(f"  Errors: {len(pipeline_state['errors'])}")


# ===== Solution 2: Conditional Routing =====

def solution_2_conditional_routing():
    """
    Solution 2: Routing system with conditional logic and fallbacks.
    """
    client = LLMClient("openai")

    print("Solution 2: Conditional Routing")
    print("=" * 50)

    requests = [
        "How do I reset my password?",
        "I want to cancel my subscription immediately",
        "Can you explain your pricing plans?",
        "Bug: The app crashes when I upload photos"
    ]

    print("Requests to route:")
    for req in requests:
        print(f"  - {req}")

    # Process each request
    for request in requests[:3]:  # Process first 3 for demo
        print(f"\n" + "=" * 40)
        print(f"Processing: {request}")
        print("-" * 40)

        # Step 1: Multi-label classification
        classify_prompt = f"""Classify this request into categories:

        Request: {request}

        Primary Category: [TECHNICAL/BILLING/SALES/BUG_REPORT/GENERAL]
        Urgency: [HIGH/MEDIUM/LOW]
        Requires Human: [YES/NO]

        Classification:"""

        print("Classifying...")
        classification = client.complete(classify_prompt, temperature=0.1, max_tokens=50).strip()
        print(f"Classification: {classification}\n")

        # Parse classification (simplified)
        is_technical = "TECHNICAL" in classification
        is_billing = "BILLING" in classification
        is_sales = "SALES" in classification
        is_bug = "BUG_REPORT" in classification
        is_urgent = "HIGH" in classification

        # Step 2: Route based on classification
        routed = False

        if is_bug:
            print("→ ROUTING TO: Bug Report System")
            bug_prompt = f"""Create a bug report ticket:

            Description: {request}

            Ticket:
            - Title:
            - Severity:
            - Component:
            - Steps to Reproduce:
            - Expected Behavior:"""

            bug_ticket = client.complete(bug_prompt, temperature=0.2, max_tokens=150).strip()
            print(f"Bug Ticket:\n{bug_ticket}\n")
            routed = True

        elif is_technical:
            print("→ ROUTING TO: Technical Support")
            tech_prompt = f"""Provide technical solution:

            Question: {request}

            Solution:
            1. Step-by-step instructions:"""

            solution = client.complete(tech_prompt, temperature=0.3, max_tokens=150).strip()
            print(f"Technical Solution:\n{solution}\n")
            routed = True

        elif is_billing:
            print("→ ROUTING TO: Billing Department")
            billing_prompt = f"""Handle billing request:

            Request: {request}

            Response:
            - Action Required:
            - Account Status Check:
            - Options Available:"""

            billing_response = client.complete(billing_prompt, temperature=0.2, max_tokens=150).strip()
            print(f"Billing Response:\n{billing_response}\n")
            routed = True

        elif is_sales:
            print("→ ROUTING TO: Sales Team")
            sales_prompt = f"""Provide sales information:

            Question: {request}

            Sales Response:
            - Key Information:
            - Benefits:
            - Next Steps:"""

            sales_response = client.complete(sales_prompt, temperature=0.3, max_tokens=150).strip()
            print(f"Sales Response:\n{sales_response}\n")
            routed = True

        # Step 3: Fallback handling
        if not routed:
            print("→ FALLBACK: General Support")
            fallback_prompt = f"""Provide general assistance:

            Request: {request}

            General Response:"""

            fallback_response = client.complete(fallback_prompt, temperature=0.3, max_tokens=100).strip()
            print(f"General Response:\n{fallback_response}\n")

        # Urgent handling
        if is_urgent:
            print("⚠️ URGENT FLAG: Escalating to supervisor")
            escalate_prompt = f"""Create escalation notice:

            Urgent Request: {request}

            Escalation Summary:"""

            escalation = client.complete(escalate_prompt, temperature=0.2, max_tokens=50).strip()
            print(f"Escalation: {escalation}")


# ===== Solution 3: Parallel Processing =====

def solution_3_parallel_processing():
    """
    Solution 3: Parallel processing with aggregation.
    """
    client = LLMClient("openai")

    print("Solution 3: Parallel Processing")
    print("=" * 50)

    article = """
    The rise of remote work has fundamentally changed how companies operate.
    Productivity has increased for many workers, but collaboration challenges
    persist. Mental health concerns have grown as work-life boundaries blur.
    Companies are saving on office costs but investing more in digital tools.
    """

    print(f"Article:\n{article}\n")

    # Define parallel analysis tasks
    parallel_tasks = {
        "summary": f"Summarize this article in one sentence:\n\n{article}\n\nSummary:",
        "pros_cons": f"List the pros and cons mentioned in:\n\n{article}\n\nPros and Cons:",
        "predictions": f"Predict 3 future trends based on:\n\n{article}\n\nPredictions:",
        "questions": f"Generate 3 discussion questions about:\n\n{article}\n\nQuestions:",
        "keywords": f"Extract 5 keywords from:\n\n{article}\n\nKeywords:"
    }

    print("Executing 5 analyses in parallel...")
    start_time = time.time()
    results = {}

    def analyze_aspect(task_name, prompt):
        """Execute single analysis task."""
        try:
            result = client.complete(prompt, temperature=0.3, max_tokens=150)
            return task_name, result.strip()
        except Exception as e:
            return task_name, f"Error: {str(e)}"

    # Execute parallel analyses
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_aspect, name, prompt): name
            for name, prompt in parallel_tasks.items()
        }

        for future in as_completed(futures):
            task_name, result = future.result()
            results[task_name] = result
            print(f"  ✓ Completed: {task_name}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds\n")

    # Display results
    print("Individual Results:")
    print("-" * 40)
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(result[:150] + ("..." if len(result) > 150 else ""))

    # Aggregate results into final report
    print("\n" + "=" * 40)
    print("AGGREGATING RESULTS...")

    aggregate_prompt = f"""Create a comprehensive analysis report from these components:

    Summary: {results.get('summary', 'N/A')[:100]}
    Pros/Cons: {results.get('pros_cons', 'N/A')[:100]}
    Predictions: {results.get('predictions', 'N/A')[:100]}
    Keywords: {results.get('keywords', 'N/A')}

    Comprehensive Report:"""

    final_report = client.complete(aggregate_prompt, temperature=0.3, max_tokens=250).strip()
    print(f"\nFinal Report:\n{final_report}")


# ===== Solution 4: Error Recovery Chain =====

def solution_4_error_recovery():
    """
    Solution 4: Robust chain with error recovery mechanisms.
    """
    client = LLMClient("openai")

    print("Solution 4: Error Recovery Chain")
    print("=" * 50)

    document = """
    The system architecture needs refactoring. Database queries are slow.
    We should implement caching and add monitoring. Deploy by Friday.
    """

    print(f"Document: {document}\n")

    class ResilientChain:
        def __init__(self, client):
            self.client = client
            self.max_retries = 3
            self.retry_delay = 1
            self.execution_log = []

        def execute_with_retry(self, prompt, step_name):
            """Execute with retry logic and exponential backoff."""
            for attempt in range(self.max_retries):
                try:
                    print(f"  Attempt {attempt + 1} for {step_name}")
                    result = self.client.complete(prompt, temperature=0.3, max_tokens=200)
                    self.execution_log.append({
                        "step": step_name,
                        "attempt": attempt + 1,
                        "status": "success"
                    })
                    return result.strip()
                except Exception as e:
                    self.execution_log.append({
                        "step": step_name,
                        "attempt": attempt + 1,
                        "status": "failed",
                        "error": str(e)
                    })
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        print(f"    Failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"    Max retries reached for {step_name}")
                        raise

        def execute_with_fallback(self, primary_prompt, fallback_prompt, step_name):
            """Execute with fallback strategy."""
            try:
                print(f"  Primary strategy for {step_name}")
                result = self.execute_with_retry(primary_prompt, f"{step_name}_primary")
                return result
            except Exception as e:
                print(f"  Fallback strategy for {step_name}")
                try:
                    result = self.client.complete(fallback_prompt, temperature=0.3, max_tokens=150)
                    self.execution_log.append({
                        "step": f"{step_name}_fallback",
                        "status": "success"
                    })
                    return result.strip()
                except Exception as e2:
                    self.execution_log.append({
                        "step": f"{step_name}_fallback",
                        "status": "failed",
                        "error": str(e2)
                    })
                    return f"[Failed: {step_name}]"

        def run_chain(self, steps):
            """Run chain with comprehensive error recovery."""
            results = {}
            for step in steps:
                print(f"\nExecuting: {step['name']}")
                result = self.execute_with_fallback(
                    step["primary"],
                    step["fallback"],
                    step["name"]
                )
                results[step["name"]] = result
                print(f"  Result: {result[:50]}...")
            return results

    # Initialize resilient chain
    chain = ResilientChain(client)

    # Define chain steps with fallbacks
    chain_steps = [
        {
            "name": "translate",
            "primary": f"Translate to Spanish with technical accuracy:\n\n{document}\n\nSpanish:",
            "fallback": f"Simple Spanish translation of:\n\n{document}\n\nSpanish:"
        },
        {
            "name": "summarize",
            "primary": "Summarize the Spanish text in 2 sentences:\n\n[Previous result]\n\nSummary:",
            "fallback": f"Brief summary of:\n\n{document}\n\nSummary:"
        },
        {
            "name": "extract_actions",
            "primary": "Extract action items from summary:\n\n[Previous result]\n\nActions:",
            "fallback": f"List tasks mentioned in:\n\n{document}\n\nTasks:"
        },
        {
            "name": "format_email",
            "primary": "Format as professional email:\n\n[Previous results]\n\nEmail:",
            "fallback": "Create simple email about:\n\nSystem refactoring needed\n\nEmail:"
        }
    ]

    # Execute chain with error recovery
    results = chain.run_chain(chain_steps)

    # Display execution summary
    print("\n" + "=" * 40)
    print("EXECUTION SUMMARY:")
    print(f"Total attempts: {len(chain.execution_log)}")
    successful = sum(1 for log in chain.execution_log if log["status"] == "success")
    print(f"Successful: {successful}")
    print(f"Failed: {len(chain.execution_log) - successful}")

    # Show final output
    print("\nFinal Email Output:")
    print(results.get("format_email", "No output generated"))


# ===== Solution 5: State Management =====

def solution_5_state_management():
    """
    Solution 5: Complete state management implementation.
    """
    client = LLMClient("openai")

    print("Solution 5: State Management")
    print("=" * 50)

    conversation_flow = [
        "My name is Alice and I'm interested in your Pro plan",
        "What's included in that plan?",
        "Can I get a discount?",
        "Ok, I'll take it. How do I sign up?"
    ]

    print("Conversation flow:\n")
    for msg in conversation_flow:
        print(f"  User: {msg}")
    print()

    class ChainState:
        def __init__(self):
            self.context = {}
            self.history = []
            self.extracted_info = {}
            self.state_snapshots = []

        def update(self, key, value):
            """Update state with new information."""
            self.context[key] = value
            self.state_snapshots.append({
                "timestamp": datetime.now().isoformat(),
                "action": "update",
                "key": key,
                "value": value
            })

        def add_to_history(self, role, message):
            """Add message to conversation history."""
            self.history.append({"role": role, "message": message})

        def extract_and_store(self, text, info_type):
            """Extract and store specific information."""
            if info_type not in self.extracted_info:
                self.extracted_info[info_type] = []
            self.extracted_info[info_type].append(text)

        def get_context_prompt(self):
            """Generate context prompt from current state."""
            context_parts = []

            if self.extracted_info.get("user_name"):
                context_parts.append(f"User Name: {self.extracted_info['user_name'][-1]}")

            if self.extracted_info.get("interest"):
                context_parts.append(f"Interested In: {', '.join(self.extracted_info['interest'])}")

            if self.history:
                recent_history = self.history[-3:]  # Last 3 messages
                history_text = "\n".join([f"{h['role']}: {h['message'][:50]}..." for h in recent_history])
                context_parts.append(f"Recent Conversation:\n{history_text}")

            return "\n".join(context_parts) if context_parts else "No context available"

        def rollback(self, steps=1):
            """Rollback to previous state."""
            if len(self.state_snapshots) >= steps:
                for _ in range(steps):
                    self.state_snapshots.pop()
                # Rebuild context from snapshots
                self.context = {}
                for snapshot in self.state_snapshots:
                    if snapshot["action"] == "update":
                        self.context[snapshot["key"]] = snapshot["value"]
                return True
            return False

    # Initialize state manager
    state = ChainState()

    # Process conversation with state management
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n{'='*40}")
        print(f"Turn {i}: User: {message}")
        print("-" * 40)

        # Add to history
        state.add_to_history("user", message)

        # Extract information from user message
        extract_prompt = f"""Extract key information from this message:

        Message: {message}
        Context: {state.get_context_prompt()}

        Extract:
        - Name (if mentioned):
        - Product Interest:
        - Intent:
        - Questions:"""

        print("Extracting information...")
        extraction = client.complete(extract_prompt, temperature=0.2, max_tokens=100).strip()
        print(f"Extracted: {extraction[:100]}...")

        # Update state based on extraction
        if "Alice" in message:
            state.extract_and_store("Alice", "user_name")
            state.update("current_user", "Alice")

        if "Pro plan" in message:
            state.extract_and_store("Pro plan", "interest")
            state.update("plan_interest", "Pro")

        if "discount" in message.lower():
            state.update("requesting_discount", True)

        if "sign up" in message.lower():
            state.update("ready_to_purchase", True)

        # Generate context-aware response
        response_prompt = f"""Generate a helpful response:

        User Message: {message}
        Context: {state.get_context_prompt()}
        Extracted Info: {extraction}

        Response (personalized and context-aware):"""

        response = client.complete(response_prompt, temperature=0.3, max_tokens=150).strip()
        print(f"\nAssistant: {response}")

        # Add response to history
        state.add_to_history("assistant", response)

    # Display final state summary
    print("\n" + "=" * 40)
    print("FINAL STATE SUMMARY:")
    print(f"User Name: {state.extracted_info.get('user_name', ['Unknown'])[0]}")
    print(f"Interests: {state.extracted_info.get('interest', [])}")
    print(f"Context Keys: {list(state.context.keys())}")
    print(f"Conversation Turns: {len(state.history) // 2}")
    print(f"State Snapshots: {len(state.state_snapshots)}")


# ===== Challenge Solution: Adaptive Workflow Orchestrator =====

def challenge_solution_adaptive_orchestrator():
    """
    Challenge Solution: Complete adaptive workflow orchestrator.
    """
    client = LLMClient("openai")

    print("Challenge Solution: Adaptive Workflow Orchestrator")
    print("=" * 50)

    class AdaptiveOrchestrator:
        def __init__(self, client):
            self.client = client
            self.cache = {}
            self.cache_ttl = 300  # 5 minutes
            self.metrics = {
                "total_calls": 0,
                "cache_hits": 0,
                "total_latency": 0,
                "errors": 0,
                "strategy_usage": {"sequential": 0, "parallel": 0, "hybrid": 0}
            }

        def analyze_task_complexity(self, task):
            """Analyze task to determine complexity level."""
            complexity_prompt = f"""Analyze the complexity of this task:

            Task: {task}

            Consider:
            - Number of steps required
            - Dependencies between steps
            - Data volume
            - Time sensitivity

            Complexity (simple/moderate/complex):"""

            result = self.client.complete(complexity_prompt, temperature=0.2, max_tokens=50).strip()

            if "complex" in result.lower():
                return "complex"
            elif "moderate" in result.lower():
                return "moderate"
            else:
                return "simple"

        def select_strategy(self, complexity, requirements):
            """Select execution strategy based on complexity and requirements."""
            time_limit = requirements.get("time_limit", float('inf'))
            quality_target = requirements.get("quality_target", "medium")

            if complexity == "simple":
                strategy = "sequential"
            elif complexity == "moderate":
                if time_limit < 5:
                    strategy = "parallel"
                else:
                    strategy = "sequential"
            else:  # complex
                strategy = "hybrid"

            self.metrics["strategy_usage"][strategy] += 1
            return strategy

        def execute_sequential(self, steps):
            """Execute steps sequentially."""
            results = []
            for step in steps:
                start = time.time()
                result = self._execute_step(step)
                latency = time.time() - start
                self.metrics["total_latency"] += latency
                results.append(result)
            return results

        def execute_parallel(self, tasks):
            """Execute tasks in parallel."""
            results = []
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [executor.submit(self._execute_step, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
            return results

        def execute_hybrid(self, workflow):
            """Execute hybrid workflow with parallel and sequential parts."""
            results = {}

            # Phase 1: Parallel initial processing
            if "parallel_phase" in workflow:
                print("  Executing parallel phase...")
                parallel_results = self.execute_parallel(workflow["parallel_phase"])
                results["parallel"] = parallel_results

            # Phase 2: Sequential processing
            if "sequential_phase" in workflow:
                print("  Executing sequential phase...")
                sequential_results = self.execute_sequential(workflow["sequential_phase"])
                results["sequential"] = sequential_results

            return results

        def _execute_step(self, step):
            """Execute a single step with caching."""
            # Generate cache key
            cache_key = hashlib.md5(step.encode()).hexdigest()

            # Check cache
            cached = self.get_cached_result(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached

            # Execute step
            self.metrics["total_calls"] += 1
            try:
                result = self.client.complete(step, temperature=0.3, max_tokens=150).strip()
                self.cache_result(cache_key, result)
                return result
            except Exception as e:
                self.metrics["errors"] += 1
                return f"Error: {str(e)}"

        def cache_result(self, key, result):
            """Cache result with TTL."""
            self.cache[key] = {
                "result": result,
                "timestamp": time.time()
            }

        def get_cached_result(self, key):
            """Retrieve cached result if valid."""
            if key in self.cache:
                cached = self.cache[key]
                if time.time() - cached["timestamp"] < self.cache_ttl:
                    return cached["result"]
                else:
                    del self.cache[key]  # Remove expired entry
            return None

        def optimize_execution(self, workflow, constraints):
            """Optimize workflow execution based on constraints."""
            complexity = self.analyze_task_complexity(workflow)
            strategy = self.select_strategy(complexity, constraints)

            print(f"  Complexity: {complexity}")
            print(f"  Selected Strategy: {strategy}")

            if strategy == "sequential":
                steps = [f"Execute: {workflow}"]
                return self.execute_sequential(steps)
            elif strategy == "parallel":
                tasks = [f"Parallel task for: {workflow}"] * 3
                return self.execute_parallel(tasks)
            else:  # hybrid
                hybrid_workflow = {
                    "parallel_phase": [f"Analyze: {workflow}", f"Research: {workflow}"],
                    "sequential_phase": [f"Synthesize results for: {workflow}"]
                }
                return self.execute_hybrid(hybrid_workflow)

        def generate_analytics(self):
            """Generate execution analytics report."""
            total_calls = self.metrics["total_calls"]
            if total_calls == 0:
                return "No executions yet"

            cache_hit_rate = (self.metrics["cache_hits"] / total_calls * 100) if total_calls > 0 else 0
            avg_latency = self.metrics["total_latency"] / total_calls if total_calls > 0 else 0
            error_rate = (self.metrics["errors"] / total_calls * 100) if total_calls > 0 else 0

            return f"""
Execution Analytics:
  Total API Calls: {total_calls}
  Cache Hit Rate: {cache_hit_rate:.1f}%
  Average Latency: {avg_latency:.2f}s
  Error Rate: {error_rate:.1f}%
  Strategy Usage: {self.metrics['strategy_usage']}
"""

    # Initialize orchestrator
    orchestrator = AdaptiveOrchestrator(client)

    # Test scenarios
    test_scenarios = [
        {
            "task": "Analyze customer feedback and generate report",
            "constraints": {"time_limit": 5, "quality_target": "high"}
        },
        {
            "task": "Process 100 documents for classification",
            "constraints": {"cost_limit": 1.0, "time_limit": 60}
        },
        {
            "task": "Real-time chat response with context",
            "constraints": {"time_limit": 2, "quality_target": "medium"}
        }
    ]

    print("Processing Test Scenarios:\n")

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: {scenario['task']}")
        print(f"Constraints: {scenario['constraints']}")

        # Execute with optimization
        start = time.time()
        result = orchestrator.optimize_execution(scenario["task"], scenario["constraints"])
        elapsed = time.time() - start

        print(f"  Execution Time: {elapsed:.2f}s")
        print(f"  Result: {str(result)[:100]}...")

    # Demonstrate caching benefits
    print("\n" + "=" * 40)
    print("CACHE DEMONSTRATION:")

    # Execute same task twice
    test_task = "Analyze customer sentiment"
    print(f"Task: {test_task}")

    print("First execution...")
    start = time.time()
    result1 = orchestrator.optimize_execution(test_task, {"time_limit": 5})
    time1 = time.time() - start

    print("Second execution (should use cache)...")
    start = time.time()
    result2 = orchestrator.optimize_execution(test_task, {"time_limit": 5})
    time2 = time.time() - start

    print(f"\nFirst execution time: {time1:.2f}s")
    print(f"Second execution time: {time2:.2f}s")
    print(f"Speed improvement: {(time1/time2 - 1) * 100:.1f}% faster")

    # Generate analytics
    print("\n" + "=" * 40)
    print(orchestrator.generate_analytics())


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 05: Prompt Chaining Solutions")
    parser.add_argument("--solution", type=int, help="Run specific solution (1-5)")
    parser.add_argument("--challenge", action="store_true", help="Run challenge solution")
    parser.add_argument("--all", action="store_true", help="Run all solutions")

    args = parser.parse_args()

    solutions = {
        1: solution_1_sequential_pipeline,
        2: solution_2_conditional_routing,
        3: solution_3_parallel_processing,
        4: solution_4_error_recovery,
        5: solution_5_state_management
    }

    if args.all:
        for sol_num, sol_func in solutions.items():
            print(f"\n{'='*60}")
            sol_func()
            print(f"\n{'='*60}\n")
        challenge_solution_adaptive_orchestrator()
    elif args.challenge:
        challenge_solution_adaptive_orchestrator()
    elif args.solution and args.solution in solutions:
        solutions[args.solution]()
    else:
        print("Module 05: Prompt Chaining - Solutions")
        print("\nUsage:")
        print("  python solutions.py --solution N  # Run solution N")
        print("  python solutions.py --challenge    # Run challenge solution")
        print("  python solutions.py --all          # Run all solutions")
        print("\nAvailable solutions:")
        print("  1: Sequential Pipeline")
        print("  2: Conditional Routing")
        print("  3: Parallel Processing")
        print("  4: Error Recovery")
        print("  5: State Management")
        print("  Challenge: Adaptive Workflow Orchestrator")