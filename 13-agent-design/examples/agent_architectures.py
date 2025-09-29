"""
Module 13: Agent Design
Agent Architectures Examples

This file demonstrates various autonomous agent architectures:
1. ReAct (Reasoning and Acting) agent
2. Tree of Thoughts agent
3. Self-reflective agent
4. Goal decomposition agent
5. Tool-using agent
6. Multi-step reasoning agent
7. Production agent with monitoring

Each example shows progressively more sophisticated agent patterns.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Example 1: ReAct (Reasoning and Acting) Agent
print("=" * 50)
print("Example 1: ReAct Agent")
print("=" * 50)


class ReActAgent:
    """Agent that interleaves reasoning and acting."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.model = model
        self.max_steps = 10
        self.history = []
        self.tools = {
            "calculate": self.calculate,
            "search": self.search,
            "compare": self.compare
        }

    def solve(self, problem: str) -> str:
        """Solve a problem using ReAct pattern."""
        print(f"Problem: {problem}\n")

        for step in range(self.max_steps):
            # Generate thought
            thought = self.think(problem)
            print(f"Thought {step + 1}: {thought}")

            # Decide on action
            action = self.decide_action(thought)
            print(f"Action {step + 1}: {action}")

            # Execute action
            observation = self.execute_action(action)
            print(f"Observation {step + 1}: {observation}\n")

            # Update history
            self.history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

            # Check if problem is solved
            if self.is_solved(problem, observation):
                return self.format_solution(observation)

        return "Could not solve the problem within the step limit."

    def think(self, problem: str) -> str:
        """Generate reasoning about the current state."""
        context = self.get_context()

        prompt = f"""Problem: {problem}

Previous steps:
{context}

What should I think about next? Reason step by step.

Thought:"""

        # Simulate LLM response
        thoughts = [
            "I need to break down this problem into smaller parts.",
            "Let me calculate the intermediate values first.",
            "Now I should verify my calculation.",
            "I have all the information needed to solve this."
        ]

        return thoughts[min(len(self.history), len(thoughts) - 1)]

    def decide_action(self, thought: str) -> Dict:
        """Decide what action to take based on thought."""
        # Simulate action decision
        actions = [
            {"tool": "calculate", "input": "2 + 2"},
            {"tool": "search", "input": "definition of AI"},
            {"tool": "compare", "input": ["option1", "option2"]}
        ]

        return actions[len(self.history) % len(actions)]

    def execute_action(self, action: Dict) -> str:
        """Execute the selected action."""
        tool = action.get("tool")
        if tool in self.tools:
            return self.tools[tool](action.get("input"))
        return "Tool not found"

    def calculate(self, expression: str) -> str:
        """Calculation tool."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Calculation error"

    def search(self, query: str) -> str:
        """Search tool (simulated)."""
        return f"Found information about: {query}"

    def compare(self, items: List) -> str:
        """Comparison tool."""
        return f"Compared {len(items)} items"

    def is_solved(self, problem: str, observation: str) -> bool:
        """Check if the problem is solved."""
        return len(self.history) >= 3  # Simplified check

    def get_context(self) -> str:
        """Get formatted history context."""
        if not self.history:
            return "No previous steps"

        context = []
        for i, step in enumerate(self.history[-3:]):  # Last 3 steps
            context.append(f"Step {i+1}:")
            context.append(f"  Thought: {step['thought']}")
            context.append(f"  Action: {step['action']}")
            context.append(f"  Result: {step['observation']}")

        return "\n".join(context)

    def format_solution(self, observation: str) -> str:
        """Format the final solution."""
        return f"Solution: Based on my reasoning and actions, {observation}"


# Example usage
react_agent = ReActAgent()
solution = react_agent.solve("What is the sum of 2+2 and how does it relate to AI?")
print(f"Final Solution: {solution}")


# Example 2: Tree of Thoughts Agent
print("\n" + "=" * 50)
print("Example 2: Tree of Thoughts Agent")
print("=" * 50)


@dataclass
class ThoughtNode:
    """Node in the thought tree."""
    content: str
    score: float = 0.0
    children: List['ThoughtNode'] = field(default_factory=list)
    parent: Optional['ThoughtNode'] = None
    depth: int = 0


class TreeOfThoughtsAgent:
    """Agent that explores multiple reasoning paths."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.model = model
        self.max_depth = 3
        self.branches_per_node = 3

    def solve(self, problem: str) -> str:
        """Solve using tree of thoughts."""
        print(f"Problem: {problem}\n")

        # Create root node
        root = ThoughtNode(content=problem)

        # Build thought tree
        self.expand_tree(root)

        # Find best path
        best_path = self.find_best_path(root)

        # Execute best path
        return self.execute_path(best_path)

    def expand_tree(self, node: ThoughtNode, depth: int = 0):
        """Recursively expand the thought tree."""
        if depth >= self.max_depth:
            return

        # Generate multiple thoughts
        thoughts = self.generate_thoughts(node.content)

        for thought_content in thoughts:
            # Create child node
            child = ThoughtNode(
                content=thought_content,
                parent=node,
                depth=depth + 1
            )

            # Score the thought
            child.score = self.evaluate_thought(thought_content)

            # Add to tree
            node.children.append(child)

            # Recursive expansion
            self.expand_tree(child, depth + 1)

        print(f"Expanded node at depth {depth}: {len(node.children)} children")

    def generate_thoughts(self, context: str) -> List[str]:
        """Generate multiple thought continuations."""
        # Simulate generating diverse thoughts
        thoughts = []

        for i in range(self.branches_per_node):
            thought = f"Thought branch {i+1} from: {context[:30]}..."
            thoughts.append(thought)

        return thoughts

    def evaluate_thought(self, thought: str) -> float:
        """Evaluate the quality/promise of a thought."""
        # Simulate evaluation
        # In practice, would use LLM to score
        score = np.random.random()

        # Bonus for certain keywords (simplified)
        if "solution" in thought.lower():
            score += 0.3
        if "therefore" in thought.lower():
            score += 0.2

        return min(score, 1.0)

    def find_best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Find the highest scoring path through the tree."""
        best_path = []
        best_score = -float('inf')

        def dfs(node: ThoughtNode, path: List[ThoughtNode], score: float):
            nonlocal best_path, best_score

            path.append(node)
            score += node.score

            if not node.children:  # Leaf node
                if score > best_score:
                    best_score = score
                    best_path = path.copy()
            else:
                for child in node.children:
                    dfs(child, path, score)

            path.pop()

        dfs(root, [], 0)

        print(f"Best path score: {best_score:.2f}")
        print(f"Path length: {len(best_path)}")

        return best_path

    def execute_path(self, path: List[ThoughtNode]) -> str:
        """Execute the selected reasoning path."""
        reasoning_chain = []

        for i, node in enumerate(path):
            reasoning_chain.append(f"Step {i}: {node.content}")

        return "\n".join(reasoning_chain)

    def visualize_tree(self, root: ThoughtNode, indent: int = 0):
        """Visualize the thought tree."""
        print("  " * indent + f"[{root.score:.2f}] {root.content[:50]}...")

        for child in root.children:
            self.visualize_tree(child, indent + 1)


# Example usage
tot_agent = TreeOfThoughtsAgent()
tot_solution = tot_agent.solve("How can we improve software development efficiency?")
print(f"\nSolution path:\n{tot_solution}")


# Example 3: Self-Reflective Agent
print("\n" + "=" * 50)
print("Example 3: Self-Reflective Agent")
print("=" * 50)


class SelfReflectiveAgent:
    """Agent that learns from self-reflection and critique."""

    def __init__(self):
        self.attempt_history = []
        self.learned_strategies = {}
        self.improvement_log = []

    def solve_with_reflection(self, task: str, max_attempts: int = 3) -> Dict:
        """Solve task with self-reflection and improvement."""
        print(f"Task: {task}\n")

        best_result = None
        best_score = 0

        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}:")

            # Generate solution
            solution = self.generate_solution(task, attempt)
            print(f"  Solution: {solution}")

            # Self-critique
            critique = self.critique_solution(task, solution)
            print(f"  Critique: {critique}")

            # Score solution
            score = critique.get("score", 0)
            print(f"  Score: {score}/10")

            # Store attempt
            self.attempt_history.append({
                "task": task,
                "attempt": attempt + 1,
                "solution": solution,
                "critique": critique,
                "score": score
            })

            # Update best
            if score > best_score:
                best_score = score
                best_result = solution

            # Check if good enough
            if score >= 8:
                print(f"  ✓ Solution acceptable!")
                break

            # Reflect and learn
            if attempt < max_attempts - 1:
                improvements = self.reflect_and_improve(task, solution, critique)
                print(f"  Improvements for next attempt: {improvements}")
                self.learn_from_reflection(task, improvements)

        return {
            "task": task,
            "best_solution": best_result,
            "best_score": best_score,
            "attempts": len(self.attempt_history)
        }

    def generate_solution(self, task: str, attempt: int) -> str:
        """Generate a solution for the task."""
        # Apply learned strategies
        strategies = self.learned_strategies.get(task, [])

        base_solution = f"Solution for '{task}'"

        if strategies:
            base_solution += f" using strategies: {', '.join(strategies)}"

        # Simulate improvement over attempts
        if attempt > 0:
            base_solution += f" (improved version {attempt + 1})"

        return base_solution

    def critique_solution(self, task: str, solution: str) -> Dict:
        """Critique the generated solution."""
        # Simulate critique generation
        critique_aspects = {
            "completeness": np.random.randint(5, 10),
            "accuracy": np.random.randint(5, 10),
            "clarity": np.random.randint(5, 10),
            "efficiency": np.random.randint(5, 10)
        }

        # Overall score
        score = np.mean(list(critique_aspects.values()))

        # Generate feedback
        feedback = []
        for aspect, value in critique_aspects.items():
            if value < 7:
                feedback.append(f"Could improve {aspect}")

        return {
            "aspects": critique_aspects,
            "score": score,
            "feedback": feedback
        }

    def reflect_and_improve(self, task: str, solution: str, critique: Dict) -> List[str]:
        """Reflect on the critique and identify improvements."""
        improvements = []

        # Analyze weak aspects
        for aspect, score in critique["aspects"].items():
            if score < 7:
                improvement = f"Focus on improving {aspect}"
                improvements.append(improvement)

        # Add general improvements
        if critique["score"] < 6:
            improvements.append("Reconsider approach entirely")
        elif critique["score"] < 8:
            improvements.append("Refine existing approach")

        self.improvement_log.append({
            "task": task,
            "identified_improvements": improvements,
            "timestamp": datetime.now()
        })

        return improvements

    def learn_from_reflection(self, task: str, improvements: List[str]):
        """Store learned strategies for future use."""
        if task not in self.learned_strategies:
            self.learned_strategies[task] = []

        # Add unique improvements as strategies
        for improvement in improvements:
            if improvement not in self.learned_strategies[task]:
                self.learned_strategies[task].append(improvement)

    def get_learning_summary(self) -> Dict:
        """Summarize what the agent has learned."""
        return {
            "total_attempts": len(self.attempt_history),
            "tasks_attempted": len(self.learned_strategies),
            "strategies_learned": sum(len(s) for s in self.learned_strategies.values()),
            "average_improvement": self.calculate_average_improvement()
        }

    def calculate_average_improvement(self) -> float:
        """Calculate average score improvement across attempts."""
        if len(self.attempt_history) < 2:
            return 0

        improvements = []
        current_task = None
        first_score = None

        for attempt in self.attempt_history:
            if attempt["task"] != current_task:
                current_task = attempt["task"]
                first_score = attempt["score"]
            else:
                improvement = attempt["score"] - first_score
                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0


# Example usage
reflective_agent = SelfReflectiveAgent()
result = reflective_agent.solve_with_reflection("Create an efficient sorting algorithm")
print(f"\nFinal Result: {result}")
print(f"Learning Summary: {reflective_agent.get_learning_summary()}")


# Example 4: Goal Decomposition Agent
print("\n" + "=" * 50)
print("Example 4: Goal Decomposition Agent")
print("=" * 50)


@dataclass
class Goal:
    """Represents a goal or subgoal."""
    description: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None


class GoalDecompositionAgent:
    """Agent that decomposes complex goals into manageable subgoals."""

    def __init__(self):
        self.goals = {}
        self.execution_order = []
        self.completed_goals = []

    def achieve_goal(self, main_goal: str) -> Dict:
        """Achieve a complex goal through decomposition."""
        print(f"Main Goal: {main_goal}\n")

        # Decompose into subgoals
        subgoals = self.decompose_goal(main_goal)
        print(f"Decomposed into {len(subgoals)} subgoals")

        # Create goal objects
        for subgoal in subgoals:
            goal = Goal(
                description=subgoal["description"],
                priority=subgoal["priority"],
                dependencies=subgoal.get("dependencies", [])
            )
            self.goals[subgoal["id"]] = goal

        # Determine execution order
        self.execution_order = self.topological_sort()
        print(f"Execution order: {self.execution_order}\n")

        # Execute goals
        results = self.execute_goals()

        return {
            "main_goal": main_goal,
            "subgoals_completed": len(self.completed_goals),
            "total_subgoals": len(self.goals),
            "success": len(self.completed_goals) == len(self.goals),
            "results": results
        }

    def decompose_goal(self, goal: str) -> List[Dict]:
        """Decompose a high-level goal into subgoals."""
        # Simulate goal decomposition
        # In practice, would use LLM for intelligent decomposition

        if "build" in goal.lower():
            subgoals = [
                {"id": "research", "description": "Research requirements", "priority": 1, "dependencies": []},
                {"id": "design", "description": "Create design", "priority": 2, "dependencies": ["research"]},
                {"id": "implement", "description": "Implement solution", "priority": 3, "dependencies": ["design"]},
                {"id": "test", "description": "Test thoroughly", "priority": 4, "dependencies": ["implement"]},
                {"id": "deploy", "description": "Deploy to production", "priority": 5, "dependencies": ["test"]}
            ]
        else:
            subgoals = [
                {"id": "analyze", "description": "Analyze the problem", "priority": 1, "dependencies": []},
                {"id": "plan", "description": "Create action plan", "priority": 2, "dependencies": ["analyze"]},
                {"id": "execute", "description": "Execute the plan", "priority": 3, "dependencies": ["plan"]}
            ]

        return subgoals

    def topological_sort(self) -> List[str]:
        """Determine execution order respecting dependencies."""
        visited = set()
        stack = []

        def visit(goal_id: str):
            if goal_id in visited:
                return

            visited.add(goal_id)

            goal = self.goals.get(goal_id)
            if goal:
                for dep in goal.dependencies:
                    if dep in self.goals:
                        visit(dep)

            stack.append(goal_id)

        for goal_id in self.goals:
            visit(goal_id)

        return stack

    def execute_goals(self) -> List[Dict]:
        """Execute goals in the determined order."""
        results = []

        for goal_id in self.execution_order:
            goal = self.goals[goal_id]

            print(f"Executing: {goal.description}")

            # Check dependencies
            deps_met = all(
                self.goals[dep].status == "completed"
                for dep in goal.dependencies
                if dep in self.goals
            )

            if not deps_met:
                print(f"  ✗ Dependencies not met")
                goal.status = "failed"
                results.append({
                    "goal": goal_id,
                    "status": "failed",
                    "reason": "dependencies"
                })
                continue

            # Execute goal
            goal.status = "in_progress"
            success, result = self.execute_single_goal(goal)

            if success:
                goal.status = "completed"
                goal.result = result
                self.completed_goals.append(goal_id)
                print(f"  ✓ Completed")
            else:
                goal.status = "failed"
                print(f"  ✗ Failed")

            results.append({
                "goal": goal_id,
                "status": goal.status,
                "result": result
            })

        return results

    def execute_single_goal(self, goal: Goal) -> Tuple[bool, Any]:
        """Execute a single goal."""
        # Simulate goal execution
        # In practice, would map to actual actions

        # Simulate success with 90% probability
        success = np.random.random() > 0.1

        result = f"Result of {goal.description}" if success else "Execution failed"

        return success, result


# Example usage
goal_agent = GoalDecompositionAgent()
achievement = goal_agent.achieve_goal("Build a machine learning model for classification")
print(f"\nGoal Achievement Summary: {achievement}")


# Example 5: Tool-Using Agent
print("\n" + "=" * 50)
print("Example 5: Tool-Using Agent")
print("=" * 50)


class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0

    def execute(self, *args, **kwargs):
        """Execute the tool."""
        self.usage_count += 1
        return self._execute(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        """Actual execution logic (to be overridden)."""
        raise NotImplementedError


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Perform mathematical calculations")

    def _execute(self, expression: str):
        try:
            return eval(expression)
        except:
            return "Error in calculation"


class WebSearchTool(Tool):
    def __init__(self):
        super().__init__("web_search", "Search the web for information")

    def _execute(self, query: str):
        # Simulate web search
        return f"Search results for: {query}"


class DatabaseTool(Tool):
    def __init__(self):
        super().__init__("database", "Query and update database")

    def _execute(self, query: str):
        # Simulate database operation
        return f"Database result: {query}"


class ToolUsingAgent:
    """Agent that can select and use appropriate tools."""

    def __init__(self):
        self.tools = {}
        self.tool_history = []
        self.register_default_tools()

    def register_default_tools(self):
        """Register default tools."""
        self.register_tool(CalculatorTool())
        self.register_tool(WebSearchTool())
        self.register_tool(DatabaseTool())

    def register_tool(self, tool: Tool):
        """Register a tool for use."""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")

    def solve_task(self, task: str) -> Dict:
        """Solve a task using available tools."""
        print(f"Task: {task}\n")

        # Analyze task and plan tool usage
        plan = self.plan_tool_usage(task)
        print(f"Tool usage plan: {plan}\n")

        results = []

        for step in plan:
            tool_name = step["tool"]
            tool_input = step["input"]

            if tool_name in self.tools:
                print(f"Using {tool_name}: {tool_input}")
                result = self.tools[tool_name].execute(tool_input)
                print(f"  Result: {result}")

                self.tool_history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": result,
                    "timestamp": datetime.now()
                })

                results.append(result)
            else:
                print(f"Tool {tool_name} not found")
                results.append(None)

        # Synthesize final answer
        final_answer = self.synthesize_answer(task, results)

        return {
            "task": task,
            "tools_used": [step["tool"] for step in plan],
            "intermediate_results": results,
            "final_answer": final_answer
        }

    def plan_tool_usage(self, task: str) -> List[Dict]:
        """Plan which tools to use for the task."""
        # Simulate intelligent tool selection
        # In practice, would use LLM for planning

        plans = {
            "calculate": [
                {"tool": "calculator", "input": "15 * 23"},
                {"tool": "calculator", "input": "345 / 15"}
            ],
            "research": [
                {"tool": "web_search", "input": "latest AI developments"},
                {"tool": "database", "input": "SELECT * FROM research_papers"}
            ],
            "default": [
                {"tool": "web_search", "input": task}
            ]
        }

        # Select plan based on task keywords
        for keyword, plan in plans.items():
            if keyword in task.lower():
                return plan

        return plans["default"]

    def synthesize_answer(self, task: str, results: List) -> str:
        """Synthesize final answer from tool results."""
        # Simulate answer synthesis
        # In practice, would use LLM to combine results

        if not results:
            return "No results obtained"

        non_null_results = [r for r in results if r is not None]

        if non_null_results:
            return f"Based on tool results: {', '.join(map(str, non_null_results))}"

        return "Tools execution completed but no valid results"

    def get_tool_statistics(self) -> Dict:
        """Get statistics about tool usage."""
        stats = {
            "total_tools": len(self.tools),
            "total_uses": len(self.tool_history),
            "tool_usage": {}
        }

        for tool_name, tool in self.tools.items():
            stats["tool_usage"][tool_name] = tool.usage_count

        return stats


# Example usage
tool_agent = ToolUsingAgent()
result = tool_agent.solve_task("Calculate 15 * 23 and then search for applications of the result")
print(f"\nTask Result: {result}")
print(f"Tool Statistics: {tool_agent.get_tool_statistics()}")


# Example 6: Multi-Step Reasoning Agent
print("\n" + "=" * 50)
print("Example 6: Multi-Step Reasoning Agent")
print("=" * 50)


class ReasoningStep:
    """Represents a single reasoning step."""

    def __init__(self, step_type: str, content: str, confidence: float = 1.0):
        self.step_type = step_type  # hypothesis, deduction, induction, etc.
        self.content = content
        self.confidence = confidence
        self.supporting_evidence = []
        self.contradicting_evidence = []


class MultiStepReasoningAgent:
    """Agent that performs complex multi-step reasoning."""

    def __init__(self):
        self.reasoning_chain = []
        self.hypotheses = []
        self.conclusions = []

    def reason_about(self, question: str, context: Dict = None) -> Dict:
        """Perform multi-step reasoning about a question."""
        print(f"Question: {question}\n")

        # Initialize reasoning
        self.reasoning_chain = []

        # Step 1: Understand the question
        understanding = self.understand_question(question)
        self.add_reasoning_step("understanding", understanding)

        # Step 2: Generate hypotheses
        hypotheses = self.generate_hypotheses(question, context)
        for hyp in hypotheses:
            self.add_reasoning_step("hypothesis", hyp)
            self.hypotheses.append(hyp)

        # Step 3: Evaluate each hypothesis
        evaluations = self.evaluate_hypotheses(hypotheses, context)
        for eval in evaluations:
            self.add_reasoning_step("evaluation", eval["reasoning"], eval["confidence"])

        # Step 4: Draw conclusions
        conclusions = self.draw_conclusions(evaluations)
        for conclusion in conclusions:
            self.add_reasoning_step("conclusion", conclusion)
            self.conclusions.append(conclusion)

        # Step 5: Synthesize final answer
        final_answer = self.synthesize_answer(conclusions)

        return {
            "question": question,
            "reasoning_steps": len(self.reasoning_chain),
            "hypotheses": self.hypotheses,
            "conclusions": self.conclusions,
            "answer": final_answer,
            "confidence": self.calculate_overall_confidence()
        }

    def understand_question(self, question: str) -> str:
        """Understand what the question is asking."""
        # Simulate question understanding
        understanding = f"The question asks about: {question[:50]}..."

        # Identify question type
        if "how" in question.lower():
            understanding += " (Process/Method question)"
        elif "why" in question.lower():
            understanding += " (Causal/Reasoning question)"
        elif "what" in question.lower():
            understanding += " (Definition/Description question)"

        return understanding

    def generate_hypotheses(self, question: str, context: Dict = None) -> List[str]:
        """Generate possible hypotheses."""
        # Simulate hypothesis generation
        hypotheses = [
            f"Hypothesis 1: The answer might involve {question.split()[0]}",
            f"Hypothesis 2: This could be related to {question.split()[-1]}",
            f"Hypothesis 3: Consider alternative interpretation"
        ]

        if context:
            hypotheses.append(f"Hypothesis 4: Context suggests {list(context.keys())[0]}")

        return hypotheses

    def evaluate_hypotheses(self, hypotheses: List[str], context: Dict = None) -> List[Dict]:
        """Evaluate each hypothesis."""
        evaluations = []

        for i, hypothesis in enumerate(hypotheses):
            # Simulate evaluation
            confidence = np.random.random()

            evaluation = {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "reasoning": f"Evaluation of {hypothesis}: confidence {confidence:.2f}",
                "supported": confidence > 0.5
            }

            evaluations.append(evaluation)

        return evaluations

    def draw_conclusions(self, evaluations: List[Dict]) -> List[str]:
        """Draw conclusions from evaluations."""
        conclusions = []

        # Find supported hypotheses
        supported = [e for e in evaluations if e["supported"]]

        if supported:
            # Primary conclusion from best hypothesis
            best = max(supported, key=lambda x: x["confidence"])
            conclusions.append(f"Primary conclusion based on: {best['hypothesis']}")

            # Secondary conclusions
            for eval in supported:
                if eval != best:
                    conclusions.append(f"Alternative: {eval['hypothesis']}")
        else:
            conclusions.append("No strong conclusions can be drawn")

        return conclusions

    def synthesize_answer(self, conclusions: List[str]) -> str:
        """Synthesize final answer from conclusions."""
        if not conclusions:
            return "Unable to determine answer"

        # Combine conclusions into coherent answer
        answer = "Based on multi-step reasoning: "
        answer += conclusions[0]

        if len(conclusions) > 1:
            answer += f" (with {len(conclusions) - 1} alternative considerations)"

        return answer

    def add_reasoning_step(self, step_type: str, content: str, confidence: float = 1.0):
        """Add a step to the reasoning chain."""
        step = ReasoningStep(step_type, content, confidence)
        self.reasoning_chain.append(step)

        print(f"Step {len(self.reasoning_chain)} ({step_type}): {content[:80]}...")
        if confidence < 1.0:
            print(f"  Confidence: {confidence:.2f}")

    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in the reasoning."""
        if not self.reasoning_chain:
            return 0

        confidences = [step.confidence for step in self.reasoning_chain]
        return np.mean(confidences)

    def visualize_reasoning(self):
        """Visualize the reasoning chain."""
        print("\nReasoning Chain Visualization:")
        print("-" * 40)

        for i, step in enumerate(self.reasoning_chain):
            indent = "  " * (step.step_type == "evaluation")
            confidence_bar = "█" * int(step.confidence * 10)
            print(f"{indent}[{i+1}] {step.step_type.upper()}: {confidence_bar}")
            print(f"{indent}    {step.content[:60]}...")


# Example usage
reasoning_agent = MultiStepReasoningAgent()
result = reasoning_agent.reason_about(
    "How can artificial intelligence improve healthcare?",
    context={"domain": "healthcare", "focus": "diagnosis"}
)
print(f"\nReasoning Result: {result}")
reasoning_agent.visualize_reasoning()


# Example 7: Production Agent with Monitoring
print("\n" + "=" * 50)
print("Example 7: Production Agent with Monitoring")
print("=" * 50)


class AgentState(Enum):
    """Agent states."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Metrics for agent monitoring."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_thinking_time: float = 0
    total_acting_time: float = 0
    errors_encountered: int = 0
    average_task_time: float = 0
    success_rate: float = 0


class ProductionAgent:
    """Production-ready agent with monitoring and safety features."""

    def __init__(self, agent_id: str, max_concurrent_tasks: int = 5):
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = []
        self.task_queue = deque()
        self.metrics = AgentMetrics()
        self.error_log = []
        self.safety_checks = []
        self.audit_trail = []

    async def process_task(self, task: Dict) -> Dict:
        """Process a task with full monitoring."""
        task_id = task.get("id", "unknown")
        start_time = time.time()

        print(f"Agent {self.agent_id} processing task {task_id}")

        # Audit trail
        self.audit_trail.append({
            "timestamp": datetime.now(),
            "task_id": task_id,
            "action": "task_started"
        })

        try:
            # Safety check
            if not self.pass_safety_checks(task):
                raise ValueError("Task failed safety checks")

            # Think phase
            self.state = AgentState.THINKING
            thinking_start = time.time()
            plan = await self.think(task)
            self.metrics.total_thinking_time += time.time() - thinking_start

            # Act phase
            self.state = AgentState.ACTING
            acting_start = time.time()
            result = await self.act(plan)
            self.metrics.total_acting_time += time.time() - acting_start

            # Success
            self.metrics.tasks_completed += 1
            self.state = AgentState.IDLE

            # Record success
            self.audit_trail.append({
                "timestamp": datetime.now(),
                "task_id": task_id,
                "action": "task_completed",
                "duration": time.time() - start_time
            })

            return {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "duration": time.time() - start_time
            }

        except Exception as e:
            # Handle error
            self.state = AgentState.ERROR
            self.metrics.tasks_failed += 1
            self.metrics.errors_encountered += 1

            error_info = {
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now()
            }

            self.error_log.append(error_info)

            # Record failure
            self.audit_trail.append({
                "timestamp": datetime.now(),
                "task_id": task_id,
                "action": "task_failed",
                "error": str(e)
            })

            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }

        finally:
            # Update metrics
            self.update_metrics()

    async def think(self, task: Dict) -> Dict:
        """Thinking/planning phase."""
        # Simulate thinking
        await asyncio.sleep(0.1)

        plan = {
            "task": task,
            "steps": ["analyze", "prepare", "execute", "verify"],
            "estimated_time": 1.0
        }

        return plan

    async def act(self, plan: Dict) -> Any:
        """Acting/execution phase."""
        # Simulate action execution
        await asyncio.sleep(0.2)

        # Execute plan steps
        results = []
        for step in plan["steps"]:
            result = f"Completed: {step}"
            results.append(result)

        return results

    def pass_safety_checks(self, task: Dict) -> bool:
        """Run safety checks on task."""
        # Check task structure
        if not isinstance(task, dict):
            return False

        if "id" not in task:
            return False

        # Check for malicious content (simplified)
        if "eval" in str(task) or "exec" in str(task):
            return False

        # Run custom safety checks
        for check in self.safety_checks:
            if not check(task):
                return False

        return True

    def add_safety_check(self, check_function):
        """Add a custom safety check."""
        self.safety_checks.append(check_function)

    def update_metrics(self):
        """Update agent metrics."""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed

        if total_tasks > 0:
            self.metrics.success_rate = self.metrics.tasks_completed / total_tasks

            total_time = self.metrics.total_thinking_time + self.metrics.total_acting_time
            self.metrics.average_task_time = total_time / total_tasks

    def get_status(self) -> Dict:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "current_tasks": len(self.current_tasks),
            "queued_tasks": len(self.task_queue),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": f"{self.metrics.success_rate:.2%}",
                "avg_task_time": f"{self.metrics.average_task_time:.2f}s",
                "errors": self.metrics.errors_encountered
            }
        }

    def get_health_check(self) -> Dict:
        """Health check for monitoring."""
        health_status = "healthy"

        # Check error rate
        if self.metrics.success_rate < 0.5:
            health_status = "degraded"

        # Check for recent errors
        recent_errors = [e for e in self.error_log
                        if (datetime.now() - e["timestamp"]).seconds < 300]
        if len(recent_errors) > 5:
            health_status = "unhealthy"

        return {
            "agent_id": self.agent_id,
            "status": health_status,
            "state": self.state.value,
            "recent_errors": len(recent_errors),
            "uptime": self.calculate_uptime()
        }

    def calculate_uptime(self) -> str:
        """Calculate agent uptime."""
        if self.audit_trail:
            first_event = self.audit_trail[0]["timestamp"]
            uptime = datetime.now() - first_event
            return str(uptime)
        return "0:00:00"


# Example usage
async def production_demo():
    agent = ProductionAgent("agent-001")

    # Add custom safety check
    agent.add_safety_check(lambda task: task.get("priority", 0) < 100)

    # Process some tasks
    tasks = [
        {"id": "task-1", "type": "analysis", "priority": 1},
        {"id": "task-2", "type": "synthesis", "priority": 2},
        {"id": "task-3", "type": "invalid"},  # Will fail safety
    ]

    for task in tasks:
        result = await agent.process_task(task)
        print(f"  Result: {result['status']}\n")

    # Get status and health
    print("Agent Status:", agent.get_status())
    print("Health Check:", agent.get_health_check())


# Run production demo
print("Running production agent demo...")
asyncio.run(production_demo())


print("\n" + "=" * 50)
print("All Agent Architecture Examples Complete!")
print("=" * 50)