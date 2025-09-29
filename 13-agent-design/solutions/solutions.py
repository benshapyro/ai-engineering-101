"""
Module 13: Agent Design
Solutions

Complete implementations for all agent design exercises.
"""

import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()


# Solution 1: Build a Basic Autonomous Agent
print("=" * 50)
print("Solution 1: Build a Basic Autonomous Agent")
print("=" * 50)


class BasicAgent:
    """Basic autonomous agent implementation."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.observations = deque(maxlen=100)
        self.actions_taken = deque(maxlen=100)
        self.learning_rate = 0.1

        # Q-learning table for decision making
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = 0.1  # Exploration rate

        # Performance tracking
        self.rewards_earned = []
        self.episode_count = 0

    def perceive(self, environment: Dict) -> Dict:
        """
        Perceive the environment and extract relevant information.

        Args:
            environment: Current state of the environment

        Returns:
            Processed observations
        """
        # Extract features from environment
        observations = {
            "current_position": environment.get("current_position", 0),
            "goal_position": environment.get("goal", 10),
            "obstacles": environment.get("obstacles", []),
            "timestamp": datetime.now()
        }

        # Calculate derived features
        observations["distance_to_goal"] = abs(
            observations["goal_position"] - observations["current_position"]
        )

        # Check for nearby obstacles
        observations["obstacle_ahead"] = (
            observations["current_position"] + 1 in observations["obstacles"]
        )

        # Store observation
        self.observations.append(observations)

        print(f"Perceived: position={observations['current_position']}, "
              f"goal={observations['goal_position']}")

        return observations

    def decide(self, observations: Dict) -> str:
        """
        Make a decision based on observations using Q-learning.

        Args:
            observations: Processed observations from perception

        Returns:
            Selected action
        """
        # Create state representation
        state = self._get_state_key(observations)

        # Get available actions
        available_actions = self._get_available_actions(observations)

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(available_actions)
            print(f"Exploring: {action}")
        else:
            # Exploit: best known action
            q_values = {a: self.q_table[state][a] for a in available_actions}

            if not q_values or all(v == 0 for v in q_values.values()):
                action = np.random.choice(available_actions)
            else:
                action = max(q_values, key=q_values.get)

            print(f"Exploiting: {action} (Q={self.q_table[state][action]:.2f})")

        return action

    def act(self, action: str, environment: Dict) -> Dict:
        """
        Execute an action and return the result.

        Args:
            action: Action to execute
            environment: Current environment (for simulation)

        Returns:
            Action result
        """
        current_pos = environment.get("current_position", 0)
        goal = environment.get("goal", 10)
        obstacles = environment.get("obstacles", [])

        # Execute action
        if action == "move_forward":
            new_pos = current_pos + 1
        elif action == "move_backward":
            new_pos = current_pos - 1
        elif action == "jump":
            new_pos = current_pos + 2
        else:
            new_pos = current_pos

        # Check for obstacles
        if new_pos in obstacles:
            new_pos = current_pos
            reward = -10  # Penalty for hitting obstacle
            success = False
        elif new_pos == goal:
            reward = 100  # Reward for reaching goal
            success = True
        elif new_pos < 0 or new_pos > goal + 5:
            new_pos = current_pos
            reward = -5  # Penalty for going out of bounds
            success = False
        else:
            reward = -1  # Small penalty for each step
            success = True

        result = {
            "action": action,
            "old_position": current_pos,
            "new_position": new_pos,
            "reward": reward,
            "success": success,
            "goal_reached": new_pos == goal
        }

        # Store action
        self.actions_taken.append({
            "action": action,
            "result": result,
            "timestamp": datetime.now()
        })

        print(f"Executed {action}: pos {current_pos} -> {new_pos}, reward={reward}")

        return result

    def learn(self, state: Dict, action: str, reward: float, next_state: Dict):
        """
        Learn from the outcome using Q-learning update.

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Q-learning update formula
        old_q = self.q_table[state_key][action]

        # Get max Q-value for next state
        next_actions = self._get_available_actions(next_state)
        if next_actions:
            max_next_q = max(self.q_table[next_state_key][a] for a in next_actions)
        else:
            max_next_q = 0

        # Update Q-value
        new_q = old_q + self.learning_rate * (reward + 0.9 * max_next_q - old_q)
        self.q_table[state_key][action] = new_q

        # Store reward
        self.rewards_earned.append(reward)

        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)

        print(f"Learned: Q[{state_key}][{action}] = {old_q:.2f} -> {new_q:.2f}")

    def run_episode(self, environment: Dict) -> Dict:
        """
        Run a complete perception-decision-action-learning cycle.

        Args:
            environment: Current environment state

        Returns:
            Episode summary
        """
        self.episode_count += 1
        print(f"\n=== Episode {self.episode_count} ===")

        episode_reward = 0
        steps = 0
        max_steps = 50

        # Make a copy of environment to modify
        env = environment.copy()

        while steps < max_steps:
            # Perceive
            observations = self.perceive(env)

            # Decide
            action = self.decide(observations)

            # Act
            result = self.act(action, env)

            # Update environment
            env["current_position"] = result["new_position"]

            # Learn
            next_observations = self.perceive(env)
            self.learn(observations, action, result["reward"], next_observations)

            episode_reward += result["reward"]
            steps += 1

            # Check if goal reached
            if result["goal_reached"]:
                print(f"Goal reached in {steps} steps!")
                break

        summary = {
            "episode": self.episode_count,
            "steps": steps,
            "total_reward": episode_reward,
            "goal_reached": env["current_position"] == env.get("goal", 10),
            "final_position": env["current_position"],
            "epsilon": self.epsilon
        }

        print(f"Episode summary: {summary}")

        return summary

    def _get_state_key(self, observations: Dict) -> str:
        """Create a hashable state representation."""
        pos = observations.get("current_position", 0)
        goal = observations.get("goal_position", 10)
        obstacle_ahead = observations.get("obstacle_ahead", False)

        return f"{pos}_{goal}_{obstacle_ahead}"

    def _get_available_actions(self, observations: Dict) -> List[str]:
        """Get list of available actions based on current state."""
        actions = ["move_forward", "move_backward", "stay"]

        # Add jump if obstacle ahead
        if observations.get("obstacle_ahead", False):
            actions.append("jump")

        return actions


# Test implementation
agent = BasicAgent("agent_001")
environment = {"obstacles": [3, 5, 7], "goal": 10, "current_position": 0}

# Run multiple episodes to see learning
for i in range(3):
    result = agent.run_episode(environment.copy())
    print(f"Episode {i+1} reward: {result['total_reward']}")


# Solution 2: Implement Memory Retrieval System
print("\n" + "=" * 50)
print("Solution 2: Implement Memory Retrieval System")
print("=" * 50)


class MemorySystem:
    """Memory system with semantic retrieval capabilities."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.importance_scores = {}
        self.access_counts = defaultdict(int)
        self.memory_index = 0

    def store(self, content: Any, metadata: Dict = None, importance: float = 1.0):
        """
        Store a memory with metadata and importance.

        Args:
            content: Memory content
            metadata: Additional metadata
            importance: Importance score (0-1)
        """
        memory_id = f"mem_{self.memory_index}"
        self.memory_index += 1

        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "timestamp": datetime.now(),
            "last_accessed": datetime.now()
        }

        # Handle capacity
        if len(self.memories) >= self.capacity:
            # Remove least important old memory
            min_importance = float('inf')
            min_idx = -1

            for i, mem in enumerate(self.memories):
                # Calculate weighted importance (importance * recency)
                age = (datetime.now() - mem["timestamp"]).seconds + 1
                weighted_importance = mem["importance"] / np.log(age + 1)

                if weighted_importance < min_importance:
                    min_importance = weighted_importance
                    min_idx = i

            if min_idx >= 0:
                removed = self.memories[min_idx]
                del self.memories[min_idx]
                if removed["id"] in self.importance_scores:
                    del self.importance_scores[removed["id"]]

        self.memories.append(memory)
        self.importance_scores[memory_id] = importance

        print(f"Stored memory {memory_id}: {str(content)[:50]}...")

        return memory_id

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve k most relevant memories for a query.

        Args:
            query: Search query
            k: Number of memories to retrieve

        Returns:
            List of relevant memories
        """
        if not self.memories:
            return []

        query_terms = set(query.lower().split())
        scored_memories = []

        for memory in self.memories:
            # Calculate relevance score
            content_str = str(memory["content"]).lower()
            content_terms = set(content_str.split())

            # Term overlap similarity
            overlap = len(query_terms & content_terms)
            similarity = overlap / max(len(query_terms), 1)

            # Boost by importance
            importance = memory["importance"]

            # Recency factor
            age = (datetime.now() - memory["timestamp"]).seconds
            recency_factor = 1.0 / (1 + age / 3600)  # Decay over hours

            # Access frequency factor
            access_factor = 1 + self.access_counts[memory["id"]] * 0.1

            # Combined score
            score = similarity * importance * (0.5 + 0.3 * recency_factor + 0.2 * access_factor)

            scored_memories.append((memory, score))

        # Sort by score and return top k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        results = []

        for memory, score in scored_memories[:k]:
            # Update access info
            memory["last_accessed"] = datetime.now()
            self.access_counts[memory["id"]] += 1

            result = memory.copy()
            result["relevance_score"] = score
            results.append(result)

        print(f"Retrieved {len(results)} memories for query: {query}")

        return results

    def consolidate(self, threshold: float = 0.8):
        """
        Consolidate similar memories to save space.

        Args:
            threshold: Similarity threshold for consolidation
        """
        if len(self.memories) < 2:
            return

        consolidated_count = 0
        memories_list = list(self.memories)

        i = 0
        while i < len(memories_list) - 1:
            j = i + 1
            while j < len(memories_list):
                # Calculate similarity
                mem1_content = str(memories_list[i]["content"]).lower()
                mem2_content = str(memories_list[j]["content"]).lower()

                terms1 = set(mem1_content.split())
                terms2 = set(mem2_content.split())

                if terms1 and terms2:
                    similarity = len(terms1 & terms2) / len(terms1 | terms2)

                    if similarity >= threshold:
                        # Consolidate memories
                        consolidated = self._merge_memories(memories_list[i], memories_list[j])

                        # Replace first memory with consolidated
                        memories_list[i] = consolidated

                        # Remove second memory
                        removed = memories_list.pop(j)
                        if removed["id"] in self.importance_scores:
                            del self.importance_scores[removed["id"]]

                        consolidated_count += 1
                    else:
                        j += 1
                else:
                    j += 1
            i += 1

        # Update memories
        self.memories = deque(memories_list, maxlen=self.capacity)

        print(f"Consolidated {consolidated_count} memory pairs")

    def forget(self, decay_rate: float = 0.1):
        """
        Apply forgetting to reduce importance of old memories.

        Args:
            decay_rate: Rate of importance decay
        """
        forgotten = []
        threshold = 0.1  # Forget memories below this importance

        for memory in self.memories:
            # Calculate age in hours
            age_hours = (datetime.now() - memory["timestamp"]).seconds / 3600

            # Apply exponential decay
            decay_factor = np.exp(-decay_rate * age_hours)

            # Update importance
            old_importance = memory["importance"]
            memory["importance"] = old_importance * decay_factor
            self.importance_scores[memory["id"]] = memory["importance"]

            # Mark for forgetting if below threshold
            if memory["importance"] < threshold:
                forgotten.append(memory)

        # Remove forgotten memories
        for memory in forgotten:
            self.memories.remove(memory)
            if memory["id"] in self.importance_scores:
                del self.importance_scores[memory["id"]]

        print(f"Forgot {len(forgotten)} memories (decay_rate={decay_rate})")

    def _merge_memories(self, mem1: Dict, mem2: Dict) -> Dict:
        """Merge two similar memories."""
        # Combine content
        content = f"{mem1['content']} | {mem2['content']}"

        # Merge metadata
        metadata = {**mem1.get("metadata", {}), **mem2.get("metadata", {})}

        # Average importance
        importance = (mem1["importance"] + mem2["importance"]) / 2

        # Use earlier timestamp
        timestamp = min(mem1["timestamp"], mem2["timestamp"])

        consolidated = {
            "id": mem1["id"],  # Keep first ID
            "content": content,
            "metadata": metadata,
            "importance": importance,
            "timestamp": timestamp,
            "last_accessed": datetime.now(),
            "consolidated": True
        }

        return consolidated


# Test implementation
memory = MemorySystem(capacity=50)

# Store various memories
memory.store("Python is a programming language", {"category": "fact"}, importance=0.9)
memory.store("AI uses Python extensively", {"category": "fact"}, importance=0.8)
memory.store("Machine learning requires Python", {"category": "fact"}, importance=0.7)
memory.store("Data science uses Python", {"category": "fact"}, importance=0.85)
memory.store("Completed task successfully", {"category": "experience"}, importance=0.6)

# Retrieve memories
results = memory.retrieve("Python programming", k=3)
for mem in results:
    print(f"  - {mem['content'][:50]}... (score: {mem['relevance_score']:.3f})")

# Consolidate similar memories
memory.consolidate(threshold=0.6)

# Apply forgetting
memory.forget(decay_rate=0.2)


# Solution 3: Create Adaptive Planning Mechanism
print("\n" + "=" * 50)
print("Solution 3: Create Adaptive Planning Mechanism")
print("=" * 50)


@dataclass
class Plan:
    """Represents a plan with steps and goals."""
    goal: str
    steps: List[str]
    preconditions: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    success_probability: float = 1.0
    status: str = "pending"


class AdaptivePlanner:
    """Adaptive planning system that learns from failures."""

    def __init__(self):
        self.current_plan = None
        self.plan_history = []
        self.failure_patterns = defaultdict(int)
        self.successful_strategies = defaultdict(list)
        self.plan_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict:
        """Initialize planning templates for common goals."""
        return {
            "analysis": ["gather_data", "clean_data", "analyze", "report"],
            "development": ["design", "implement", "test", "deploy"],
            "research": ["literature_review", "hypothesis", "experiment", "conclude"],
            "default": ["understand", "plan", "execute", "verify"]
        }

    def create_plan(self, goal: str, constraints: Dict = None) -> Plan:
        """
        Create a plan to achieve a goal.

        Args:
            goal: Goal to achieve
            constraints: Planning constraints

        Returns:
            Generated plan
        """
        print(f"Creating plan for goal: {goal}")

        # Check for successful strategies
        if goal in self.successful_strategies:
            # Use previously successful strategy
            steps = self.successful_strategies[goal][-1]  # Most recent success
            print(f"Using previously successful strategy")
        else:
            # Select template based on goal keywords
            template_key = self._select_template(goal)
            steps = self.plan_templates[template_key].copy()

            # Customize steps based on goal
            steps = [f"{step}_for_{goal.replace(' ', '_')}" for step in steps]

        # Determine preconditions
        preconditions = []
        if constraints:
            if "resources" in constraints:
                preconditions.append("resources_available")
            if "deadline" in constraints:
                preconditions.append("time_sufficient")
            if "dependencies" in constraints:
                preconditions.extend(constraints["dependencies"])

        # Estimate cost and success probability
        base_cost = len(steps) * 10
        if constraints and "budget" in constraints:
            cost_factor = min(constraints["budget"] / base_cost, 2.0)
        else:
            cost_factor = 1.0

        estimated_cost = base_cost * cost_factor

        # Adjust success probability based on failure history
        failure_key = self._get_failure_key(goal)
        failure_count = self.failure_patterns[failure_key]
        success_probability = max(0.3, 1.0 - failure_count * 0.1)

        plan = Plan(
            goal=goal,
            steps=steps,
            preconditions=preconditions,
            estimated_cost=estimated_cost,
            success_probability=success_probability
        )

        self.current_plan = plan
        self.plan_history.append(plan)

        print(f"Created plan with {len(steps)} steps, success probability: {success_probability:.2f}")

        return plan

    def execute_step(self, step: str) -> Tuple[bool, Any]:
        """
        Execute a single plan step.

        Args:
            step: Step to execute

        Returns:
            Tuple of (success, result)
        """
        print(f"Executing step: {step}")

        # Simulate execution with probability of failure
        if "test" in step.lower():
            # Tests have higher failure rate
            success = np.random.random() > 0.3
        elif "deploy" in step.lower():
            # Deployment is risky
            success = np.random.random() > 0.2
        else:
            # Normal steps
            success = np.random.random() > 0.1

        if success:
            result = f"Successfully completed: {step}"
        else:
            result = f"Failed at: {step}"

        print(f"  Result: {'✓' if success else '✗'} {result}")

        return success, result

    def monitor_execution(self, plan: Plan) -> Dict:
        """
        Monitor plan execution and detect failures.

        Args:
            plan: Plan being executed

        Returns:
            Execution status
        """
        print(f"Monitoring execution of plan for: {plan.goal}")

        execution_status = {
            "plan_goal": plan.goal,
            "total_steps": len(plan.steps),
            "completed_steps": 0,
            "failed_steps": [],
            "status": "in_progress",
            "need_replan": False
        }

        for i, step in enumerate(plan.steps):
            success, result = self.execute_step(step)

            if success:
                execution_status["completed_steps"] += 1
            else:
                execution_status["failed_steps"].append({
                    "step": step,
                    "index": i,
                    "reason": result
                })

                # Check if we should continue or replan
                if len(execution_status["failed_steps"]) >= 2:
                    execution_status["status"] = "failed"
                    execution_status["need_replan"] = True
                    break

        if execution_status["completed_steps"] == len(plan.steps):
            execution_status["status"] = "completed"
            # Record successful strategy
            self.successful_strategies[plan.goal].append(plan.steps)
        elif not execution_status["need_replan"]:
            execution_status["status"] = "partial_success"

        print(f"Execution status: {execution_status['status']}")
        print(f"  Completed: {execution_status['completed_steps']}/{execution_status['total_steps']}")

        return execution_status

    def replan(self, failed_plan: Plan, failure_info: Dict) -> Plan:
        """
        Create alternative plan when current plan fails.

        Args:
            failed_plan: Plan that failed
            failure_info: Information about the failure

        Returns:
            New alternative plan
        """
        print(f"Replanning after failure in: {failed_plan.goal}")

        # Record failure pattern
        failure_key = self._get_failure_key(failed_plan.goal)
        self.failure_patterns[failure_key] += 1

        # Analyze failed steps
        failed_steps = failure_info.get("failed_steps", [])

        # Create alternative plan
        new_steps = []

        for i, original_step in enumerate(failed_plan.steps):
            # Check if this step failed
            step_failed = any(f["index"] == i for f in failed_steps)

            if step_failed:
                # Replace with alternative approach
                if "test" in original_step:
                    # Add more thorough testing
                    new_steps.extend([
                        f"unit_{original_step}",
                        f"integration_{original_step}",
                        f"validate_{original_step}"
                    ])
                elif "deploy" in original_step:
                    # Add safer deployment
                    new_steps.extend([
                        f"backup_before_{original_step}",
                        f"staged_{original_step}",
                        f"verify_{original_step}"
                    ])
                else:
                    # Add retry with preparation
                    new_steps.extend([
                        f"prepare_{original_step}",
                        f"retry_{original_step}",
                        f"verify_{original_step}"
                    ])
            else:
                # Keep successful steps
                new_steps.append(original_step)

        # Create new plan with adjusted probability
        new_plan = Plan(
            goal=f"revised_{failed_plan.goal}",
            steps=new_steps,
            preconditions=failed_plan.preconditions + ["learn_from_failure"],
            estimated_cost=failed_plan.estimated_cost * 1.5,
            success_probability=max(0.5, failed_plan.success_probability - 0.1)
        )

        self.current_plan = new_plan
        self.plan_history.append(new_plan)

        print(f"Created revised plan with {len(new_steps)} steps")
        print(f"  New success probability: {new_plan.success_probability:.2f}")

        return new_plan

    def _select_template(self, goal: str) -> str:
        """Select appropriate template based on goal."""
        goal_lower = goal.lower()

        if "analy" in goal_lower:
            return "analysis"
        elif "develop" in goal_lower or "build" in goal_lower:
            return "development"
        elif "research" in goal_lower or "study" in goal_lower:
            return "research"
        else:
            return "default"

    def _get_failure_key(self, goal: str) -> str:
        """Get key for tracking failure patterns."""
        # Simplify goal to key words
        words = goal.lower().split()
        key_words = [w for w in words if len(w) > 4]
        return "_".join(key_words[:3])


# Test implementation
planner = AdaptivePlanner()

# Create and execute plan
plan1 = planner.create_plan("Complete data analysis task", {"budget": 100, "deadline": "tomorrow"})
print(f"\nPlan steps: {plan1.steps}")

# Monitor execution (will have some failures)
status1 = planner.monitor_execution(plan1)

# Replan if needed
if status1["need_replan"]:
    plan2 = planner.replan(plan1, status1)
    print(f"\nRevised plan steps: {plan2.steps}")
    status2 = planner.monitor_execution(plan2)


# Solution 4: Design Multi-Agent Collaboration
print("\n" + "=" * 50)
print("Solution 4: Design Multi-Agent Collaboration")
print("=" * 50)


class CollaborativeAgent:
    """Agent that can collaborate with others."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.inbox = deque(maxlen=100)
        self.collaborators = []
        self.expertise = self._get_expertise()
        self.reputation = 1.0

    def _get_expertise(self) -> Dict:
        """Get expertise levels based on role."""
        expertise_map = {
            "analyzer": {"analysis": 0.9, "planning": 0.7, "execution": 0.5},
            "executor": {"analysis": 0.5, "planning": 0.6, "execution": 0.9},
            "validator": {"analysis": 0.7, "planning": 0.5, "execution": 0.8},
            "coordinator": {"analysis": 0.6, "planning": 0.9, "execution": 0.7}
        }
        return expertise_map.get(self.role, {"general": 0.7})

    def send_message(self, recipient: 'CollaborativeAgent', message: Dict):
        """
        Send message to another agent.

        Args:
            recipient: Agent to receive message
            message: Message content
        """
        full_message = {
            "from": self.agent_id,
            "to": recipient.agent_id,
            "content": message,
            "timestamp": datetime.now()
        }

        recipient.inbox.append(full_message)
        print(f"{self.agent_id} -> {recipient.agent_id}: {message.get('type', 'message')}")

    def propose_solution(self, problem: Dict) -> Dict:
        """
        Propose a solution to a problem based on role expertise.

        Args:
            problem: Problem description

        Returns:
            Proposed solution
        """
        problem_type = problem.get("type", "general")

        # Generate solution based on role
        if self.role == "analyzer":
            solution = {
                "approach": "analytical",
                "steps": ["decompose", "analyze_components", "identify_patterns"],
                "confidence": self.expertise.get("analysis", 0.7)
            }
        elif self.role == "executor":
            solution = {
                "approach": "practical",
                "steps": ["prepare_resources", "execute_tasks", "verify_results"],
                "confidence": self.expertise.get("execution", 0.7)
            }
        elif self.role == "validator":
            solution = {
                "approach": "verification",
                "steps": ["define_criteria", "test_solution", "validate_output"],
                "confidence": self.expertise.get("analysis", 0.7) * 0.8
            }
        else:
            solution = {
                "approach": "coordinated",
                "steps": ["plan", "delegate", "integrate"],
                "confidence": self.expertise.get("planning", 0.7)
            }

        solution["proposed_by"] = self.agent_id
        solution["problem_id"] = problem.get("id", "unknown")

        print(f"{self.agent_id} proposes {solution['approach']} approach "
              f"(confidence: {solution['confidence']:.2f})")

        return solution

    def vote_on_proposal(self, proposal: Dict) -> str:
        """
        Vote on a proposal from another agent.

        Args:
            proposal: Proposal to vote on

        Returns:
            Vote (approve/reject/abstain)
        """
        # Don't vote on own proposal
        if proposal.get("proposed_by") == self.agent_id:
            return "abstain"

        # Evaluate proposal based on expertise
        confidence = proposal.get("confidence", 0.5)
        approach = proposal.get("approach", "")

        # Calculate alignment with own expertise
        if approach == "analytical" and self.role == "analyzer":
            alignment = 0.9
        elif approach == "practical" and self.role == "executor":
            alignment = 0.9
        elif approach == "verification" and self.role == "validator":
            alignment = 0.9
        else:
            alignment = 0.5

        # Make decision
        score = confidence * alignment * self.reputation

        if score > 0.6:
            vote = "approve"
        elif score < 0.4:
            vote = "reject"
        else:
            vote = "abstain"

        print(f"{self.agent_id} votes: {vote} (score: {score:.2f})")

        return vote

    def process_messages(self) -> List[Dict]:
        """Process messages in inbox."""
        responses = []

        while self.inbox:
            message = self.inbox.popleft()
            content = message["content"]

            if content.get("type") == "request":
                # Handle request
                response = {
                    "type": "response",
                    "request_id": content.get("id"),
                    "data": f"Response from {self.agent_id}"
                }
                responses.append(response)
            elif content.get("type") == "proposal":
                # Vote on proposal
                vote = self.vote_on_proposal(content)
                response = {
                    "type": "vote",
                    "proposal_id": content.get("id"),
                    "vote": vote
                }
                responses.append(response)

        return responses


class MultiAgentSystem:
    """System for multi-agent collaboration."""

    def __init__(self):
        self.agents = {}
        self.task_queue = deque()
        self.completed_tasks = []
        self.collaboration_history = []

    def add_agent(self, agent: CollaborativeAgent):
        """Add agent to the system and connect with others."""
        self.agents[agent.agent_id] = agent

        # Connect with existing agents
        for other_id, other_agent in self.agents.items():
            if other_id != agent.agent_id:
                agent.collaborators.append(other_agent)
                other_agent.collaborators.append(agent)

        print(f"Added {agent.role} agent: {agent.agent_id} "
              f"(connected to {len(agent.collaborators)} agents)")

    def assign_task(self, task: Dict) -> Dict:
        """
        Assign task to appropriate agents based on roles.

        Args:
            task: Task to assign

        Returns:
            Assignment result
        """
        task_type = task.get("type", "general")
        required_roles = task.get("required_roles", [])

        # Find suitable agents
        assigned_agents = []

        for role in required_roles:
            for agent_id, agent in self.agents.items():
                if agent.role == role and agent not in assigned_agents:
                    assigned_agents.append(agent)
                    break

        if not assigned_agents:
            # Assign to any available agent
            assigned_agents = list(self.agents.values())[:1]

        if not assigned_agents:
            return {"status": "no_agents_available"}

        print(f"\nAssigning task to {len(assigned_agents)} agents")

        # Collaborative problem solving
        proposals = []
        for agent in assigned_agents:
            proposal = agent.propose_solution(task)
            proposals.append(proposal)

        # Reach consensus
        best_proposal = self.reach_consensus(proposals)

        # Execute task
        result = {
            "task": task,
            "assigned_agents": [a.agent_id for a in assigned_agents],
            "selected_proposal": best_proposal,
            "status": "completed"
        }

        self.completed_tasks.append(result)

        return result

    def reach_consensus(self, proposals: List[Dict]) -> Dict:
        """
        Reach consensus among agents on proposals.

        Args:
            proposals: List of proposals

        Returns:
            Consensus decision
        """
        if not proposals:
            return {"error": "no_proposals"}

        print(f"\nReaching consensus on {len(proposals)} proposals")

        # Collect votes for each proposal
        proposal_votes = defaultdict(lambda: {"approve": 0, "reject": 0, "abstain": 0})

        for proposal in proposals:
            proposal_id = f"{proposal['proposed_by']}_{proposal['problem_id']}"

            # Each agent votes
            for agent in self.agents.values():
                vote = agent.vote_on_proposal(proposal)
                proposal_votes[proposal_id][vote] += 1

        # Select proposal with most approvals
        best_proposal = None
        best_score = -1

        for i, proposal in enumerate(proposals):
            proposal_id = f"{proposal['proposed_by']}_{proposal['problem_id']}"
            votes = proposal_votes[proposal_id]

            # Calculate score
            score = votes["approve"] - votes["reject"]

            if score > best_score:
                best_score = score
                best_proposal = proposal

        if best_proposal:
            print(f"Consensus reached: {best_proposal['approach']} approach selected")
        else:
            print("No consensus reached, using first proposal")
            best_proposal = proposals[0] if proposals else {}

        self.collaboration_history.append({
            "proposals": proposals,
            "selected": best_proposal,
            "votes": dict(proposal_votes),
            "timestamp": datetime.now()
        })

        return best_proposal

    def handle_agent_failure(self, failed_agent: str):
        """
        Handle failure of an agent by redistributing work.

        Args:
            failed_agent: ID of failed agent
        """
        print(f"\nHandling failure of agent: {failed_agent}")

        if failed_agent not in self.agents:
            print(f"Agent {failed_agent} not found")
            return

        failed = self.agents[failed_agent]

        # Remove from system
        del self.agents[failed_agent]

        # Remove from other agents' collaborators
        for agent in self.agents.values():
            agent.collaborators = [c for c in agent.collaborators
                                 if c.agent_id != failed_agent]

        # Redistribute any pending tasks
        # (In a real system, would track and reassign active tasks)

        print(f"Removed {failed_agent}, system now has {len(self.agents)} agents")

        # Notify remaining agents
        notification = {
            "type": "agent_failure",
            "failed_agent": failed_agent,
            "timestamp": datetime.now()
        }

        for agent in self.agents.values():
            agent.inbox.append({
                "from": "system",
                "to": agent.agent_id,
                "content": notification,
                "timestamp": datetime.now()
            })


# Test implementation
system = MultiAgentSystem()

# Add diverse agents
system.add_agent(CollaborativeAgent("alpha", "analyzer"))
system.add_agent(CollaborativeAgent("beta", "executor"))
system.add_agent(CollaborativeAgent("gamma", "validator"))
system.add_agent(CollaborativeAgent("delta", "coordinator"))

# Assign tasks requiring collaboration
task1 = {
    "id": "task_001",
    "type": "analysis",
    "description": "Analyze system performance",
    "required_roles": ["analyzer", "validator"]
}

result1 = system.assign_task(task1)
print(f"\nTask result: {result1['status']}")
print(f"Selected approach: {result1['selected_proposal']['approach']}")

# Handle agent failure
system.handle_agent_failure("beta")

# Continue with remaining agents
task2 = {
    "id": "task_002",
    "type": "coordination",
    "description": "Coordinate remaining work",
    "required_roles": ["coordinator"]
}

result2 = system.assign_task(task2)


# Solution 5: Build Tool Selection System
print("\n" + "=" * 50)
print("Solution 5: Build Tool Selection System")
print("=" * 50)


class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.usage_count if self.usage_count > 0 else 0

    def execute(self, input_data: Any) -> Tuple[bool, Any]:
        """Execute tool on input data."""
        start_time = time.time()
        self.usage_count += 1

        try:
            # Simulate tool execution
            result = self._execute_logic(input_data)
            self.success_count += 1
            success = True
        except Exception as e:
            result = f"Error: {str(e)}"
            self.failure_count += 1
            success = False
        finally:
            self.total_execution_time += time.time() - start_time

        return success, result

    def _execute_logic(self, input_data: Any) -> Any:
        """Override in subclasses."""
        return f"Processed {input_data} with {self.name}"


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Performs mathematical calculations")

    def _execute_logic(self, input_data: Any) -> Any:
        if isinstance(input_data, str):
            return eval(input_data)
        return input_data


class SearchTool(Tool):
    def __init__(self):
        super().__init__("search", "Searches for information")

    def _execute_logic(self, input_data: Any) -> Any:
        return f"Search results for: {input_data}"


class TransformTool(Tool):
    def __init__(self):
        super().__init__("transform", "Transforms data formats")

    def _execute_logic(self, input_data: Any) -> Any:
        if isinstance(input_data, list):
            return {"items": input_data, "count": len(input_data)}
        return str(input_data)


class ToolSelector:
    """System for selecting appropriate tools."""

    def __init__(self):
        self.tools = {}
        self.usage_history = []
        self.tool_preferences = defaultdict(float)
        self.task_tool_mapping = defaultdict(list)

    def register_tool(self, tool: Tool):
        """
        Register a new tool.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        self.tool_preferences[tool.name] = 1.0  # Initial preference

        # Extract capabilities from description
        capabilities = tool.description.lower().split()
        for cap in capabilities:
            if len(cap) > 4:  # Filter short words
                self.task_tool_mapping[cap].append(tool.name)

        print(f"Registered tool: {tool.name}")

    def select_tool(self, task: Dict) -> Optional[Tool]:
        """
        Select best tool for a task.

        Args:
            task: Task requiring tool

        Returns:
            Selected tool or None
        """
        task_type = task.get("type", "")
        task_description = task.get("description", "")

        # Find candidate tools
        candidates = set()

        # Check task type mapping
        for word in task_description.lower().split():
            if word in self.task_tool_mapping:
                candidates.update(self.task_tool_mapping[word])

        # If no candidates, consider all tools
        if not candidates:
            candidates = set(self.tools.keys())

        if not candidates:
            return None

        # Score each candidate
        best_tool = None
        best_score = -1

        for tool_name in candidates:
            tool = self.tools[tool_name]

            # Calculate score based on:
            # 1. Success rate
            # 2. Preference score
            # 3. Execution time (inverse)

            success_component = tool.success_rate
            preference_component = self.tool_preferences[tool_name]
            speed_component = 1.0 / (1 + tool.avg_execution_time)

            score = (0.4 * success_component +
                    0.4 * preference_component +
                    0.2 * speed_component)

            if score > best_score:
                best_score = score
                best_tool = tool

        if best_tool:
            print(f"Selected tool '{best_tool.name}' for task "
                  f"(score: {best_score:.2f})")

        return best_tool

    def execute_with_tool(self, tool: Tool, input_data: Any) -> Dict:
        """
        Execute task with selected tool and track results.

        Args:
            tool: Tool to use
            input_data: Input for the tool

        Returns:
            Execution result
        """
        print(f"Executing with {tool.name}...")

        success, result = tool.execute(input_data)

        # Record usage
        usage_record = {
            "tool": tool.name,
            "input": str(input_data)[:50],
            "success": success,
            "result": str(result)[:100],
            "timestamp": datetime.now()
        }

        self.usage_history.append(usage_record)

        # Update preferences
        self.update_preferences(tool, success)

        return {
            "tool_used": tool.name,
            "success": success,
            "result": result,
            "execution_time": tool.avg_execution_time
        }

    def update_preferences(self, tool: Tool, success: bool):
        """
        Update tool preferences based on outcome.

        Args:
            tool: Tool that was used
            success: Whether execution was successful
        """
        # Adjust preference based on success
        if success:
            self.tool_preferences[tool.name] = min(2.0,
                self.tool_preferences[tool.name] * 1.1)
        else:
            self.tool_preferences[tool.name] = max(0.1,
                self.tool_preferences[tool.name] * 0.9)

        print(f"Updated preference for {tool.name}: "
              f"{self.tool_preferences[tool.name]:.2f}")

    def get_tool_statistics(self) -> Dict:
        """Get comprehensive tool usage statistics."""
        stats = {
            "total_tools": len(self.tools),
            "total_executions": len(self.usage_history),
            "tool_performance": {}
        }

        for tool_name, tool in self.tools.items():
            stats["tool_performance"][tool_name] = {
                "usage_count": tool.usage_count,
                "success_rate": tool.success_rate,
                "avg_time": tool.avg_execution_time,
                "preference": self.tool_preferences[tool_name]
            }

        return stats


# Test implementation
selector = ToolSelector()

# Register tools
selector.register_tool(CalculatorTool())
selector.register_tool(SearchTool())
selector.register_tool(TransformTool())

# Test tool selection and execution
tasks = [
    {"type": "math", "description": "performs mathematical calculation", "input": "2 + 3 * 4"},
    {"type": "query", "description": "searches for information", "input": "AI applications"},
    {"type": "data", "description": "transforms data", "input": [1, 2, 3, 4, 5]}
]

for task in tasks:
    print(f"\nProcessing task: {task['type']}")
    selected = selector.select_tool(task)

    if selected:
        result = selector.execute_with_tool(selected, task["input"])
        print(f"  Result: {result['result']}")

# Get statistics
stats = selector.get_tool_statistics()
print(f"\nTool Statistics:")
for tool_name, perf in stats["tool_performance"].items():
    print(f"  {tool_name}:")
    print(f"    Usage: {perf['usage_count']}")
    print(f"    Success rate: {perf['success_rate']:.2%}")
    print(f"    Preference: {perf['preference']:.2f}")


print("\n" + "=" * 50)
print("All Solutions Implemented!")
print("=" * 50)