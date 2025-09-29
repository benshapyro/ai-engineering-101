"""
Module 13: Agent Design
Exercises

Practice implementing various agent architectures and patterns.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from enum import Enum
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Exercise 1: Build a Basic Autonomous Agent
print("=" * 50)
print("Exercise 1: Build a Basic Autonomous Agent")
print("=" * 50)
print("""
Task: Create a basic autonomous agent that can:
1. Perceive its environment
2. Make decisions based on observations
3. Execute actions
4. Learn from outcomes

Requirements:
- Implement perception, decision, and action modules
- Add a simple learning mechanism
- Track agent performance over time
""")


class BasicAgent:
    """Basic autonomous agent to implement."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.observations = []
        self.actions_taken = []
        self.learning_rate = 0.1

    def perceive(self, environment: Dict) -> Dict:
        """
        Perceive the environment and extract relevant information.

        Args:
            environment: Current state of the environment

        Returns:
            Processed observations
        """
        # TODO: Implement perception logic
        # Extract relevant features from environment
        # Store observations for later use
        pass

    def decide(self, observations: Dict) -> str:
        """
        Make a decision based on observations.

        Args:
            observations: Processed observations from perception

        Returns:
            Selected action
        """
        # TODO: Implement decision logic
        # Use observations to select best action
        # Consider past experiences
        pass

    def act(self, action: str) -> Dict:
        """
        Execute an action and return the result.

        Args:
            action: Action to execute

        Returns:
            Action result
        """
        # TODO: Implement action execution
        # Execute the selected action
        # Return results and any feedback
        pass

    def learn(self, outcome: Dict):
        """
        Learn from the outcome of an action.

        Args:
            outcome: Result of the action including reward/penalty
        """
        # TODO: Implement learning mechanism
        # Update internal model based on outcome
        # Adjust future decision making
        pass

    def run_episode(self, environment: Dict) -> Dict:
        """
        Run a complete perception-decision-action-learning cycle.

        Args:
            environment: Current environment state

        Returns:
            Episode summary
        """
        # TODO: Implement complete agent cycle
        # 1. Perceive environment
        # 2. Make decision
        # 3. Execute action
        # 4. Learn from outcome
        # 5. Return summary
        pass


# Test your implementation
# agent = BasicAgent("agent_001")
# environment = {"obstacles": [1, 3, 5], "goal": 10, "current_position": 0}
# result = agent.run_episode(environment)
# print(f"Episode result: {result}")


# Exercise 2: Implement Memory Retrieval System
print("\n" + "=" * 50)
print("Exercise 2: Implement Memory Retrieval System")
print("=" * 50)
print("""
Task: Create a memory system with semantic retrieval capabilities.

Requirements:
1. Store memories with metadata
2. Implement similarity-based retrieval
3. Add importance weighting
4. Support memory consolidation
""")


class MemorySystem:
    """Memory system with retrieval capabilities."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories = []
        self.importance_scores = {}

    def store(self, content: Any, metadata: Dict = None, importance: float = 1.0):
        """
        Store a memory with metadata and importance.

        Args:
            content: Memory content
            metadata: Additional metadata
            importance: Importance score (0-1)
        """
        # TODO: Implement memory storage
        # Handle capacity limits
        # Store with timestamp
        pass

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve k most relevant memories for a query.

        Args:
            query: Search query
            k: Number of memories to retrieve

        Returns:
            List of relevant memories
        """
        # TODO: Implement retrieval logic
        # Calculate similarity between query and memories
        # Weight by importance and recency
        # Return top k matches
        pass

    def consolidate(self, threshold: float = 0.8):
        """
        Consolidate similar memories to save space.

        Args:
            threshold: Similarity threshold for consolidation
        """
        # TODO: Implement consolidation
        # Find similar memories
        # Merge them intelligently
        # Update importance scores
        pass

    def forget(self, decay_rate: float = 0.1):
        """
        Apply forgetting to reduce importance of old memories.

        Args:
            decay_rate: Rate of importance decay
        """
        # TODO: Implement forgetting mechanism
        # Reduce importance of old memories
        # Remove memories below threshold
        pass


# Test your implementation
# memory = MemorySystem(capacity=50)
# memory.store("Python is a programming language", {"category": "fact"}, importance=0.9)
# memory.store("AI uses Python extensively", {"category": "fact"}, importance=0.8)
# results = memory.retrieve("Python programming", k=2)
# print(f"Retrieved memories: {results}")


# Exercise 3: Create Adaptive Planning Mechanism
print("\n" + "=" * 50)
print("Exercise 3: Create Adaptive Planning Mechanism")
print("=" * 50)
print("""
Task: Build a planning system that can adapt when plans fail.

Requirements:
1. Create hierarchical plans with subgoals
2. Monitor plan execution
3. Detect failures and replan
4. Learn from planning failures
""")


@dataclass
class Plan:
    """Represents a plan with steps and goals."""
    goal: str
    steps: List[str]
    preconditions: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    success_probability: float = 1.0


class AdaptivePlanner:
    """Adaptive planning system."""

    def __init__(self):
        self.current_plan = None
        self.plan_history = []
        self.failure_patterns = defaultdict(int)

    def create_plan(self, goal: str, constraints: Dict = None) -> Plan:
        """
        Create a plan to achieve a goal.

        Args:
            goal: Goal to achieve
            constraints: Planning constraints

        Returns:
            Generated plan
        """
        # TODO: Implement planning logic
        # Decompose goal into steps
        # Check preconditions
        # Estimate cost and success probability
        pass

    def execute_step(self, step: str) -> Tuple[bool, Any]:
        """
        Execute a single plan step.

        Args:
            step: Step to execute

        Returns:
            Tuple of (success, result)
        """
        # TODO: Implement step execution
        # Try to execute step
        # Return success status and result
        pass

    def monitor_execution(self, plan: Plan) -> Dict:
        """
        Monitor plan execution and detect failures.

        Args:
            plan: Plan being executed

        Returns:
            Execution status
        """
        # TODO: Implement monitoring
        # Track progress through plan
        # Detect failures or deviations
        # Return status report
        pass

    def replan(self, failed_plan: Plan, failure_info: Dict) -> Plan:
        """
        Create alternative plan when current plan fails.

        Args:
            failed_plan: Plan that failed
            failure_info: Information about the failure

        Returns:
            New alternative plan
        """
        # TODO: Implement replanning
        # Analyze failure
        # Generate alternative approach
        # Learn from failure for future planning
        pass


# Test your implementation
# planner = AdaptivePlanner()
# plan = planner.create_plan("Complete data analysis task")
# print(f"Created plan: {plan}")
# status = planner.monitor_execution(plan)
# print(f"Execution status: {status}")


# Exercise 4: Design Multi-Agent Collaboration
print("\n" + "=" * 50)
print("Exercise 4: Design Multi-Agent Collaboration")
print("=" * 50)
print("""
Task: Create a system where multiple agents collaborate on tasks.

Requirements:
1. Implement agent communication protocol
2. Create role-based task assignment
3. Add consensus mechanism for decisions
4. Handle agent failures gracefully
""")


class CollaborativeAgent:
    """Agent that can collaborate with others."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.inbox = []
        self.collaborators = []

    def send_message(self, recipient: 'CollaborativeAgent', message: Dict):
        """
        Send message to another agent.

        Args:
            recipient: Agent to receive message
            message: Message content
        """
        # TODO: Implement messaging
        pass

    def propose_solution(self, problem: Dict) -> Dict:
        """
        Propose a solution to a problem.

        Args:
            problem: Problem description

        Returns:
            Proposed solution
        """
        # TODO: Create solution based on role
        pass

    def vote_on_proposal(self, proposal: Dict) -> str:
        """
        Vote on a proposal from another agent.

        Args:
            proposal: Proposal to vote on

        Returns:
            Vote (approve/reject/abstain)
        """
        # TODO: Implement voting logic
        pass


class MultiAgentSystem:
    """System for multi-agent collaboration."""

    def __init__(self):
        self.agents = []
        self.task_queue = deque()
        self.completed_tasks = []

    def add_agent(self, agent: CollaborativeAgent):
        """Add agent to the system."""
        # TODO: Register agent and connect with others
        pass

    def assign_task(self, task: Dict) -> Dict:
        """
        Assign task to appropriate agents.

        Args:
            task: Task to assign

        Returns:
            Assignment result
        """
        # TODO: Implement task assignment based on roles
        pass

    def reach_consensus(self, proposals: List[Dict]) -> Dict:
        """
        Reach consensus among agents on proposals.

        Args:
            proposals: List of proposals

        Returns:
            Consensus decision
        """
        # TODO: Implement consensus mechanism
        pass

    def handle_agent_failure(self, failed_agent: str):
        """
        Handle failure of an agent.

        Args:
            failed_agent: ID of failed agent
        """
        # TODO: Implement failure handling
        # Reassign tasks
        # Notify other agents
        pass


# Test your implementation
# system = MultiAgentSystem()
# agent1 = CollaborativeAgent("agent1", "analyzer")
# agent2 = CollaborativeAgent("agent2", "executor")
# system.add_agent(agent1)
# system.add_agent(agent2)
# task = {"type": "analysis", "data": [1, 2, 3]}
# result = system.assign_task(task)
# print(f"Task result: {result}")


# Exercise 5: Build Tool Selection System
print("\n" + "=" * 50)
print("Exercise 5: Build Tool Selection System")
print("=" * 50)
print("""
Task: Create a system for agents to select and use appropriate tools.

Requirements:
1. Register tools with descriptions and capabilities
2. Match tools to tasks based on requirements
3. Track tool usage and effectiveness
4. Learn tool preferences over time
""")


class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.success_rate = 1.0

    def execute(self, input_data: Any) -> Tuple[bool, Any]:
        """Execute tool on input data."""
        # TODO: Implement tool execution
        pass


class ToolSelector:
    """System for selecting appropriate tools."""

    def __init__(self):
        self.tools = {}
        self.usage_history = []
        self.tool_preferences = defaultdict(float)

    def register_tool(self, tool: Tool):
        """
        Register a new tool.

        Args:
            tool: Tool to register
        """
        # TODO: Register tool and extract capabilities
        pass

    def select_tool(self, task: Dict) -> Optional[Tool]:
        """
        Select best tool for a task.

        Args:
            task: Task requiring tool

        Returns:
            Selected tool or None
        """
        # TODO: Implement tool selection
        # Match task requirements to tool capabilities
        # Consider past performance
        # Return best match
        pass

    def execute_with_tool(self, tool: Tool, input_data: Any) -> Dict:
        """
        Execute task with selected tool.

        Args:
            tool: Tool to use
            input_data: Input for the tool

        Returns:
            Execution result
        """
        # TODO: Execute and track results
        pass

    def update_preferences(self, tool: Tool, success: bool):
        """
        Update tool preferences based on outcome.

        Args:
            tool: Tool that was used
            success: Whether execution was successful
        """
        # TODO: Update preference scores
        pass


# Test your implementation
# selector = ToolSelector()
# calc_tool = Tool("calculator", "Performs mathematical calculations")
# selector.register_tool(calc_tool)
# task = {"type": "calculation", "expression": "2 + 2"}
# selected = selector.select_tool(task)
# print(f"Selected tool: {selected.name if selected else 'None'}")


# Challenge Exercise: Complete Agent Framework
print("\n" + "=" * 50)
print("CHALLENGE: Complete Agent Framework")
print("=" * 50)
print("""
Task: Create a complete agent framework combining all previous exercises.

Requirements:
1. Autonomous agent with perception, decision, action, learning
2. Memory system for storing and retrieving experiences
3. Adaptive planning that handles failures
4. Multi-agent collaboration capabilities
5. Tool selection and usage
6. Performance monitoring and optimization

The framework should:
- Support different agent types (reactive, deliberative, hybrid)
- Handle complex multi-step tasks
- Learn and improve over time
- Collaborate with other agents
- Use appropriate tools
- Monitor and report performance
""")


class CompleteAgentFramework:
    """Complete agent framework combining all capabilities."""

    def __init__(self, agent_id: str, agent_type: str = "hybrid"):
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Initialize all subsystems
        self.memory_system = None  # TODO: Initialize
        self.planner = None  # TODO: Initialize
        self.tool_selector = None  # TODO: Initialize

        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_time": 0,
            "learning_rate": 0
        }

    def process_task(self, task: Dict) -> Dict:
        """
        Process a complete task using all capabilities.

        Args:
            task: Task to process

        Returns:
            Task result
        """
        # TODO: Implement complete task processing
        # 1. Perceive task requirements
        # 2. Recall relevant memories
        # 3. Create plan
        # 4. Select tools
        # 5. Execute plan with monitoring
        # 6. Learn from outcome
        # 7. Store experience
        # 8. Return results
        pass

    def collaborate_on_task(self, task: Dict, collaborators: List['CompleteAgentFramework']) -> Dict:
        """
        Collaborate with other agents on a task.

        Args:
            task: Task requiring collaboration
            collaborators: Other agents to work with

        Returns:
            Collaborative result
        """
        # TODO: Implement collaboration
        # 1. Communicate with collaborators
        # 2. Divide task based on capabilities
        # 3. Coordinate execution
        # 4. Integrate results
        # 5. Reach consensus on output
        pass

    def adapt_behavior(self, feedback: Dict):
        """
        Adapt agent behavior based on feedback.

        Args:
            feedback: Performance feedback
        """
        # TODO: Implement adaptation
        # Adjust planning strategies
        # Update tool preferences
        # Modify decision thresholds
        pass

    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.

        Returns:
            Performance metrics and analysis
        """
        # TODO: Generate detailed report
        # Include all metrics
        # Analyze trends
        # Identify areas for improvement
        pass


# Test your complete framework
# agent = CompleteAgentFramework("master_agent", "hybrid")
# task = {
#     "type": "complex_analysis",
#     "requirements": ["data_processing", "pattern_recognition", "reporting"],
#     "deadline": "2 hours",
#     "priority": "high"
# }
# result = agent.process_task(task)
# print(f"Task result: {result}")
# report = agent.get_performance_report()
# print(f"Performance report: {report}")


print("\n" + "=" * 50)
print("Exercises Complete!")
print("=" * 50)
print("""
These exercises cover key agent design concepts:
1. Basic autonomous agent architecture
2. Memory systems with retrieval
3. Adaptive planning mechanisms
4. Multi-agent collaboration
5. Tool selection and usage
6. Complete integrated framework

Try implementing each exercise and experiment with different approaches!
""")