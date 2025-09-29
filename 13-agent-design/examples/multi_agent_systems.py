"""
Module 13: Agent Design
Multi-Agent Systems Examples

This file demonstrates various multi-agent collaboration patterns:
1. Agent communication protocols
2. Role-based agent specialization
3. Collaborative problem solving
4. Consensus mechanisms
5. Task delegation and coordination
6. Emergent behavior patterns
7. Production multi-agent orchestration

Each example shows progressively more sophisticated multi-agent patterns.
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from queue import Queue, PriorityQueue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


# Example 1: Agent Communication Protocols
print("=" * 50)
print("Example 1: Agent Communication Protocols")
print("=" * 50)


class MessageType(Enum):
    """Types of messages agents can send."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    SUBSCRIBE = "subscribe"
    PUBLISH = "publish"
    QUERY = "query"
    INFORM = "inform"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class Message:
    """Message structure for agent communication."""
    sender: str
    receiver: str
    type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: Optional[str] = None
    priority: int = 0


class CommunicationProtocol:
    """Base communication protocol for agents."""

    def __init__(self):
        self.message_queue = Queue()
        self.agents = {}
        self.subscriptions = defaultdict(list)
        self.conversations = {}

    def register_agent(self, agent_id: str, agent):
        """Register an agent in the communication system."""
        self.agents[agent_id] = agent
        print(f"Registered agent: {agent_id}")

    def send_message(self, message: Message):
        """Send a message through the protocol."""
        self.message_queue.put(message)

        print(f"Message: {message.sender} -> {message.receiver}")
        print(f"  Type: {message.type.value}")
        print(f"  Content: {str(message.content)[:50]}...")

        # Route message
        if message.receiver == "broadcast":
            self._broadcast(message)
        elif message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
        else:
            print(f"  Warning: Receiver {message.receiver} not found")

    def _broadcast(self, message: Message):
        """Broadcast message to all agents."""
        for agent_id, agent in self.agents.items():
            if agent_id != message.sender:
                agent.receive_message(message)

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic."""
        self.subscriptions[topic].append(agent_id)
        print(f"Agent {agent_id} subscribed to {topic}")

    def publish(self, topic: str, content: Any, sender: str):
        """Publish to a topic."""
        subscribers = self.subscriptions.get(topic, [])

        for subscriber_id in subscribers:
            if subscriber_id != sender and subscriber_id in self.agents:
                message = Message(
                    sender=sender,
                    receiver=subscriber_id,
                    type=MessageType.PUBLISH,
                    content={"topic": topic, "data": content}
                )
                self.agents[subscriber_id].receive_message(message)

        print(f"Published to {topic}: {len(subscribers)} subscribers")

    def start_conversation(self, initiator: str, participants: List[str]) -> str:
        """Start a multi-agent conversation."""
        conversation_id = f"conv_{len(self.conversations)}"

        self.conversations[conversation_id] = {
            "initiator": initiator,
            "participants": participants,
            "messages": [],
            "status": "active"
        }

        print(f"Started conversation {conversation_id} with {len(participants)} participants")

        return conversation_id


class CommunicatingAgent:
    """Agent that can communicate using protocols."""

    def __init__(self, agent_id: str, protocol: CommunicationProtocol):
        self.agent_id = agent_id
        self.protocol = protocol
        self.inbox = []
        self.sent_messages = []
        self.knowledge = {}

        # Register with protocol
        protocol.register_agent(agent_id, self)

    def send(self, receiver: str, message_type: MessageType, content: Any):
        """Send a message to another agent."""
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            type=message_type,
            content=content
        )

        self.sent_messages.append(message)
        self.protocol.send_message(message)

    def receive_message(self, message: Message):
        """Receive and process a message."""
        self.inbox.append(message)

        # Process based on message type
        if message.type == MessageType.QUERY:
            self._handle_query(message)
        elif message.type == MessageType.REQUEST:
            self._handle_request(message)
        elif message.type == MessageType.INFORM:
            self._handle_inform(message)

    def _handle_query(self, message: Message):
        """Handle a query message."""
        query = message.content

        # Check if we have the information
        if query in self.knowledge:
            # Send response
            self.send(
                message.sender,
                MessageType.RESPONSE,
                self.knowledge[query]
            )
        else:
            self.send(
                message.sender,
                MessageType.RESPONSE,
                "Unknown"
            )

    def _handle_request(self, message: Message):
        """Handle a request message."""
        # Simple echo for demo
        self.send(
            message.sender,
            MessageType.RESPONSE,
            f"Processed request: {message.content}"
        )

    def _handle_inform(self, message: Message):
        """Handle an inform message."""
        # Store information
        if isinstance(message.content, dict):
            self.knowledge.update(message.content)


# Example usage
protocol = CommunicationProtocol()

# Create agents
agent1 = CommunicatingAgent("agent1", protocol)
agent2 = CommunicatingAgent("agent2", protocol)
agent3 = CommunicatingAgent("agent3", protocol)

# Set knowledge
agent2.knowledge = {"weather": "sunny", "temperature": 25}

# Communication examples
agent1.send("agent2", MessageType.QUERY, "weather")
agent1.send("broadcast", MessageType.INFORM, {"status": "ready"})

# Subscribe/publish
protocol.subscribe("agent1", "updates")
protocol.subscribe("agent3", "updates")
protocol.publish("updates", "System online", "agent2")

print(f"\nAgent1 inbox: {len(agent1.inbox)} messages")
print(f"Agent3 inbox: {len(agent3.inbox)} messages")


# Example 2: Role-Based Agent Specialization
print("\n" + "=" * 50)
print("Example 2: Role-Based Agent Specialization")
print("=" * 50)


class AgentRole(Enum):
    """Specialized agent roles."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    VALIDATOR = "validator"


class SpecializedAgent:
    """Agent with specialized role and capabilities."""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = self._get_capabilities()
        self.task_queue = deque()
        self.completed_tasks = []

    def _get_capabilities(self) -> List[str]:
        """Get capabilities based on role."""
        role_capabilities = {
            AgentRole.COORDINATOR: ["plan", "delegate", "coordinate", "monitor_progress"],
            AgentRole.RESEARCHER: ["search", "gather_data", "summarize", "verify_sources"],
            AgentRole.ANALYZER: ["analyze_data", "identify_patterns", "generate_insights", "statistical_analysis"],
            AgentRole.EXECUTOR: ["execute_action", "implement_solution", "apply_changes", "run_tests"],
            AgentRole.MONITOR: ["track_metrics", "detect_anomalies", "generate_alerts", "report_status"],
            AgentRole.VALIDATOR: ["validate_results", "check_quality", "verify_correctness", "approve_outputs"]
        }

        return role_capabilities.get(self.role, [])

    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle a task type."""
        return task_type in self.capabilities

    def execute_task(self, task: Dict) -> Dict:
        """Execute a task based on role."""
        task_type = task.get("type")

        if not self.can_handle(task_type):
            return {
                "status": "rejected",
                "reason": f"Agent {self.agent_id} cannot handle {task_type}"
            }

        print(f"{self.role.value} agent {self.agent_id} executing: {task_type}")

        # Simulate task execution based on role
        if self.role == AgentRole.COORDINATOR:
            result = self._coordinate_task(task)
        elif self.role == AgentRole.RESEARCHER:
            result = self._research_task(task)
        elif self.role == AgentRole.ANALYZER:
            result = self._analyze_task(task)
        elif self.role == AgentRole.EXECUTOR:
            result = self._execute_action_task(task)
        elif self.role == AgentRole.MONITOR:
            result = self._monitor_task(task)
        elif self.role == AgentRole.VALIDATOR:
            result = self._validate_task(task)
        else:
            result = {"status": "unknown_role"}

        # Record completion
        self.completed_tasks.append({
            "task": task,
            "result": result,
            "timestamp": datetime.now()
        })

        return result

    def _coordinate_task(self, task: Dict) -> Dict:
        """Coordinate a task (COORDINATOR role)."""
        subtasks = task.get("subtasks", [])

        return {
            "status": "coordinated",
            "subtasks_assigned": len(subtasks),
            "coordination_plan": f"Delegated {len(subtasks)} subtasks"
        }

    def _research_task(self, task: Dict) -> Dict:
        """Research a task (RESEARCHER role)."""
        query = task.get("query", "")

        return {
            "status": "researched",
            "findings": f"Research results for: {query}",
            "sources": ["source1", "source2", "source3"]
        }

    def _analyze_task(self, task: Dict) -> Dict:
        """Analyze a task (ANALYZER role)."""
        data = task.get("data", [])

        return {
            "status": "analyzed",
            "insights": f"Analyzed {len(data)} data points",
            "patterns_found": 3,
            "recommendations": ["recommendation1", "recommendation2"]
        }

    def _execute_action_task(self, task: Dict) -> Dict:
        """Execute an action task (EXECUTOR role)."""
        action = task.get("action", "")

        return {
            "status": "executed",
            "action": action,
            "outcome": "successful",
            "execution_time": 1.5
        }

    def _monitor_task(self, task: Dict) -> Dict:
        """Monitor a task (MONITOR role)."""
        metrics = task.get("metrics", [])

        return {
            "status": "monitored",
            "metrics_tracked": len(metrics),
            "anomalies_detected": 0,
            "health_status": "healthy"
        }

    def _validate_task(self, task: Dict) -> Dict:
        """Validate a task (VALIDATOR role)."""
        data = task.get("data", {})

        return {
            "status": "validated",
            "validation_passed": True,
            "quality_score": 0.95,
            "issues_found": []
        }

    def get_workload(self) -> Dict:
        """Get current workload statistics."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "capabilities": self.capabilities
        }


class RoleBasedTeam:
    """Team of specialized agents."""

    def __init__(self):
        self.agents = {}
        self.role_registry = defaultdict(list)

    def add_agent(self, agent: SpecializedAgent):
        """Add a specialized agent to the team."""
        self.agents[agent.agent_id] = agent
        self.role_registry[agent.role].append(agent.agent_id)

        print(f"Added {agent.role.value} agent: {agent.agent_id}")

    def assign_task(self, task: Dict) -> Dict:
        """Assign task to appropriate agent."""
        task_type = task.get("type")

        # Find capable agents
        capable_agents = []

        for agent_id, agent in self.agents.items():
            if agent.can_handle(task_type):
                capable_agents.append(agent)

        if not capable_agents:
            return {"status": "no_capable_agent", "task_type": task_type}

        # Select agent with lowest workload
        selected = min(capable_agents, key=lambda a: len(a.task_queue))

        # Execute task
        result = selected.execute_task(task)

        return {
            "assigned_to": selected.agent_id,
            "role": selected.role.value,
            "result": result
        }

    def get_team_status(self) -> Dict:
        """Get status of all team members."""
        status = {
            "team_size": len(self.agents),
            "roles": {},
            "total_completed_tasks": 0
        }

        for role in AgentRole:
            agents_in_role = self.role_registry.get(role, [])
            status["roles"][role.value] = len(agents_in_role)

        for agent in self.agents.values():
            status["total_completed_tasks"] += len(agent.completed_tasks)

        return status


# Example usage
team = RoleBasedTeam()

# Create specialized agents
team.add_agent(SpecializedAgent("coord1", AgentRole.COORDINATOR))
team.add_agent(SpecializedAgent("research1", AgentRole.RESEARCHER))
team.add_agent(SpecializedAgent("research2", AgentRole.RESEARCHER))
team.add_agent(SpecializedAgent("analyzer1", AgentRole.ANALYZER))
team.add_agent(SpecializedAgent("executor1", AgentRole.EXECUTOR))
team.add_agent(SpecializedAgent("monitor1", AgentRole.MONITOR))

# Assign various tasks
tasks = [
    {"type": "search", "query": "AI trends"},
    {"type": "analyze_data", "data": [1, 2, 3, 4, 5]},
    {"type": "execute_action", "action": "deploy_model"},
    {"type": "track_metrics", "metrics": ["latency", "throughput"]}
]

print("\nTask assignments:")
for task in tasks:
    result = team.assign_task(task)
    print(f"  Task '{task['type']}' -> {result['assigned_to']}")

print(f"\nTeam status: {team.get_team_status()}")


# Example 3: Collaborative Problem Solving
print("\n" + "=" * 50)
print("Example 3: Collaborative Problem Solving")
print("=" * 50)


class CollaborativeAgent:
    """Agent that collaborates to solve problems."""

    def __init__(self, agent_id: str, expertise: str):
        self.agent_id = agent_id
        self.expertise = expertise
        self.partial_solutions = {}
        self.shared_knowledge = {}
        self.collaborators = []

    def contribute_to_problem(self, problem_id: str, problem: Dict) -> Dict:
        """Contribute expertise to problem solving."""
        print(f"Agent {self.agent_id} ({self.expertise}) contributing to problem {problem_id}")

        # Generate contribution based on expertise
        if self.expertise == "data_analysis":
            contribution = self._analyze_data_aspect(problem)
        elif self.expertise == "optimization":
            contribution = self._optimize_solution(problem)
        elif self.expertise == "validation":
            contribution = self._validate_approach(problem)
        else:
            contribution = self._general_contribution(problem)

        # Store partial solution
        self.partial_solutions[problem_id] = contribution

        return contribution

    def _analyze_data_aspect(self, problem: Dict) -> Dict:
        """Contribute data analysis expertise."""
        return {
            "contribution_type": "data_analysis",
            "insights": ["pattern_found", "anomaly_detected"],
            "confidence": 0.85,
            "recommendation": "Use statistical approach"
        }

    def _optimize_solution(self, problem: Dict) -> Dict:
        """Contribute optimization expertise."""
        return {
            "contribution_type": "optimization",
            "improvements": ["reduced_complexity", "improved_efficiency"],
            "performance_gain": 0.3,
            "optimized_parameters": {"param1": 0.7, "param2": 1.2}
        }

    def _validate_approach(self, problem: Dict) -> Dict:
        """Contribute validation expertise."""
        return {
            "contribution_type": "validation",
            "validation_passed": True,
            "edge_cases_found": 2,
            "robustness_score": 0.9
        }

    def _general_contribution(self, problem: Dict) -> Dict:
        """General contribution."""
        return {
            "contribution_type": "general",
            "suggestion": "Consider alternative approach",
            "confidence": 0.6
        }

    def share_knowledge(self, knowledge_item: Dict):
        """Share knowledge with collaborators."""
        self.shared_knowledge[knowledge_item["key"]] = knowledge_item["value"]

        # Notify collaborators
        for collaborator in self.collaborators:
            collaborator.receive_shared_knowledge(self.agent_id, knowledge_item)

    def receive_shared_knowledge(self, sender: str, knowledge: Dict):
        """Receive shared knowledge from collaborator."""
        key = f"{sender}_{knowledge['key']}"
        self.shared_knowledge[key] = knowledge["value"]

    def integrate_solutions(self, all_contributions: List[Dict]) -> Dict:
        """Integrate contributions from all collaborators."""
        integrated = {
            "total_contributions": len(all_contributions),
            "contribution_types": [],
            "combined_confidence": 0,
            "final_solution": {}
        }

        confidences = []

        for contribution in all_contributions:
            integrated["contribution_types"].append(contribution.get("contribution_type"))

            # Extract confidence if available
            if "confidence" in contribution:
                confidences.append(contribution["confidence"])

            # Merge into final solution
            integrated["final_solution"].update(contribution)

        # Calculate combined confidence
        if confidences:
            integrated["combined_confidence"] = np.mean(confidences)

        return integrated


class CollaborativeProblemSolver:
    """System for collaborative problem solving."""

    def __init__(self):
        self.agents = []
        self.problems = {}
        self.solutions = {}

    def add_agent(self, agent: CollaborativeAgent):
        """Add an agent to the collaboration."""
        self.agents.append(agent)

        # Connect with other agents
        for other_agent in self.agents:
            if other_agent != agent:
                agent.collaborators.append(other_agent)
                other_agent.collaborators.append(agent)

        print(f"Added collaborative agent: {agent.agent_id} (expertise: {agent.expertise})")

    def solve_problem(self, problem_id: str, problem: Dict) -> Dict:
        """Collaboratively solve a problem."""
        print(f"\nSolving problem: {problem_id}")
        print(f"Problem: {problem}")

        self.problems[problem_id] = problem

        # Phase 1: Individual contributions
        contributions = []

        for agent in self.agents:
            contribution = agent.contribute_to_problem(problem_id, problem)
            contributions.append(contribution)

        # Phase 2: Knowledge sharing
        for agent in self.agents:
            # Share relevant knowledge
            if agent.expertise in problem.get("relevant_expertise", []):
                agent.share_knowledge({
                    "key": f"{agent.expertise}_insight",
                    "value": f"Expert insight from {agent.expertise}"
                })

        # Phase 3: Solution integration
        if self.agents:
            integrator = self.agents[0]  # First agent acts as integrator
            integrated_solution = integrator.integrate_solutions(contributions)
        else:
            integrated_solution = {"error": "No agents available"}

        self.solutions[problem_id] = integrated_solution

        return integrated_solution

    def evaluate_collaboration(self) -> Dict:
        """Evaluate collaboration effectiveness."""
        if not self.solutions:
            return {"effectiveness": 0}

        total_problems = len(self.problems)
        solved_problems = len([s for s in self.solutions.values()
                              if s.get("combined_confidence", 0) > 0.7])

        knowledge_shared = sum(len(agent.shared_knowledge) for agent in self.agents)

        return {
            "total_problems": total_problems,
            "solved_problems": solved_problems,
            "solve_rate": solved_problems / total_problems if total_problems > 0 else 0,
            "total_agents": len(self.agents),
            "knowledge_items_shared": knowledge_shared,
            "average_confidence": np.mean([s.get("combined_confidence", 0)
                                          for s in self.solutions.values()])
        }


# Example usage
solver = CollaborativeProblemSolver()

# Add agents with different expertise
solver.add_agent(CollaborativeAgent("agent_a", "data_analysis"))
solver.add_agent(CollaborativeAgent("agent_b", "optimization"))
solver.add_agent(CollaborativeAgent("agent_c", "validation"))

# Solve a problem collaboratively
problem = {
    "type": "optimization_problem",
    "description": "Optimize resource allocation",
    "constraints": ["budget_limit", "time_limit"],
    "relevant_expertise": ["optimization", "data_analysis"]
}

solution = solver.solve_problem("prob_001", problem)
print(f"\nIntegrated solution:")
print(f"  Contribution types: {solution['contribution_types']}")
print(f"  Combined confidence: {solution['combined_confidence']:.2f}")

# Evaluate collaboration
evaluation = solver.evaluate_collaboration()
print(f"\nCollaboration evaluation: {evaluation}")


# Example 4: Consensus Mechanisms
print("\n" + "=" * 50)
print("Example 4: Consensus Mechanisms")
print("=" * 50)


class VotingAgent:
    """Agent that participates in consensus voting."""

    def __init__(self, agent_id: str, bias: float = 0.0):
        self.agent_id = agent_id
        self.bias = bias  # -1 to 1, affects voting behavior
        self.vote_history = []
        self.reputation = 1.0

    def vote(self, proposal: Dict) -> str:
        """Vote on a proposal."""
        # Analyze proposal
        score = self._evaluate_proposal(proposal)

        # Apply bias
        score += self.bias

        # Make decision
        if score > 0.5:
            vote = "approve"
        elif score < -0.5:
            vote = "reject"
        else:
            vote = "abstain"

        # Record vote
        self.vote_history.append({
            "proposal_id": proposal.get("id"),
            "vote": vote,
            "timestamp": datetime.now()
        })

        return vote

    def _evaluate_proposal(self, proposal: Dict) -> float:
        """Evaluate a proposal (simplified)."""
        # Random evaluation with some logic
        base_score = np.random.uniform(-1, 1)

        # Adjust based on proposal attributes
        if proposal.get("priority") == "high":
            base_score += 0.3

        if proposal.get("risk") == "low":
            base_score += 0.2

        if proposal.get("cost", 0) > 1000:
            base_score -= 0.4

        return np.clip(base_score, -1, 1)

    def weighted_vote(self, proposal: Dict) -> Tuple[str, float]:
        """Vote with weight based on reputation."""
        vote = self.vote(proposal)
        weight = self.reputation

        return vote, weight


class ConsensusProtocol:
    """Protocol for reaching consensus among agents."""

    def __init__(self, consensus_type: str = "majority"):
        self.consensus_type = consensus_type
        self.agents = []
        self.proposals = {}
        self.voting_results = {}

    def add_agent(self, agent: VotingAgent):
        """Add an agent to the consensus protocol."""
        self.agents.append(agent)
        print(f"Added agent {agent.agent_id} to consensus (reputation: {agent.reputation:.2f})")

    def propose(self, proposal: Dict) -> str:
        """Submit a proposal for consensus."""
        proposal_id = proposal.get("id", f"prop_{len(self.proposals)}")
        proposal["id"] = proposal_id

        self.proposals[proposal_id] = proposal

        print(f"Proposal submitted: {proposal_id}")
        print(f"  Type: {proposal.get('type')}")
        print(f"  Priority: {proposal.get('priority')}")

        return proposal_id

    def reach_consensus(self, proposal_id: str) -> Dict:
        """Reach consensus on a proposal."""
        if proposal_id not in self.proposals:
            return {"error": "Proposal not found"}

        proposal = self.proposals[proposal_id]

        print(f"\nReaching consensus on: {proposal_id}")

        # Collect votes based on consensus type
        if self.consensus_type == "majority":
            result = self._majority_voting(proposal)
        elif self.consensus_type == "weighted":
            result = self._weighted_voting(proposal)
        elif self.consensus_type == "byzantine":
            result = self._byzantine_consensus(proposal)
        elif self.consensus_type == "unanimous":
            result = self._unanimous_voting(proposal)
        else:
            result = {"error": "Unknown consensus type"}

        self.voting_results[proposal_id] = result

        return result

    def _majority_voting(self, proposal: Dict) -> Dict:
        """Simple majority voting."""
        votes = {"approve": 0, "reject": 0, "abstain": 0}

        for agent in self.agents:
            vote = agent.vote(proposal)
            votes[vote] += 1

        total_votes = sum(votes.values())
        approval_rate = votes["approve"] / total_votes if total_votes > 0 else 0

        decision = "approved" if votes["approve"] > votes["reject"] else "rejected"

        return {
            "consensus_type": "majority",
            "decision": decision,
            "votes": votes,
            "approval_rate": approval_rate,
            "participation": total_votes
        }

    def _weighted_voting(self, proposal: Dict) -> Dict:
        """Weighted voting based on reputation."""
        weighted_votes = {"approve": 0, "reject": 0, "abstain": 0}

        for agent in self.agents:
            vote, weight = agent.weighted_vote(proposal)
            weighted_votes[vote] += weight

        total_weight = sum(weighted_votes.values())
        approval_weight = weighted_votes["approve"] / total_weight if total_weight > 0 else 0

        decision = "approved" if weighted_votes["approve"] > weighted_votes["reject"] else "rejected"

        return {
            "consensus_type": "weighted",
            "decision": decision,
            "weighted_votes": weighted_votes,
            "approval_weight": approval_weight,
            "total_weight": total_weight
        }

    def _byzantine_consensus(self, proposal: Dict) -> Dict:
        """Byzantine fault-tolerant consensus (simplified)."""
        votes = []

        # Multiple rounds of voting
        rounds = 3

        for round_num in range(rounds):
            round_votes = {"approve": 0, "reject": 0, "abstain": 0}

            for agent in self.agents:
                vote = agent.vote(proposal)
                round_votes[vote] += 1

            votes.append(round_votes)

        # Require 2/3 majority across rounds
        final_approve = sum(v["approve"] for v in votes)
        final_reject = sum(v["reject"] for v in votes)
        total_votes = len(self.agents) * rounds

        byzantine_threshold = 2 * total_votes / 3

        if final_approve >= byzantine_threshold:
            decision = "approved"
        elif final_reject >= byzantine_threshold:
            decision = "rejected"
        else:
            decision = "no_consensus"

        return {
            "consensus_type": "byzantine",
            "decision": decision,
            "rounds": rounds,
            "final_approve": final_approve,
            "final_reject": final_reject,
            "byzantine_threshold": byzantine_threshold
        }

    def _unanimous_voting(self, proposal: Dict) -> Dict:
        """Unanimous consensus required."""
        votes = {"approve": 0, "reject": 0, "abstain": 0}

        for agent in self.agents:
            vote = agent.vote(proposal)
            votes[vote] += 1

        # All must approve (abstentions allowed)
        decision = "approved" if votes["reject"] == 0 and votes["approve"] > 0 else "rejected"

        return {
            "consensus_type": "unanimous",
            "decision": decision,
            "votes": votes,
            "unanimous": votes["reject"] == 0
        }


# Example usage
consensus = ConsensusProtocol(consensus_type="weighted")

# Add agents with different biases and reputations
agent1 = VotingAgent("conservative", bias=-0.3)
agent1.reputation = 0.9

agent2 = VotingAgent("progressive", bias=0.3)
agent2.reputation = 1.2

agent3 = VotingAgent("neutral", bias=0.0)
agent3.reputation = 1.0

agent4 = VotingAgent("skeptical", bias=-0.5)
agent4.reputation = 0.8

consensus.add_agent(agent1)
consensus.add_agent(agent2)
consensus.add_agent(agent3)
consensus.add_agent(agent4)

# Submit proposals
proposals = [
    {"type": "feature", "priority": "high", "cost": 500, "risk": "low"},
    {"type": "refactor", "priority": "medium", "cost": 2000, "risk": "medium"},
    {"type": "experiment", "priority": "low", "cost": 100, "risk": "high"}
]

print("\nConsensus results:")
for prop in proposals:
    prop_id = consensus.propose(prop)
    result = consensus.reach_consensus(prop_id)
    print(f"  {prop['type']}: {result['decision']} (approval weight: {result.get('approval_weight', 0):.2f})")


# Example 5: Task Delegation and Coordination
print("\n" + "=" * 50)
print("Example 5: Task Delegation and Coordination")
print("=" * 50)


@dataclass
class Task:
    """Task to be executed by agents."""
    task_id: str
    description: str
    priority: int  # 1 (highest) to 5 (lowest)
    dependencies: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    estimated_effort: float = 1.0
    status: str = "pending"
    assigned_to: Optional[str] = None
    result: Optional[Any] = None


class TaskCoordinator:
    """Coordinates task delegation among agents."""

    def __init__(self):
        self.agents = {}
        self.task_queue = PriorityQueue()
        self.tasks = {}
        self.completed_tasks = []
        self.task_dependencies = defaultdict(list)

    def register_agent(self, agent_id: str, skills: List[str], capacity: int = 3):
        """Register an agent with skills and capacity."""
        self.agents[agent_id] = {
            "skills": skills,
            "capacity": capacity,
            "current_load": 0,
            "assigned_tasks": []
        }

        print(f"Registered agent {agent_id} with skills: {skills}")

    def submit_task(self, task: Task):
        """Submit a task for delegation."""
        self.tasks[task.task_id] = task

        # Add to priority queue (negative priority for min-heap)
        self.task_queue.put((-task.priority, task.task_id))

        # Track dependencies
        for dep in task.dependencies:
            self.task_dependencies[dep].append(task.task_id)

        print(f"Submitted task: {task.task_id} (priority: {task.priority})")

    def delegate_tasks(self) -> List[Dict]:
        """Delegate pending tasks to available agents."""
        delegations = []

        while not self.task_queue.empty():
            _, task_id = self.task_queue.get()
            task = self.tasks[task_id]

            if task.status != "pending":
                continue

            # Check dependencies
            if not self._dependencies_met(task):
                # Re-queue for later
                self.task_queue.put((-task.priority, task_id))
                continue

            # Find suitable agent
            agent_id = self._find_suitable_agent(task)

            if agent_id:
                # Assign task
                self._assign_task(task, agent_id)

                delegations.append({
                    "task_id": task_id,
                    "assigned_to": agent_id,
                    "skills_matched": len(set(task.required_skills) &
                                        set(self.agents[agent_id]["skills"]))
                })

                print(f"  Delegated {task_id} to {agent_id}")
            else:
                # No suitable agent available, re-queue
                self.task_queue.put((-task.priority, task_id))
                break

        return delegations

    def _dependencies_met(self, task: Task) -> bool:
        """Check if task dependencies are met."""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != "completed":
                    return False
        return True

    def _find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find the most suitable available agent for a task."""
        suitable_agents = []

        for agent_id, agent_info in self.agents.items():
            # Check capacity
            if agent_info["current_load"] >= agent_info["capacity"]:
                continue

            # Check skills
            skills_match = len(set(task.required_skills) & set(agent_info["skills"]))

            if skills_match > 0 or not task.required_skills:
                suitable_agents.append((agent_id, skills_match, agent_info["current_load"]))

        if not suitable_agents:
            return None

        # Sort by skills match (desc) and current load (asc)
        suitable_agents.sort(key=lambda x: (-x[1], x[2]))

        return suitable_agents[0][0]

    def _assign_task(self, task: Task, agent_id: str):
        """Assign a task to an agent."""
        task.status = "assigned"
        task.assigned_to = agent_id

        agent = self.agents[agent_id]
        agent["current_load"] += task.estimated_effort
        agent["assigned_tasks"].append(task.task_id)

    def complete_task(self, task_id: str, result: Any = None):
        """Mark a task as completed."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.status = "completed"
        task.result = result

        # Update agent load
        if task.assigned_to:
            agent = self.agents[task.assigned_to]
            agent["current_load"] = max(0, agent["current_load"] - task.estimated_effort)
            agent["assigned_tasks"].remove(task_id)

        # Move to completed
        self.completed_tasks.append(task_id)

        print(f"Task {task_id} completed by {task.assigned_to}")

        # Check and queue dependent tasks
        for dependent_id in self.task_dependencies[task_id]:
            if dependent_id in self.tasks:
                dependent_task = self.tasks[dependent_id]
                if dependent_task.status == "pending":
                    self.task_queue.put((-dependent_task.priority, dependent_id))

    def get_status(self) -> Dict:
        """Get coordination status."""
        return {
            "total_tasks": len(self.tasks),
            "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
            "assigned": sum(1 for t in self.tasks.values() if t.status == "assigned"),
            "completed": len(self.completed_tasks),
            "agents": {
                agent_id: {
                    "load": f"{info['current_load']}/{info['capacity']}",
                    "tasks": len(info["assigned_tasks"])
                }
                for agent_id, info in self.agents.items()
            }
        }


# Example usage
coordinator = TaskCoordinator()

# Register agents with different skills
coordinator.register_agent("dev1", ["python", "testing"], capacity=2)
coordinator.register_agent("dev2", ["python", "database"], capacity=3)
coordinator.register_agent("analyst1", ["data_analysis", "visualization"], capacity=2)

# Create tasks with dependencies
tasks = [
    Task("task1", "Setup database", priority=1, required_skills=["database"]),
    Task("task2", "Implement API", priority=2, required_skills=["python"],
         dependencies=["task1"]),
    Task("task3", "Write tests", priority=3, required_skills=["testing"],
         dependencies=["task2"]),
    Task("task4", "Analyze data", priority=2, required_skills=["data_analysis"]),
    Task("task5", "Create visualizations", priority=3,
         required_skills=["visualization"], dependencies=["task4"])
]

# Submit tasks
for task in tasks:
    coordinator.submit_task(task)

# Delegate tasks
print("\nInitial delegation:")
delegations = coordinator.delegate_tasks()

print(f"\nCoordination status: {coordinator.get_status()}")

# Simulate task completion
coordinator.complete_task("task1", "Database setup complete")
print("\nAfter completing task1:")
delegations = coordinator.delegate_tasks()  # This should now delegate task2

print(f"\nFinal status: {coordinator.get_status()}")


# Example 6: Emergent Behavior Patterns
print("\n" + "=" * 50)
print("Example 6: Emergent Behavior Patterns")
print("=" * 50)


class SwarmAgent:
    """Agent that exhibits swarm behavior."""

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.velocity = np.random.randn(2) * 0.1
        self.neighbors = []
        self.goal = None
        self.trail = [self.position.copy()]

    def update_neighbors(self, all_agents: List['SwarmAgent'], radius: float = 5.0):
        """Update list of neighboring agents."""
        self.neighbors = []

        for agent in all_agents:
            if agent != self:
                distance = np.linalg.norm(self.position - agent.position)
                if distance < radius:
                    self.neighbors.append(agent)

    def swarm_behavior(self,
                      separation_weight: float = 1.0,
                      alignment_weight: float = 1.0,
                      cohesion_weight: float = 1.0):
        """Update position based on swarm rules."""

        if not self.neighbors:
            # Random walk if no neighbors
            self.velocity += np.random.randn(2) * 0.1
        else:
            # Separation: avoid crowding neighbors
            separation = self._separation_vector()

            # Alignment: steer towards average heading of neighbors
            alignment = self._alignment_vector()

            # Cohesion: steer towards average position of neighbors
            cohesion = self._cohesion_vector()

            # Combine behaviors
            self.velocity += (separation * separation_weight +
                            alignment * alignment_weight +
                            cohesion * cohesion_weight)

        # Goal seeking
        if self.goal is not None:
            goal_vector = self.goal - self.position
            self.velocity += goal_vector * 0.1

        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > 1.0:
            self.velocity = self.velocity / speed

        # Update position
        self.position += self.velocity
        self.trail.append(self.position.copy())

    def _separation_vector(self) -> np.ndarray:
        """Calculate separation vector (avoid crowding)."""
        if not self.neighbors:
            return np.zeros(2)

        separation = np.zeros(2)

        for neighbor in self.neighbors:
            diff = self.position - neighbor.position
            distance = np.linalg.norm(diff)

            if distance > 0:
                # Inverse distance weighting
                separation += diff / (distance ** 2)

        return separation / len(self.neighbors)

    def _alignment_vector(self) -> np.ndarray:
        """Calculate alignment vector (match neighbor velocities)."""
        if not self.neighbors:
            return np.zeros(2)

        avg_velocity = np.mean([n.velocity for n in self.neighbors], axis=0)

        return avg_velocity - self.velocity

    def _cohesion_vector(self) -> np.ndarray:
        """Calculate cohesion vector (move towards center of neighbors)."""
        if not self.neighbors:
            return np.zeros(2)

        center = np.mean([n.position for n in self.neighbors], axis=0)

        return (center - self.position) * 0.01


class SwarmSystem:
    """System demonstrating emergent swarm behavior."""

    def __init__(self, num_agents: int = 20):
        self.agents = []
        self.time_step = 0

        # Create agents with random initial positions
        for i in range(num_agents):
            position = np.random.randn(2) * 10
            agent = SwarmAgent(f"swarm_{i}", position)
            self.agents.append(agent)

        print(f"Created swarm with {num_agents} agents")

    def simulate_step(self):
        """Simulate one time step of swarm behavior."""
        # Update neighbors for all agents
        for agent in self.agents:
            agent.update_neighbors(self.agents)

        # Update positions based on swarm behavior
        for agent in self.agents:
            agent.swarm_behavior()

        self.time_step += 1

    def set_goal(self, goal: Tuple[float, float]):
        """Set a common goal for the swarm."""
        for agent in self.agents:
            agent.goal = np.array(goal)

        print(f"Set swarm goal: {goal}")

    def get_swarm_metrics(self) -> Dict:
        """Calculate metrics about swarm behavior."""
        positions = np.array([agent.position for agent in self.agents])

        # Center of mass
        center = np.mean(positions, axis=0)

        # Spread (standard deviation)
        spread = np.std(positions, axis=0)

        # Average distance between agents
        distances = []
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                dist = np.linalg.norm(agent1.position - agent2.position)
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0

        # Velocity alignment
        velocities = np.array([agent.velocity for agent in self.agents])
        avg_velocity = np.mean(velocities, axis=0)
        velocity_alignment = np.mean([np.dot(v, avg_velocity) /
                                     (np.linalg.norm(v) * np.linalg.norm(avg_velocity) + 1e-10)
                                     for v in velocities])

        return {
            "center": center.tolist(),
            "spread": spread.tolist(),
            "avg_distance": avg_distance,
            "velocity_alignment": velocity_alignment,
            "time_step": self.time_step
        }

    def detect_patterns(self) -> List[str]:
        """Detect emergent patterns in swarm behavior."""
        patterns = []
        metrics = self.get_swarm_metrics()

        # Check for flocking (high alignment)
        if metrics["velocity_alignment"] > 0.7:
            patterns.append("flocking")

        # Check for clustering (low spread)
        if np.mean(metrics["spread"]) < 3.0:
            patterns.append("clustering")

        # Check for dispersal (high spread)
        if np.mean(metrics["spread"]) > 10.0:
            patterns.append("dispersal")

        # Check for circling (if velocities perpendicular to center)
        positions = np.array([agent.position for agent in self.agents])
        center = np.mean(positions, axis=0)

        perpendicular_count = 0
        for agent in self.agents:
            to_center = center - agent.position
            if np.linalg.norm(to_center) > 0:
                to_center_normalized = to_center / np.linalg.norm(to_center)
                vel_normalized = agent.velocity / (np.linalg.norm(agent.velocity) + 1e-10)

                if abs(np.dot(to_center_normalized, vel_normalized)) < 0.3:
                    perpendicular_count += 1

        if perpendicular_count > len(self.agents) * 0.6:
            patterns.append("circling")

        return patterns


# Example usage
swarm = SwarmSystem(num_agents=15)

# Simulate swarm behavior
print("\nSimulating swarm behavior:")
for step in range(10):
    swarm.simulate_step()

    if step % 3 == 0:
        metrics = swarm.get_swarm_metrics()
        patterns = swarm.detect_patterns()

        print(f"  Step {step}: patterns={patterns}, "
              f"alignment={metrics['velocity_alignment']:.2f}")

# Set a goal and observe convergence
swarm.set_goal((10, 10))
print("\nSwarm with goal:")
for step in range(10):
    swarm.simulate_step()

    if step % 3 == 0:
        metrics = swarm.get_swarm_metrics()
        distance_to_goal = np.linalg.norm(metrics["center"] - np.array([10, 10]))

        print(f"  Step {swarm.time_step}: distance to goal={distance_to_goal:.2f}")


# Example 7: Production Multi-Agent Orchestration
print("\n" + "=" * 50)
print("Example 7: Production Multi-Agent Orchestration")
print("=" * 50)


class OrchestratedAgent:
    """Production-ready orchestrated agent."""

    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = "idle"
        self.health = 100
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_response_time": 0,
            "uptime": 0
        }
        self.message_buffer = deque(maxlen=100)

    async def process_task(self, task: Dict) -> Dict:
        """Process a task asynchronously."""
        self.status = "processing"
        start_time = time.time()

        try:
            # Simulate task processing
            await asyncio.sleep(np.random.uniform(0.5, 2.0))

            # Simulate occasional failures
            if np.random.random() > 0.9:
                raise Exception("Task processing failed")

            result = {
                "status": "success",
                "result": f"Processed by {self.agent_id}",
                "processing_time": time.time() - start_time
            }

            self.performance_metrics["tasks_completed"] += 1

        except Exception as e:
            result = {
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

            self.performance_metrics["tasks_failed"] += 1
            self.health -= 10

        finally:
            self.status = "idle"

            # Update average response time
            total_tasks = (self.performance_metrics["tasks_completed"] +
                          self.performance_metrics["tasks_failed"])

            self.performance_metrics["avg_response_time"] = (
                (self.performance_metrics["avg_response_time"] * (total_tasks - 1) +
                 result["processing_time"]) / total_tasks
            )

        return result

    def get_health_status(self) -> Dict:
        """Get agent health status."""
        health_level = "healthy" if self.health > 70 else "degraded" if self.health > 30 else "critical"

        return {
            "agent_id": self.agent_id,
            "health": self.health,
            "health_level": health_level,
            "status": self.status,
            "metrics": self.performance_metrics
        }


class ProductionOrchestrator:
    """Production-ready multi-agent orchestrator."""

    def __init__(self):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks = []
        self.failed_tasks = []
        self.orchestration_rules = []
        self.monitoring_data = defaultdict(list)

    def register_agent(self, agent: OrchestratedAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        print(f"Registered agent: {agent.agent_id} (capabilities: {agent.capabilities})")

    async def submit_task(self, task: Dict):
        """Submit a task to the orchestrator."""
        task["submitted_at"] = datetime.now()
        task["task_id"] = f"task_{len(self.completed_tasks) + len(self.failed_tasks)}"

        await self.task_queue.put(task)
        print(f"Submitted task: {task['task_id']}")

    async def orchestrate(self):
        """Main orchestration loop."""
        print("Starting orchestration...")

        while True:
            try:
                # Get next task (with timeout)
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Select agent
                agent = self._select_agent(task)

                if not agent:
                    print(f"No suitable agent for task {task['task_id']}")
                    self.failed_tasks.append(task)
                    continue

                # Process task
                print(f"Assigning {task['task_id']} to {agent.agent_id}")
                result = await agent.process_task(task)

                # Record result
                task["result"] = result
                task["processed_by"] = agent.agent_id
                task["completed_at"] = datetime.now()

                if result["status"] == "success":
                    self.completed_tasks.append(task)
                else:
                    self.failed_tasks.append(task)

                    # Retry logic
                    if task.get("retry_count", 0) < 3:
                        task["retry_count"] = task.get("retry_count", 0) + 1
                        await self.task_queue.put(task)

                # Monitor performance
                self._update_monitoring(agent.agent_id, result)

            except asyncio.TimeoutError:
                # No tasks available
                await self._health_check()

            except Exception as e:
                print(f"Orchestration error: {e}")

    def _select_agent(self, task: Dict) -> Optional[OrchestratedAgent]:
        """Select the best agent for a task."""
        required_capability = task.get("required_capability")

        # Filter capable and healthy agents
        suitable_agents = []

        for agent in self.agents.values():
            if agent.status == "idle" and agent.health > 30:
                if not required_capability or required_capability in agent.capabilities:
                    suitable_agents.append(agent)

        if not suitable_agents:
            return None

        # Select agent with best performance
        best_agent = min(suitable_agents,
                        key=lambda a: a.performance_metrics["avg_response_time"]
                        if a.performance_metrics["avg_response_time"] > 0 else float('inf'))

        return best_agent

    async def _health_check(self):
        """Perform health check on all agents."""
        for agent in self.agents.values():
            # Restore health gradually
            if agent.health < 100:
                agent.health = min(100, agent.health + 5)

    def _update_monitoring(self, agent_id: str, result: Dict):
        """Update monitoring data."""
        self.monitoring_data[agent_id].append({
            "timestamp": datetime.now(),
            "status": result["status"],
            "processing_time": result.get("processing_time", 0)
        })

        # Keep only recent data
        if len(self.monitoring_data[agent_id]) > 100:
            self.monitoring_data[agent_id] = self.monitoring_data[agent_id][-100:]

    def get_orchestration_status(self) -> Dict:
        """Get comprehensive orchestration status."""
        total_agents = len(self.agents)
        healthy_agents = sum(1 for a in self.agents.values() if a.health > 70)

        return {
            "agents": {
                "total": total_agents,
                "healthy": healthy_agents,
                "degraded": sum(1 for a in self.agents.values()
                              if 30 < a.health <= 70),
                "critical": sum(1 for a in self.agents.values() if a.health <= 30)
            },
            "tasks": {
                "queued": self.task_queue.qsize(),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "success_rate": len(self.completed_tasks) /
                              max(len(self.completed_tasks) + len(self.failed_tasks), 1)
            },
            "performance": {
                agent_id: {
                    "avg_response_time": agent.performance_metrics["avg_response_time"],
                    "success_rate": agent.performance_metrics["tasks_completed"] /
                                  max(agent.performance_metrics["tasks_completed"] +
                                      agent.performance_metrics["tasks_failed"], 1)
                }
                for agent_id, agent in self.agents.items()
            }
        }


# Example usage
async def production_demo():
    orchestrator = ProductionOrchestrator()

    # Register agents
    orchestrator.register_agent(
        OrchestratedAgent("worker1", ["data_processing", "analysis"])
    )
    orchestrator.register_agent(
        OrchestratedAgent("worker2", ["data_processing", "validation"])
    )
    orchestrator.register_agent(
        OrchestratedAgent("worker3", ["analysis", "reporting"])
    )

    # Start orchestration in background
    orchestration_task = asyncio.create_task(orchestrator.orchestrate())

    # Submit tasks
    tasks = [
        {"type": "process", "required_capability": "data_processing"},
        {"type": "analyze", "required_capability": "analysis"},
        {"type": "validate", "required_capability": "validation"},
        {"type": "report", "required_capability": "reporting"},
        {"type": "general"}  # Any agent can handle
    ]

    for task in tasks:
        await orchestrator.submit_task(task)

    # Let tasks process
    await asyncio.sleep(5)

    # Get status
    status = orchestrator.get_orchestration_status()
    print(f"\nOrchestration Status:")
    print(f"  Agents: {status['agents']}")
    print(f"  Tasks: {status['tasks']}")
    print(f"  Success rate: {status['tasks']['success_rate']:.2%}")

    # Cancel orchestration
    orchestration_task.cancel()


# Run production demo
print("\nRunning production orchestration demo...")
asyncio.run(production_demo())

print("\n" + "=" * 50)
print("All Multi-Agent System Examples Complete!")
print("=" * 50)