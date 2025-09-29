"""
Module 13: Agent Design
Production Project - Comprehensive Agent Platform

A production-ready platform for deploying and managing autonomous agents
with memory, planning, collaboration, and monitoring capabilities.
"""

import os
import asyncio
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///agent_platform.db"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app
app = FastAPI(title="Agent Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for distributed operations
redis_client = None


# Database Models
class AgentRecord(Base):
    __tablename__ = "agents"

    agent_id = Column(String, primary_key=True)
    agent_type = Column(String)
    capabilities = Column(JSON)
    status = Column(String)
    health = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    performance_metrics = Column(JSON)


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True)
    description = Column(Text)
    priority = Column(Integer)
    status = Column(String)
    assigned_to = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON)


class MemoryRecord(Base):
    __tablename__ = "memories"

    memory_id = Column(String, primary_key=True)
    agent_id = Column(String)
    content = Column(Text)
    memory_type = Column(String)
    importance = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class CollaborationRecord(Base):
    __tablename__ = "collaborations"

    collaboration_id = Column(String, primary_key=True)
    participating_agents = Column(JSON)
    task_id = Column(String)
    consensus_reached = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


# Pydantic Models
class AgentConfig(BaseModel):
    agent_type: str = Field(default="general")
    capabilities: List[str] = Field(default_factory=list)
    memory_capacity: int = Field(default=1000)
    learning_rate: float = Field(default=0.1)


class TaskRequest(BaseModel):
    description: str
    priority: int = Field(default=5, ge=1, le=10)
    required_capabilities: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    context: Dict = Field(default_factory=dict)


class CollaborationRequest(BaseModel):
    task_id: str
    agent_ids: List[str]
    consensus_type: str = Field(default="majority")


# Core Agent Components
class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COLLABORATING = "collaborating"
    LEARNING = "learning"
    ERROR = "error"


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


# Memory System
class IntegratedMemorySystem:
    """Integrated memory system for agents."""

    def __init__(self, agent_id: str, capacity: int = 1000):
        self.agent_id = agent_id
        self.capacity = capacity
        self.working_memory = deque(maxlen=7)
        self.episodic_memory = deque(maxlen=capacity)
        self.semantic_memory = {}
        self.procedural_memory = {}
        self.importance_decay = 0.99

    async def store(self, content: Any, memory_type: MemoryType, importance: float = 1.0):
        """Store memory in appropriate system."""
        memory_id = str(uuid.uuid4())

        memory = {
            "id": memory_id,
            "content": content,
            "type": memory_type.value,
            "importance": importance,
            "timestamp": datetime.now()
        }

        if memory_type == MemoryType.WORKING:
            self.working_memory.append(memory)
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(memory)
        elif memory_type == MemoryType.SEMANTIC:
            key = hashlib.md5(str(content).encode()).hexdigest()[:12]
            self.semantic_memory[key] = memory
        elif memory_type == MemoryType.PROCEDURAL:
            if isinstance(content, dict) and "procedure" in content:
                self.procedural_memory[content["procedure"]] = memory

        # Persist to database
        await self._persist_memory(memory)

        logger.info(f"Agent {self.agent_id} stored {memory_type.value} memory: {memory_id}")

        return memory_id

    async def retrieve(self, query: str, memory_types: List[MemoryType] = None, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories."""
        if not memory_types:
            memory_types = list(MemoryType)

        results = []

        for memory_type in memory_types:
            if memory_type == MemoryType.WORKING:
                results.extend(list(self.working_memory))
            elif memory_type == MemoryType.EPISODIC:
                # Simple relevance search
                relevant = [m for m in self.episodic_memory
                          if query.lower() in str(m["content"]).lower()]
                results.extend(relevant[:k])
            elif memory_type == MemoryType.SEMANTIC:
                # Search semantic memory
                relevant = [m for key, m in self.semantic_memory.items()
                          if query.lower() in str(m["content"]).lower()]
                results.extend(relevant[:k])
            elif memory_type == MemoryType.PROCEDURAL:
                # Search procedures
                relevant = [m for proc, m in self.procedural_memory.items()
                          if query.lower() in proc.lower()]
                results.extend(relevant[:k])

        # Sort by importance and recency
        results.sort(key=lambda x: x["importance"] * (1 / (1 + (datetime.now() - x["timestamp"]).seconds)), reverse=True)

        return results[:k]

    async def consolidate(self):
        """Consolidate memories for efficiency."""
        # Apply importance decay
        for memory in self.episodic_memory:
            memory["importance"] *= self.importance_decay

        # Remove low importance memories
        self.episodic_memory = deque(
            [m for m in self.episodic_memory if m["importance"] > 0.1],
            maxlen=self.capacity
        )

        logger.info(f"Agent {self.agent_id} consolidated memories")

    async def _persist_memory(self, memory: Dict):
        """Persist memory to database."""
        db = SessionLocal()
        try:
            db_memory = MemoryRecord(
                memory_id=memory["id"],
                agent_id=self.agent_id,
                content=json.dumps(memory["content"]),
                memory_type=memory["type"],
                importance=memory["importance"],
                timestamp=memory["timestamp"]
            )
            db.add(db_memory)
            db.commit()
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
        finally:
            db.close()


# Planning System
class AdaptivePlanner:
    """Adaptive planning system for agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.plan_templates = self._load_templates()
        self.plan_history = deque(maxlen=100)

    def _load_templates(self) -> Dict:
        """Load planning templates."""
        return {
            "analysis": {
                "steps": ["gather_data", "process", "analyze", "report"],
                "preconditions": ["data_available"]
            },
            "execution": {
                "steps": ["prepare", "execute", "verify", "cleanup"],
                "preconditions": ["resources_ready"]
            },
            "collaboration": {
                "steps": ["coordinate", "share", "integrate", "consensus"],
                "preconditions": ["agents_available"]
            }
        }

    async def create_plan(self, task: Dict) -> Dict:
        """Create execution plan for task."""
        task_type = task.get("type", "general")
        template = self.plan_templates.get(task_type, self.plan_templates["execution"])

        plan = {
            "plan_id": str(uuid.uuid4()),
            "task_id": task.get("id"),
            "steps": template["steps"],
            "current_step": 0,
            "status": "pending",
            "created_at": datetime.now()
        }

        self.plan_history.append(plan)

        logger.info(f"Agent {self.agent_id} created plan: {plan['plan_id']}")

        return plan

    async def execute_step(self, plan: Dict) -> Tuple[bool, Any]:
        """Execute current step of plan."""
        if plan["current_step"] >= len(plan["steps"]):
            return True, "Plan completed"

        step = plan["steps"][plan["current_step"]]

        # Simulate step execution
        success = np.random.random() > 0.1  # 90% success rate

        if success:
            plan["current_step"] += 1
            result = f"Completed: {step}"
        else:
            result = f"Failed: {step}"

        logger.info(f"Agent {self.agent_id} executed step: {step} - {result}")

        return success, result

    async def replan(self, failed_plan: Dict) -> Dict:
        """Create alternative plan after failure."""
        # Add retry steps
        new_steps = []
        for step in failed_plan["steps"]:
            new_steps.append(f"retry_{step}")
            new_steps.append(f"verify_{step}")

        new_plan = {
            "plan_id": str(uuid.uuid4()),
            "task_id": failed_plan["task_id"],
            "steps": new_steps,
            "current_step": 0,
            "status": "revised",
            "original_plan": failed_plan["plan_id"],
            "created_at": datetime.now()
        }

        logger.info(f"Agent {self.agent_id} created revised plan: {new_plan['plan_id']}")

        return new_plan


# Autonomous Agent
class AutonomousAgent:
    """Production-ready autonomous agent."""

    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.IDLE
        self.health = 100
        self.memory = IntegratedMemorySystem(agent_id, config.memory_capacity)
        self.planner = AdaptivePlanner(agent_id)
        self.current_task = None
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_time": 0,
            "collaboration_count": 0
        }

    async def process_task(self, task: Dict) -> Dict:
        """Process a task autonomously."""
        self.status = AgentStatus.THINKING
        self.current_task = task
        start_time = time.time()

        logger.info(f"Agent {self.agent_id} processing task: {task.get('id')}")

        try:
            # Store task in episodic memory
            await self.memory.store(task, MemoryType.EPISODIC, importance=0.8)

            # Retrieve relevant memories
            relevant_memories = await self.memory.retrieve(task.get("description", ""), k=5)

            # Create plan
            plan = await self.planner.create_plan(task)

            # Execute plan
            self.status = AgentStatus.ACTING
            execution_results = []

            while plan["current_step"] < len(plan["steps"]):
                success, result = await self.planner.execute_step(plan)
                execution_results.append(result)

                if not success:
                    # Try replanning
                    plan = await self.planner.replan(plan)

            # Learn from experience
            self.status = AgentStatus.LEARNING
            await self._learn_from_task(task, execution_results, success=True)

            # Update metrics
            task_time = time.time() - start_time
            self._update_metrics(success=True, task_time=task_time)

            result = {
                "task_id": task.get("id"),
                "agent_id": self.agent_id,
                "status": "completed",
                "execution_time": task_time,
                "steps_completed": len(execution_results),
                "result": execution_results
            }

        except Exception as e:
            logger.error(f"Agent {self.agent_id} error processing task: {e}")

            # Learn from failure
            await self._learn_from_task(task, str(e), success=False)

            # Update metrics
            task_time = time.time() - start_time
            self._update_metrics(success=False, task_time=task_time)

            result = {
                "task_id": task.get("id"),
                "agent_id": self.agent_id,
                "status": "failed",
                "error": str(e),
                "execution_time": task_time
            }

        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None

        return result

    async def collaborate(self, task: Dict, collaborators: List['AutonomousAgent']) -> Dict:
        """Collaborate with other agents on task."""
        self.status = AgentStatus.COLLABORATING

        logger.info(f"Agent {self.agent_id} collaborating with {len(collaborators)} agents")

        # Share knowledge
        shared_knowledge = await self.memory.retrieve(task.get("description", ""), k=3)

        # Propose solution
        proposal = {
            "agent_id": self.agent_id,
            "approach": f"Approach by {self.agent_id}",
            "confidence": np.random.random(),
            "knowledge": shared_knowledge
        }

        # Collect proposals from collaborators
        all_proposals = [proposal]
        for collaborator in collaborators:
            if collaborator.agent_id != self.agent_id:
                other_proposal = await collaborator.propose_solution(task)
                all_proposals.append(other_proposal)

        # Reach consensus
        consensus = self._reach_consensus(all_proposals)

        # Update collaboration metrics
        self.performance_metrics["collaboration_count"] += 1

        return {
            "task_id": task.get("id"),
            "collaborators": [a.agent_id for a in [self] + collaborators],
            "consensus": consensus,
            "proposals": all_proposals
        }

    async def propose_solution(self, task: Dict) -> Dict:
        """Propose solution for collaborative task."""
        # Retrieve relevant knowledge
        knowledge = await self.memory.retrieve(task.get("description", ""), k=3)

        proposal = {
            "agent_id": self.agent_id,
            "approach": f"Solution from {self.agent_id}",
            "confidence": 0.5 + np.random.random() * 0.5,
            "knowledge": knowledge
        }

        return proposal

    def _reach_consensus(self, proposals: List[Dict]) -> Dict:
        """Reach consensus from multiple proposals."""
        # Simple voting based on confidence
        best_proposal = max(proposals, key=lambda p: p["confidence"])

        consensus = {
            "selected_proposal": best_proposal["agent_id"],
            "confidence": best_proposal["confidence"],
            "vote_distribution": {p["agent_id"]: p["confidence"] for p in proposals}
        }

        return consensus

    async def _learn_from_task(self, task: Dict, result: Any, success: bool):
        """Learn from task execution."""
        # Create learning entry
        learning = {
            "task": task.get("description", ""),
            "result": str(result)[:200],
            "success": success,
            "timestamp": datetime.now()
        }

        # Store as semantic memory for future reference
        if success:
            await self.memory.store(learning, MemoryType.SEMANTIC, importance=0.9)
        else:
            await self.memory.store(learning, MemoryType.EPISODIC, importance=0.6)

        # Consolidate memories periodically
        if np.random.random() < 0.1:  # 10% chance
            await self.memory.consolidate()

    def _update_metrics(self, success: bool, task_time: float):
        """Update performance metrics."""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1

        # Update average task time
        total_tasks = (self.performance_metrics["tasks_completed"] +
                      self.performance_metrics["tasks_failed"])

        if total_tasks > 0:
            old_avg = self.performance_metrics["avg_task_time"]
            self.performance_metrics["avg_task_time"] = (
                (old_avg * (total_tasks - 1) + task_time) / total_tasks
            )

    def get_status(self) -> Dict:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "health": self.health,
            "current_task": self.current_task.get("id") if self.current_task else None,
            "performance": self.performance_metrics
        }


# Agent Manager
class AgentManager:
    """Manages multiple agents in the platform."""

    def __init__(self):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new agent."""
        agent_id = str(uuid.uuid4())
        agent = AutonomousAgent(agent_id, config)
        self.agents[agent_id] = agent

        # Persist to database
        db = SessionLocal()
        try:
            db_agent = AgentRecord(
                agent_id=agent_id,
                agent_type=config.agent_type,
                capabilities=config.capabilities,
                status=AgentStatus.IDLE.value,
                health=100,
                performance_metrics={}
            )
            db.add(db_agent)
            db.commit()
        finally:
            db.close()

        logger.info(f"Created agent: {agent_id}")

        return agent_id

    async def assign_task(self, task: TaskRequest) -> str:
        """Assign task to appropriate agent."""
        task_id = str(uuid.uuid4())

        task_dict = {
            "id": task_id,
            "description": task.description,
            "priority": task.priority,
            "required_capabilities": task.required_capabilities,
            "context": task.context
        }

        # Find suitable agent
        suitable_agent = self._find_suitable_agent(task.required_capabilities)

        if not suitable_agent:
            raise HTTPException(status_code=404, detail="No suitable agent available")

        # Assign task
        result = await suitable_agent.process_task(task_dict)

        # Persist task
        db = SessionLocal()
        try:
            db_task = TaskRecord(
                task_id=task_id,
                description=task.description,
                priority=task.priority,
                status=result["status"],
                assigned_to=suitable_agent.agent_id,
                result=result
            )
            db.add(db_task)
            db.commit()
        finally:
            db.close()

        return task_id

    def _find_suitable_agent(self, required_capabilities: List[str]) -> Optional[AutonomousAgent]:
        """Find agent with required capabilities."""
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE:
                # Check capabilities
                if not required_capabilities:
                    return agent

                agent_capabilities = set(agent.config.capabilities)
                if set(required_capabilities).issubset(agent_capabilities):
                    return agent

        return None

    async def coordinate_collaboration(self, request: CollaborationRequest) -> Dict:
        """Coordinate multi-agent collaboration."""
        # Get agents
        collaborators = []
        for agent_id in request.agent_ids:
            if agent_id in self.agents:
                collaborators.append(self.agents[agent_id])

        if len(collaborators) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 agents for collaboration")

        # Get task
        db = SessionLocal()
        try:
            db_task = db.query(TaskRecord).filter(TaskRecord.task_id == request.task_id).first()
            if not db_task:
                raise HTTPException(status_code=404, detail="Task not found")

            task = {
                "id": db_task.task_id,
                "description": db_task.description
            }
        finally:
            db.close()

        # Coordinate collaboration
        lead_agent = collaborators[0]
        result = await lead_agent.collaborate(task, collaborators[1:])

        # Persist collaboration
        db = SessionLocal()
        try:
            db_collab = CollaborationRecord(
                collaboration_id=str(uuid.uuid4()),
                participating_agents=request.agent_ids,
                task_id=request.task_id,
                consensus_reached=result["consensus"]
            )
            db.add(db_collab)
            db.commit()
        finally:
            db.close()

        return result

    def get_platform_status(self) -> Dict:
        """Get overall platform status."""
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values()
                               if a.status != AgentStatus.IDLE),
            "agent_statuses": {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            }
        }


# Initialize components
agent_manager = AgentManager()


# API Endpoints
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


@app.get("/")
async def root():
    return {
        "name": "Agent Platform",
        "version": "1.0.0",
        "endpoints": [
            "/agents",
            "/tasks",
            "/collaborate",
            "/status",
            "/ws/monitor"
        ]
    }


@app.post("/agents")
async def create_agent(config: AgentConfig):
    """Create a new autonomous agent."""
    agent_id = await agent_manager.create_agent(config)
    return {"agent_id": agent_id, "status": "created"}


@app.get("/agents")
async def list_agents():
    """List all agents."""
    db = SessionLocal()
    try:
        agents = db.query(AgentRecord).all()
        return [
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "health": agent.health,
                "created_at": agent.created_at
            }
            for agent in agents
        ]
    finally:
        db.close()


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    if agent_id not in agent_manager.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agent_manager.agents[agent_id]
    return agent.get_status()


@app.post("/tasks")
async def create_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """Create and assign a new task."""
    task_id = await agent_manager.assign_task(task)
    return {"task_id": task_id, "status": "assigned"}


@app.get("/tasks")
async def list_tasks():
    """List all tasks."""
    db = SessionLocal()
    try:
        tasks = db.query(TaskRecord).order_by(TaskRecord.created_at.desc()).limit(100).all()
        return [
            {
                "task_id": task.task_id,
                "description": task.description[:100],
                "status": task.status,
                "assigned_to": task.assigned_to,
                "created_at": task.created_at
            }
            for task in tasks
        ]
    finally:
        db.close()


@app.post("/collaborate")
async def collaborate(request: CollaborationRequest):
    """Initiate multi-agent collaboration."""
    result = await agent_manager.coordinate_collaboration(request)
    return result


@app.get("/status")
async def get_status():
    """Get platform status."""
    return agent_manager.get_platform_status()


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket for real-time monitoring."""
    await websocket.accept()

    try:
        while True:
            # Send status updates every 2 seconds
            status = agent_manager.get_platform_status()
            await websocket.send_json(status)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    checks = {
        "database": False,
        "redis": False,
        "agents": len(agent_manager.agents) > 0
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