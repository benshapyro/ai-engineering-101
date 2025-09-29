# Module 13: Agent Design

## Learning Objectives
By the end of this module, you will:
- Understand autonomous agent architectures and patterns
- Build agents that can plan, execute, and adapt
- Implement memory systems for persistent agent state
- Master multi-agent collaboration and orchestration
- Deploy production-ready agent systems

## Key Concepts

### 1. Agent Architecture Fundamentals

```python
class Agent:
    """Core components of an autonomous agent."""

    def __init__(self):
        self.perception = PerceptionModule()    # Understand environment
        self.planning = PlanningModule()        # Create action plans
        self.memory = MemoryModule()            # Store experiences
        self.tools = ToolRegistry()             # Available actions
        self.executor = ExecutionModule()       # Perform actions
        self.reflection = ReflectionModule()    # Learn from outcomes

    def run(self, objective):
        """Main agent loop."""
        while not self.is_complete(objective):
            # Perceive current state
            state = self.perception.observe()

            # Retrieve relevant memories
            context = self.memory.recall(state, objective)

            # Plan next actions
            plan = self.planning.create_plan(state, objective, context)

            # Execute plan
            results = self.executor.execute(plan)

            # Reflect and learn
            insights = self.reflection.analyze(results, objective)

            # Update memory
            self.memory.store(results, insights)

        return self.get_final_output()
```

### 2. Planning Systems

#### Goal Decomposition
```python
class GoalPlanner:
    def decompose_goal(self, high_level_goal):
        """Break down complex goals into subtasks."""
        prompt = f"""Break down this goal into concrete, actionable subtasks:

Goal: {high_level_goal}

Subtasks (ordered by priority and dependencies):"""

        subtasks = self.llm.generate(prompt)
        return self.parse_subtasks(subtasks)

    def create_execution_plan(self, subtasks):
        """Create detailed execution plan."""
        plan = []

        for task in subtasks:
            step = {
                'task': task,
                'preconditions': self.identify_preconditions(task),
                'actions': self.determine_actions(task),
                'success_criteria': self.define_success(task),
                'fallback': self.plan_fallback(task)
            }
            plan.append(step)

        return self.optimize_plan(plan)
```

#### Dynamic Replanning
```python
class AdaptivePlanner:
    def __init__(self):
        self.original_plan = None
        self.current_plan = None
        self.execution_history = []

    def replan_if_needed(self, current_state, objective):
        """Adjust plan based on execution results."""
        # Check if replanning needed
        if self.should_replan(current_state):
            # Analyze what went wrong
            failure_analysis = self.analyze_deviation(
                self.current_plan,
                self.execution_history
            )

            # Generate new plan
            new_plan = self.generate_alternative_plan(
                objective,
                current_state,
                failure_analysis
            )

            self.current_plan = new_plan
            return new_plan

        return self.current_plan
```

### 3. Memory Systems

#### Working Memory
```python
class WorkingMemory:
    """Short-term memory for current task context."""

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.attention_weights = {}

    def add(self, item, importance=1.0):
        """Add item with importance weighting."""
        self.memory.append(item)
        self.attention_weights[id(item)] = importance

    def get_context(self):
        """Get weighted context for decision making."""
        sorted_items = sorted(
            self.memory,
            key=lambda x: self.attention_weights.get(id(x), 0),
            reverse=True
        )
        return sorted_items[:self.capacity // 2]
```

#### Long-term Memory
```python
class LongTermMemory:
    """Persistent memory with semantic retrieval."""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.episodic_memory = []  # Experiences
        self.semantic_memory = {}   # Facts and knowledge
        self.procedural_memory = {} # How to do things

    def store_experience(self, experience):
        """Store episodic memory."""
        embedding = self.encode_experience(experience)

        memory_entry = {
            'timestamp': datetime.now(),
            'experience': experience,
            'embedding': embedding,
            'success': experience.get('success', False),
            'lessons': self.extract_lessons(experience)
        }

        self.episodic_memory.append(memory_entry)
        self.vector_store.add(embedding, memory_entry)

    def retrieve_relevant(self, query, k=5):
        """Retrieve relevant memories."""
        query_embedding = self.encode_experience(query)
        similar_memories = self.vector_store.search(query_embedding, k)

        # Weight by recency and relevance
        weighted_memories = self.weight_memories(similar_memories)

        return weighted_memories
```

### 4. Tool Use & Action Execution

#### Tool Registry
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = {}
        self.usage_stats = defaultdict(int)

    def register_tool(self, name, function, description):
        """Register a tool for agent use."""
        self.tools[name] = function
        self.tool_descriptions[name] = {
            'description': description,
            'parameters': self.extract_parameters(function),
            'returns': self.extract_returns(function)
        }

    def select_tool(self, task_description):
        """Select appropriate tool for task."""
        prompt = f"""Task: {task_description}

Available tools:
{self.format_tool_descriptions()}

Which tool is best? Return tool name only."""

        tool_name = self.llm.generate(prompt).strip()

        if tool_name in self.tools:
            self.usage_stats[tool_name] += 1
            return self.tools[tool_name]

        return None
```

#### Action Executor
```python
class ActionExecutor:
    def __init__(self, tool_registry):
        self.tools = tool_registry
        self.execution_history = []

    def execute_action(self, action):
        """Execute single action with error handling."""
        try:
            # Select tool
            tool = self.tools.select_tool(action['description'])

            if not tool:
                return {'error': 'No suitable tool found'}

            # Prepare parameters
            params = self.prepare_parameters(action, tool)

            # Execute with timeout
            result = self.execute_with_timeout(tool, params)

            # Record execution
            self.execution_history.append({
                'action': action,
                'result': result,
                'timestamp': datetime.now()
            })

            return result

        except Exception as e:
            return {'error': str(e)}

    def execute_parallel_actions(self, actions):
        """Execute independent actions in parallel."""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.execute_action, action)
                for action in actions
            ]
            results = [f.result() for f in futures]

        return results
```

### 5. Multi-Agent Systems

#### Agent Communication
```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = Queue()
        self.shared_memory = {}

    def register_agent(self, agent_id, agent):
        """Register agent in the system."""
        self.agents[agent_id] = agent
        agent.connect_to_system(self)

    def broadcast_message(self, sender_id, message):
        """Broadcast message to all agents."""
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                agent.receive_message(sender_id, message)

    def coordinate_task(self, task):
        """Coordinate multi-agent task execution."""
        # Assign roles
        role_assignments = self.assign_roles(task)

        # Create shared plan
        shared_plan = self.create_shared_plan(task, role_assignments)

        # Execute with coordination
        results = {}
        for agent_id, subtask in shared_plan.items():
            agent = self.agents[agent_id]
            results[agent_id] = agent.execute_subtask(subtask)

            # Share progress
            self.broadcast_progress(agent_id, results[agent_id])

        return self.aggregate_results(results)
```

#### Specialized Agent Roles
```python
class SpecializedAgents:
    """Different agent specializations."""

    @staticmethod
    def create_researcher():
        """Agent specialized in research and information gathering."""
        return Agent(
            perception=WebSearchPerception(),
            memory=ResearchMemory(),
            tools=['web_search', 'summarize', 'extract_facts']
        )

    @staticmethod
    def create_coder():
        """Agent specialized in code generation and debugging."""
        return Agent(
            perception=CodeAnalysisPerception(),
            memory=CodeMemory(),
            tools=['write_code', 'debug', 'test', 'refactor']
        )

    @staticmethod
    def create_analyst():
        """Agent specialized in data analysis."""
        return Agent(
            perception=DataPerception(),
            memory=AnalyticalMemory(),
            tools=['query_data', 'visualize', 'statistical_test']
        )
```

## Advanced Agent Patterns

### 1. ReAct Pattern
```python
class ReActAgent:
    """Reasoning and Acting agent pattern."""

    def solve(self, problem):
        max_steps = 10
        for step in range(max_steps):
            # Thought: Reasoning about current state
            thought = self.think(problem, self.get_context())

            # Action: Decide and execute action
            action = self.decide_action(thought)
            observation = self.execute(action)

            # Update context
            self.update_context(thought, action, observation)

            # Check if solved
            if self.is_solved(problem, observation):
                return self.format_solution(observation)

        return "Max steps reached without solution"

    def think(self, problem, context):
        prompt = f"""Problem: {problem}
Context: {context}

Thought: Let me think about this step by step..."""
        return self.llm.generate(prompt)
```

### 2. Tree of Thoughts
```python
class TreeOfThoughtsAgent:
    """Explore multiple reasoning paths."""

    def solve_with_tree(self, problem, branches=3, depth=3):
        # Initialize root
        root = ThoughtNode(problem)

        # Build tree
        self.expand_tree(root, branches, depth)

        # Evaluate paths
        best_path = self.find_best_path(root)

        # Execute best path
        return self.execute_path(best_path)

    def expand_tree(self, node, branches, remaining_depth):
        if remaining_depth == 0:
            return

        # Generate multiple thoughts
        thoughts = self.generate_thoughts(node.state, n=branches)

        for thought in thoughts:
            child = ThoughtNode(thought, parent=node)
            node.children.append(child)

            # Recursively expand
            self.expand_tree(child, branches, remaining_depth - 1)
```

### 3. Self-Reflection
```python
class ReflectiveAgent:
    """Agent that learns from self-reflection."""

    def __init__(self):
        self.experience_buffer = []
        self.learned_strategies = {}

    def execute_with_reflection(self, task):
        # Initial attempt
        result = self.attempt_task(task)

        # Self-critique
        critique = self.critique_performance(task, result)

        if critique['success_rate'] < 0.8:
            # Reflect and improve
            improvements = self.reflect_on_failure(task, result, critique)

            # Retry with improvements
            improved_result = self.retry_with_learning(task, improvements)

            # Store learning
            self.store_learning(task, improvements, improved_result)

            return improved_result

        return result

    def reflect_on_failure(self, task, result, critique):
        prompt = f"""Task: {task}
Result: {result}
Critique: {critique}

What went wrong and how can I improve?

Reflection:"""
        return self.llm.generate(prompt)
```

## Production Deployment

### 1. Agent Monitoring
```python
class AgentMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

    def track_agent(self, agent_id, metrics):
        """Track agent performance metrics."""
        self.metrics[agent_id].append({
            'timestamp': datetime.now(),
            'task_completion_rate': metrics['completion_rate'],
            'avg_steps': metrics['avg_steps'],
            'error_rate': metrics['error_rate'],
            'resource_usage': metrics['resource_usage']
        })

        # Check for anomalies
        self.check_anomalies(agent_id, metrics)

    def check_anomalies(self, agent_id, metrics):
        """Detect performance issues."""
        if metrics['error_rate'] > 0.1:
            self.alerts.append({
                'agent_id': agent_id,
                'issue': 'High error rate',
                'severity': 'HIGH'
            })

        if metrics['avg_steps'] > 20:
            self.alerts.append({
                'agent_id': agent_id,
                'issue': 'Excessive steps',
                'severity': 'MEDIUM'
            })
```

### 2. Scalability
```python
class ScalableAgentSystem:
    def __init__(self, max_agents=100):
        self.agent_pool = []
        self.max_agents = max_agents
        self.load_balancer = LoadBalancer()

    async def process_requests(self, requests):
        """Process multiple requests with agent pool."""
        tasks = []

        for request in requests:
            # Assign agent from pool
            agent = self.load_balancer.get_next_agent(self.agent_pool)

            # Process async
            task = asyncio.create_task(
                agent.process_async(request)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def auto_scale(self, load):
        """Automatically scale agent pool."""
        if load > 0.8 and len(self.agent_pool) < self.max_agents:
            # Scale up
            new_agents = self.create_agents(10)
            self.agent_pool.extend(new_agents)
        elif load < 0.3 and len(self.agent_pool) > 10:
            # Scale down
            self.agent_pool = self.agent_pool[:len(self.agent_pool)//2]
```

### 3. Safety & Alignment
```python
class SafeAgent:
    def __init__(self):
        self.safety_checks = []
        self.action_filter = ActionFilter()
        self.audit_log = []

    def add_safety_check(self, check_function):
        """Add safety validation."""
        self.safety_checks.append(check_function)

    def safe_execute(self, action):
        """Execute action with safety checks."""
        # Pre-execution checks
        for check in self.safety_checks:
            if not check(action):
                self.audit_log.append({
                    'action': action,
                    'status': 'blocked',
                    'reason': 'Failed safety check'
                })
                return {'error': 'Action blocked by safety check'}

        # Filter harmful actions
        filtered_action = self.action_filter.filter(action)

        # Execute with sandboxing
        result = self.sandboxed_execute(filtered_action)

        # Log execution
        self.audit_log.append({
            'action': filtered_action,
            'result': result,
            'timestamp': datetime.now()
        })

        return result
```

## Evaluation Frameworks

### 1. Task Success Metrics
- **Completion Rate**: Percentage of tasks completed
- **Step Efficiency**: Average steps to completion
- **Quality Score**: Output quality assessment
- **Autonomy Level**: Human intervention required

### 2. Learning Metrics
- **Adaptation Speed**: Time to learn new tasks
- **Strategy Discovery**: New strategies developed
- **Error Recovery**: Success after failures
- **Knowledge Transfer**: Applying learning to new domains

### 3. System Metrics
- **Scalability**: Agents handled concurrently
- **Resource Efficiency**: Compute/memory per agent
- **Robustness**: System stability over time
- **Coordination**: Multi-agent task success

## Exercises Overview

1. **Agent Builder**: Create complete autonomous agent
2. **Memory System**: Implement sophisticated memory
3. **Planner**: Build adaptive planning system
4. **Multi-Agent**: Coordinate multiple agents
5. **Production Agent**: Deploy scalable agent system

## Success Metrics
- **Task Success**: >85% completion rate
- **Autonomy**: <10% human intervention
- **Efficiency**: <10 steps average
- **Scalability**: Handle 100+ concurrent agents
- **Safety**: Zero harmful actions

## Next Steps
After mastering agent design, you'll move to Module 14: Production Patterns, where you'll learn enterprise deployment strategies, monitoring, debugging, and maintenance of LLM systems - essential for running agents in production environments.