"""
Module 09: Function Calling - Tool Ecosystem

Learn to build complete, extensible tool systems for AI agents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import inspect
import importlib
import pkgutil
from enum import Enum
import hashlib
from collections import defaultdict
import threading
import time


# ===== Example 1: Tool Registry System =====

def example_1_tool_registry():
    """Build a comprehensive tool registry system."""
    print("Example 1: Tool Registry System")
    print("=" * 50)

    class ToolCategory(Enum):
        """Tool categories for organization."""
        DATA = "data"
        COMMUNICATION = "communication"
        ANALYTICS = "analytics"
        AUTOMATION = "automation"
        SECURITY = "security"
        UTILITIES = "utilities"

    @dataclass
    class ToolMetadata:
        """Metadata for a registered tool."""
        name: str
        description: str
        category: ToolCategory
        version: str
        author: str
        tags: List[str] = field(default_factory=list)
        dependencies: List[str] = field(default_factory=list)
        required_permissions: List[str] = field(default_factory=list)
        deprecated: bool = False
        deprecation_message: Optional[str] = None
        examples: List[Dict] = field(default_factory=list)
        created_at: datetime = field(default_factory=datetime.now)
        updated_at: datetime = field(default_factory=datetime.now)

    @dataclass
    class ToolDefinition:
        """Complete tool definition."""
        metadata: ToolMetadata
        parameters: Dict[str, Any]
        function: Callable
        validator: Optional[Callable] = None
        preprocessor: Optional[Callable] = None
        postprocessor: Optional[Callable] = None

    class ToolRegistry:
        """Central registry for all tools."""

        def __init__(self):
            self.tools: Dict[str, ToolDefinition] = {}
            self.categories: Dict[ToolCategory, List[str]] = defaultdict(list)
            self.tags: Dict[str, Set[str]] = defaultdict(set)
            self.usage_stats: Dict[str, Dict] = defaultdict(lambda: {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_duration": 0,
                "last_used": None
            })

        def register_tool(self, tool_def: ToolDefinition) -> bool:
            """Register a new tool."""
            name = tool_def.metadata.name

            # Check for conflicts
            if name in self.tools:
                if not tool_def.metadata.deprecated:
                    print(f"Warning: Tool {name} already exists")
                    return False

            # Validate dependencies
            for dep in tool_def.metadata.dependencies:
                if dep not in self.tools:
                    print(f"Warning: Dependency {dep} not found for {name}")

            # Register tool
            self.tools[name] = tool_def
            self.categories[tool_def.metadata.category].append(name)

            # Index tags
            for tag in tool_def.metadata.tags:
                self.tags[tag].add(name)

            print(f"Registered tool: {name} v{tool_def.metadata.version}")
            return True

        def get_tool(self, name: str) -> Optional[ToolDefinition]:
            """Get a tool by name."""
            tool = self.tools.get(name)
            if tool and tool.metadata.deprecated:
                print(f"Warning: Tool {name} is deprecated. {tool.metadata.deprecation_message}")
            return tool

        def search_tools(self,
                        query: Optional[str] = None,
                        category: Optional[ToolCategory] = None,
                        tags: Optional[List[str]] = None) -> List[ToolDefinition]:
            """Search for tools."""
            results = []

            for name, tool in self.tools.items():
                # Skip deprecated unless specifically requested
                if tool.metadata.deprecated:
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
                    if (query_lower not in name.lower() and
                        query_lower not in tool.metadata.description.lower()):
                        continue

                results.append(tool)

            return results

        def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
            """Get all tools in a category."""
            tool_names = self.categories.get(category, [])
            return [self.tools[name] for name in tool_names if name in self.tools]

        def get_tools_by_tag(self, tag: str) -> List[ToolDefinition]:
            """Get all tools with a specific tag."""
            tool_names = self.tags.get(tag, set())
            return [self.tools[name] for name in tool_names if name in self.tools]

        def execute_tool(self, name: str, arguments: Dict) -> Dict:
            """Execute a tool with tracking."""
            tool = self.get_tool(name)
            if not tool:
                return {"error": f"Tool {name} not found"}

            start_time = time.time()
            stats = self.usage_stats[name]
            stats["call_count"] += 1
            stats["last_used"] = datetime.now()

            try:
                # Validate arguments
                if tool.validator:
                    validation_result = tool.validator(arguments)
                    if not validation_result.get("valid", False):
                        stats["error_count"] += 1
                        return {"error": f"Validation failed: {validation_result.get('error')}"}

                # Preprocess arguments
                if tool.preprocessor:
                    arguments = tool.preprocessor(arguments)

                # Execute function
                result = tool.function(**arguments)

                # Postprocess result
                if tool.postprocessor:
                    result = tool.postprocessor(result)

                stats["success_count"] += 1
                stats["total_duration"] += time.time() - start_time

                return {"status": "success", "result": result}

            except Exception as e:
                stats["error_count"] += 1
                stats["total_duration"] += time.time() - start_time
                return {"status": "error", "error": str(e)}

        def get_statistics(self) -> Dict:
            """Get usage statistics."""
            total_calls = sum(s["call_count"] for s in self.usage_stats.values())
            total_errors = sum(s["error_count"] for s in self.usage_stats.values())

            return {
                "total_tools": len(self.tools),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": total_errors / total_calls if total_calls > 0 else 0,
                "most_used": sorted(
                    [(name, stats["call_count"]) for name, stats in self.usage_stats.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "categories": {
                    cat.value: len(tools) for cat, tools in self.categories.items()
                }
            }

        def export_catalog(self) -> Dict:
            """Export tool catalog."""
            catalog = {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "tools": []
            }

            for name, tool in self.tools.items():
                catalog["tools"].append({
                    "name": name,
                    "description": tool.metadata.description,
                    "category": tool.metadata.category.value,
                    "version": tool.metadata.version,
                    "tags": tool.metadata.tags,
                    "parameters": tool.parameters,
                    "examples": tool.metadata.examples
                })

            return catalog

    # Create registry
    registry = ToolRegistry()

    # Define sample tools
    def create_sample_tools():
        """Create sample tools for the registry."""

        # Data tool
        def fetch_data(source: str, query: str, limit: int = 10) -> Dict:
            """Fetch data from a source."""
            return {
                "source": source,
                "query": query,
                "results": [f"Result {i}" for i in range(min(limit, 5))],
                "count": min(limit, 5)
            }

        fetch_tool = ToolDefinition(
            metadata=ToolMetadata(
                name="fetch_data",
                description="Fetch data from various sources",
                category=ToolCategory.DATA,
                version="1.0.0",
                author="System",
                tags=["database", "api", "retrieval"],
                examples=[
                    {"source": "database", "query": "SELECT * FROM users", "limit": 5}
                ]
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["database", "api", "file"]},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "required": ["source", "query"]
            },
            function=fetch_data,
            validator=lambda args: {"valid": "source" in args and "query" in args}
        )

        # Analytics tool
        def analyze_sentiment(text: str) -> Dict:
            """Analyze sentiment of text."""
            score = 0.5  # Mock score
            if "good" in text.lower() or "great" in text.lower():
                score = 0.8
            elif "bad" in text.lower() or "terrible" in text.lower():
                score = 0.2

            return {
                "text": text[:50] + "..." if len(text) > 50 else text,
                "sentiment": "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral",
                "score": score
            }

        sentiment_tool = ToolDefinition(
            metadata=ToolMetadata(
                name="analyze_sentiment",
                description="Analyze text sentiment",
                category=ToolCategory.ANALYTICS,
                version="2.0.0",
                author="AI Team",
                tags=["nlp", "text", "sentiment"],
                dependencies=["fetch_data"]
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 5000}
                },
                "required": ["text"]
            },
            function=analyze_sentiment
        )

        # Communication tool
        def send_notification(recipient: str, message: str, channel: str = "email") -> Dict:
            """Send a notification."""
            return {
                "sent": True,
                "recipient": recipient,
                "channel": channel,
                "timestamp": datetime.now().isoformat()
            }

        notification_tool = ToolDefinition(
            metadata=ToolMetadata(
                name="send_notification",
                description="Send notifications to users",
                category=ToolCategory.COMMUNICATION,
                version="1.5.0",
                author="Comms Team",
                tags=["email", "sms", "notification"],
                required_permissions=["notification:send"]
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                    "channel": {"type": "string", "enum": ["email", "sms", "push"]}
                },
                "required": ["recipient", "message"]
            },
            function=send_notification
        )

        return [fetch_tool, sentiment_tool, notification_tool]

    # Register tools
    for tool in create_sample_tools():
        registry.register_tool(tool)

    # Test registry operations
    print("\n" + "-" * 30)
    print("Registry Operations:")
    print("-" * 30)

    # Search by category
    data_tools = registry.get_tools_by_category(ToolCategory.DATA)
    print(f"\nData tools: {[t.metadata.name for t in data_tools]}")

    # Search by tag
    nlp_tools = registry.get_tools_by_tag("nlp")
    print(f"NLP tools: {[t.metadata.name for t in nlp_tools]}")

    # Search with query
    search_results = registry.search_tools(query="data")
    print(f"Search 'data': {[t.metadata.name for t in search_results]}")

    # Execute tool
    print("\n" + "-" * 30)
    print("Tool Execution:")
    print("-" * 30)

    result = registry.execute_tool("analyze_sentiment", {
        "text": "This tool registry system is really great!"
    })
    print(f"\nSentiment analysis result:")
    print(json.dumps(result, indent=2))

    # Get statistics
    print("\n" + "-" * 30)
    print("Registry Statistics:")
    print("-" * 30)
    stats = registry.get_statistics()
    print(json.dumps(stats, indent=2))

    # Export catalog
    catalog = registry.export_catalog()
    print(f"\nExported catalog with {len(catalog['tools'])} tools")


# ===== Example 2: Plugin Architecture =====

def example_2_plugin_architecture():
    """Implement plugin architecture for dynamic tool loading."""
    print("\nExample 2: Plugin Architecture")
    print("=" * 50)

    class ToolPlugin:
        """Base class for tool plugins."""

        def __init__(self):
            self.name = self.__class__.__name__
            self.version = "1.0.0"
            self.tools = []

        def initialize(self):
            """Initialize the plugin."""
            raise NotImplementedError

        def get_tools(self) -> List[Dict]:
            """Return tool definitions."""
            return self.tools

        def cleanup(self):
            """Cleanup resources."""
            pass

    class PluginManager:
        """Manage tool plugins."""

        def __init__(self):
            self.plugins: Dict[str, ToolPlugin] = {}
            self.plugin_paths = []
            self.loaded_modules = []

        def add_plugin_path(self, path: str):
            """Add a path to search for plugins."""
            if path not in self.plugin_paths:
                self.plugin_paths.append(path)
                if path not in sys.path:
                    sys.path.insert(0, path)

        def discover_plugins(self) -> List[str]:
            """Discover available plugins."""
            discovered = []

            for path in self.plugin_paths:
                if not os.path.exists(path):
                    continue

                # Look for Python files
                for filename in os.listdir(path):
                    if filename.endswith("_plugin.py"):
                        plugin_name = filename[:-3]  # Remove .py
                        if plugin_name not in discovered:
                            discovered.append(plugin_name)

            return discovered

        def load_plugin(self, plugin_name: str) -> bool:
            """Load a specific plugin."""
            try:
                # Import the module
                module = importlib.import_module(plugin_name)
                self.loaded_modules.append(module)

                # Find ToolPlugin subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, ToolPlugin) and
                        obj != ToolPlugin):

                        # Instantiate plugin
                        plugin_instance = obj()
                        plugin_instance.initialize()

                        # Register plugin
                        self.plugins[plugin_instance.name] = plugin_instance
                        print(f"Loaded plugin: {plugin_instance.name} v{plugin_instance.version}")
                        return True

            except Exception as e:
                print(f"Failed to load plugin {plugin_name}: {e}")
                return False

            return False

        def unload_plugin(self, plugin_name: str) -> bool:
            """Unload a plugin."""
            if plugin_name in self.plugins:
                plugin = self.plugins[plugin_name]
                plugin.cleanup()
                del self.plugins[plugin_name]
                print(f"Unloaded plugin: {plugin_name}")
                return True
            return False

        def get_all_tools(self) -> List[Dict]:
            """Get tools from all loaded plugins."""
            all_tools = []
            for plugin in self.plugins.values():
                all_tools.extend(plugin.get_tools())
            return all_tools

        def reload_plugin(self, plugin_name: str) -> bool:
            """Reload a plugin."""
            # Unload if exists
            if plugin_name in self.plugins:
                self.unload_plugin(plugin_name)

            # Find and reload module
            for module in self.loaded_modules:
                if plugin_name in module.__name__:
                    importlib.reload(module)
                    return self.load_plugin(plugin_name)

            return False

    # Create sample plugins
    class DataProcessingPlugin(ToolPlugin):
        """Plugin for data processing tools."""

        def initialize(self):
            """Initialize data processing tools."""
            self.version = "2.0.0"
            self.tools = [
                {
                    "name": "clean_data",
                    "description": "Clean and validate data",
                    "function": self._clean_data
                },
                {
                    "name": "transform_data",
                    "description": "Transform data format",
                    "function": self._transform_data
                }
            ]

        def _clean_data(self, data: List) -> List:
            """Clean data implementation."""
            return [d for d in data if d is not None]

        def _transform_data(self, data: List, format: str) -> Any:
            """Transform data implementation."""
            if format == "json":
                return json.dumps(data)
            return data

    class AutomationPlugin(ToolPlugin):
        """Plugin for automation tools."""

        def initialize(self):
            """Initialize automation tools."""
            self.tools = [
                {
                    "name": "schedule_task",
                    "description": "Schedule a task",
                    "function": self._schedule_task
                },
                {
                    "name": "run_workflow",
                    "description": "Run automation workflow",
                    "function": self._run_workflow
                }
            ]

        def _schedule_task(self, task: str, time: str) -> Dict:
            """Schedule task implementation."""
            return {"task": task, "scheduled_for": time, "status": "scheduled"}

        def _run_workflow(self, workflow_id: str) -> Dict:
            """Run workflow implementation."""
            return {"workflow_id": workflow_id, "status": "running"}

    # Test plugin system
    plugin_manager = PluginManager()

    # Register and load plugins
    print("Loading plugins...")
    print("-" * 30)

    # Simulate plugin loading
    data_plugin = DataProcessingPlugin()
    data_plugin.initialize()
    plugin_manager.plugins[data_plugin.name] = data_plugin

    automation_plugin = AutomationPlugin()
    automation_plugin.initialize()
    plugin_manager.plugins[automation_plugin.name] = automation_plugin

    # Get all tools
    all_tools = plugin_manager.get_all_tools()
    print(f"\nTotal tools available: {len(all_tools)}")
    for tool in all_tools:
        print(f"  - {tool['name']}: {tool['description']}")

    # Test plugin operations
    print("\n" + "-" * 30)
    print("Plugin Operations:")
    print("-" * 30)

    # Execute tool from plugin
    if all_tools:
        first_tool = all_tools[0]
        if "function" in first_tool:
            result = first_tool["function"]([1, None, 3, None, 5])
            print(f"\nExecuted {first_tool['name']}: {result}")

    # Unload plugin
    plugin_manager.unload_plugin("AutomationPlugin")
    print(f"\nRemaining plugins: {list(plugin_manager.plugins.keys())}")


# ===== Example 3: Tool Marketplace =====

def example_3_tool_marketplace():
    """Create a tool marketplace for discovery and installation."""
    print("\nExample 3: Tool Marketplace")
    print("=" * 50)

    @dataclass
    class MarketplaceTool:
        """Tool listing in marketplace."""
        id: str
        name: str
        description: str
        author: str
        version: str
        downloads: int = 0
        rating: float = 0.0
        reviews: List[Dict] = field(default_factory=list)
        price: float = 0.0  # 0 for free
        license: str = "MIT"
        repository: Optional[str] = None
        documentation: Optional[str] = None
        tags: List[str] = field(default_factory=list)
        requirements: List[str] = field(default_factory=list)
        screenshots: List[str] = field(default_factory=list)
        published_at: datetime = field(default_factory=datetime.now)
        updated_at: datetime = field(default_factory=datetime.now)

    class ToolMarketplace:
        """Marketplace for discovering and installing tools."""

        def __init__(self):
            self.tools: Dict[str, MarketplaceTool] = {}
            self.installed: Set[str] = set()
            self.user_ratings: Dict[str, List[float]] = defaultdict(list)
            self.trending_window = timedelta(days=7)

        def publish_tool(self, tool: MarketplaceTool) -> bool:
            """Publish a tool to the marketplace."""
            if tool.id in self.tools:
                print(f"Tool {tool.id} already exists")
                return False

            self.tools[tool.id] = tool
            print(f"Published: {tool.name} v{tool.version} by {tool.author}")
            return True

        def search(self,
                  query: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  min_rating: float = 0.0,
                  max_price: Optional[float] = None,
                  sort_by: str = "downloads") -> List[MarketplaceTool]:
            """Search marketplace."""
            results = []

            for tool in self.tools.values():
                # Filter by query
                if query:
                    query_lower = query.lower()
                    if (query_lower not in tool.name.lower() and
                        query_lower not in tool.description.lower()):
                        continue

                # Filter by tags
                if tags:
                    if not any(tag in tool.tags for tag in tags):
                        continue

                # Filter by rating
                if tool.rating < min_rating:
                    continue

                # Filter by price
                if max_price is not None and tool.price > max_price:
                    continue

                results.append(tool)

            # Sort results
            if sort_by == "downloads":
                results.sort(key=lambda x: x.downloads, reverse=True)
            elif sort_by == "rating":
                results.sort(key=lambda x: x.rating, reverse=True)
            elif sort_by == "newest":
                results.sort(key=lambda x: x.published_at, reverse=True)
            elif sort_by == "price":
                results.sort(key=lambda x: x.price)

            return results

        def get_trending(self, limit: int = 10) -> List[MarketplaceTool]:
            """Get trending tools."""
            cutoff_date = datetime.now() - self.trending_window

            # Calculate trend score
            trending = []
            for tool in self.tools.values():
                if tool.published_at > cutoff_date:
                    # New tools get a boost
                    trend_score = tool.downloads * 2 + tool.rating * 100
                else:
                    trend_score = tool.downloads + tool.rating * 50

                trending.append((tool, trend_score))

            trending.sort(key=lambda x: x[1], reverse=True)
            return [tool for tool, _ in trending[:limit]]

        def get_recommendations(self, user_tools: List[str], limit: int = 5) -> List[MarketplaceTool]:
            """Get personalized recommendations."""
            if not user_tools:
                # Return popular tools for new users
                return self.search(sort_by="downloads")[:limit]

            # Collect tags from user's tools
            user_tags = set()
            for tool_id in user_tools:
                if tool_id in self.tools:
                    user_tags.update(self.tools[tool_id].tags)

            # Find similar tools
            recommendations = []
            for tool in self.tools.values():
                if tool.id in user_tools:
                    continue

                # Calculate similarity score
                common_tags = len(set(tool.tags) & user_tags)
                if common_tags > 0:
                    score = common_tags * tool.rating
                    recommendations.append((tool, score))

            recommendations.sort(key=lambda x: x[1], reverse=True)
            return [tool for tool, _ in recommendations[:limit]]

        def install_tool(self, tool_id: str) -> bool:
            """Install a tool."""
            if tool_id not in self.tools:
                print(f"Tool {tool_id} not found")
                return False

            if tool_id in self.installed:
                print(f"Tool {tool_id} already installed")
                return False

            tool = self.tools[tool_id]

            # Check requirements
            for req in tool.requirements:
                if req not in self.installed and req not in self.tools:
                    print(f"Missing requirement: {req}")
                    return False

            # Simulate installation
            print(f"Installing {tool.name}...")
            time.sleep(0.5)  # Simulate download
            self.installed.add(tool_id)
            tool.downloads += 1
            print(f"Successfully installed {tool.name}")
            return True

        def rate_tool(self, tool_id: str, rating: float) -> bool:
            """Rate a tool."""
            if tool_id not in self.tools:
                return False

            if tool_id not in self.installed:
                print("You must install the tool before rating")
                return False

            self.user_ratings[tool_id].append(rating)
            tool = self.tools[tool_id]
            tool.rating = sum(self.user_ratings[tool_id]) / len(self.user_ratings[tool_id])
            return True

        def add_review(self, tool_id: str, review: Dict) -> bool:
            """Add a review for a tool."""
            if tool_id not in self.tools:
                return False

            if tool_id not in self.installed:
                print("You must install the tool before reviewing")
                return False

            tool = self.tools[tool_id]
            review["timestamp"] = datetime.now().isoformat()
            tool.reviews.append(review)
            return True

    # Create marketplace
    marketplace = ToolMarketplace()

    # Populate with sample tools
    sample_tools = [
        MarketplaceTool(
            id="data-wizard",
            name="Data Wizard",
            description="Comprehensive data processing toolkit",
            author="DataCorp",
            version="3.2.0",
            downloads=15000,
            rating=4.8,
            price=0.0,
            tags=["data", "etl", "analytics"],
            requirements=[]
        ),
        MarketplaceTool(
            id="api-connector",
            name="API Connector Pro",
            description="Connect to any API with ease",
            author="APITools Inc",
            version="2.5.0",
            downloads=8500,
            rating=4.6,
            price=19.99,
            tags=["api", "integration", "rest"],
            requirements=["data-wizard"]
        ),
        MarketplaceTool(
            id="ml-toolkit",
            name="ML Toolkit",
            description="Machine learning tools for everyone",
            author="AI Labs",
            version="1.0.0",
            downloads=3200,
            rating=4.9,
            price=0.0,
            tags=["ml", "ai", "analytics"],
            requirements=["data-wizard"]
        ),
        MarketplaceTool(
            id="code-analyzer",
            name="Code Analyzer",
            description="Static code analysis and optimization",
            author="DevTools",
            version="4.1.0",
            downloads=12000,
            rating=4.7,
            price=9.99,
            tags=["development", "quality", "testing"],
            requirements=[]
        ),
        MarketplaceTool(
            id="cloud-deploy",
            name="Cloud Deploy",
            description="Deploy to any cloud provider",
            author="CloudOps",
            version="2.0.0",
            downloads=6700,
            rating=4.5,
            price=29.99,
            tags=["cloud", "deployment", "devops"],
            requirements=["api-connector"]
        )
    ]

    for tool in sample_tools:
        marketplace.publish_tool(tool)

    # Test marketplace features
    print("\n" + "-" * 30)
    print("Marketplace Search:")
    print("-" * 30)

    # Search for data tools
    data_tools = marketplace.search(tags=["data"], sort_by="rating")
    print(f"\nData tools (sorted by rating):")
    for tool in data_tools[:3]:
        print(f"  - {tool.name}: ⭐ {tool.rating:.1f} (${tool.price})")

    # Get trending
    print("\n" + "-" * 30)
    print("Trending Tools:")
    print("-" * 30)
    trending = marketplace.get_trending(limit=3)
    for tool in trending:
        print(f"  - {tool.name}: {tool.downloads} downloads")

    # Install tools
    print("\n" + "-" * 30)
    print("Installation:")
    print("-" * 30)
    marketplace.install_tool("data-wizard")
    marketplace.install_tool("ml-toolkit")

    # Get recommendations
    print("\n" + "-" * 30)
    print("Recommendations:")
    print("-" * 30)
    recommendations = marketplace.get_recommendations(["data-wizard"], limit=3)
    print("Based on your tools, you might like:")
    for tool in recommendations:
        print(f"  - {tool.name}: {tool.description}")

    # Rate and review
    marketplace.rate_tool("data-wizard", 5.0)
    marketplace.add_review("data-wizard", {
        "user": "developer123",
        "comment": "Excellent tool for data processing!",
        "rating": 5
    })


# ===== Example 4: Permission System =====

def example_4_permission_system():
    """Implement permission-based tool access control."""
    print("\nExample 4: Permission System")
    print("=" * 50)

    @dataclass
    class User:
        """User with permissions."""
        id: str
        name: str
        roles: Set[str] = field(default_factory=set)
        permissions: Set[str] = field(default_factory=set)

    @dataclass
    class Role:
        """Role with permissions."""
        name: str
        permissions: Set[str]
        description: str

    class PermissionManager:
        """Manage permissions for tool access."""

        def __init__(self):
            self.users: Dict[str, User] = {}
            self.roles: Dict[str, Role] = {}
            self.tool_permissions: Dict[str, Set[str]] = {}

            # Define default roles
            self._create_default_roles()

        def _create_default_roles(self):
            """Create default system roles."""
            self.roles["admin"] = Role(
                name="admin",
                permissions={"*"},  # All permissions
                description="Full system access"
            )

            self.roles["developer"] = Role(
                name="developer",
                permissions={
                    "tool:read", "tool:execute", "tool:create",
                    "data:read", "data:write"
                },
                description="Developer access"
            )

            self.roles["analyst"] = Role(
                name="analyst",
                permissions={
                    "tool:read", "tool:execute",
                    "data:read", "analytics:*"
                },
                description="Analyst access"
            )

            self.roles["viewer"] = Role(
                name="viewer",
                permissions={"tool:read", "data:read"},
                description="Read-only access"
            )

        def create_user(self, user_id: str, name: str, roles: List[str] = None) -> User:
            """Create a new user."""
            user = User(id=user_id, name=name)

            if roles:
                for role_name in roles:
                    self.assign_role(user, role_name)

            self.users[user_id] = user
            return user

        def assign_role(self, user: User, role_name: str) -> bool:
            """Assign a role to a user."""
            if role_name not in self.roles:
                print(f"Role {role_name} not found")
                return False

            user.roles.add(role_name)
            # Add role permissions to user
            role = self.roles[role_name]
            user.permissions.update(role.permissions)
            return True

        def grant_permission(self, user: User, permission: str):
            """Grant specific permission to user."""
            user.permissions.add(permission)

        def revoke_permission(self, user: User, permission: str):
            """Revoke permission from user."""
            user.permissions.discard(permission)

        def check_permission(self, user: User, required_permission: str) -> bool:
            """Check if user has permission."""
            # Admin bypass
            if "*" in user.permissions:
                return True

            # Direct permission check
            if required_permission in user.permissions:
                return True

            # Wildcard permission check
            permission_parts = required_permission.split(":")
            for user_perm in user.permissions:
                if "*" in user_perm:
                    perm_pattern = user_perm.replace("*", "")
                    if required_permission.startswith(perm_pattern):
                        return True

            return False

        def register_tool_permissions(self, tool_name: str, required_permissions: Set[str]):
            """Register required permissions for a tool."""
            self.tool_permissions[tool_name] = required_permissions

        def can_execute_tool(self, user: User, tool_name: str) -> bool:
            """Check if user can execute a tool."""
            if tool_name not in self.tool_permissions:
                # No permissions required
                return True

            required = self.tool_permissions[tool_name]
            for perm in required:
                if not self.check_permission(user, perm):
                    return False

            return True

        def get_available_tools(self, user: User, all_tools: List[str]) -> List[str]:
            """Get tools available to a user."""
            available = []
            for tool in all_tools:
                if self.can_execute_tool(user, tool):
                    available.append(tool)
            return available

    class SecureToolExecutor:
        """Execute tools with permission checks."""

        def __init__(self, permission_manager: PermissionManager):
            self.permission_manager = permission_manager
            self.audit_log = []

        def execute(self, user: User, tool_name: str, arguments: Dict) -> Dict:
            """Execute tool with security checks."""
            # Check permission
            if not self.permission_manager.can_execute_tool(user, tool_name):
                self._log_access_denied(user, tool_name)
                return {
                    "error": "Permission denied",
                    "required_permissions": list(
                        self.permission_manager.tool_permissions.get(tool_name, set())
                    )
                }

            # Log successful access
            self._log_access_granted(user, tool_name)

            # Execute tool (mock)
            result = {
                "status": "success",
                "tool": tool_name,
                "user": user.name,
                "result": f"Executed {tool_name} with args: {arguments}"
            }

            return result

        def _log_access_granted(self, user: User, tool_name: str):
            """Log successful access."""
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "user": user.id,
                "tool": tool_name,
                "action": "execute",
                "result": "granted"
            })

        def _log_access_denied(self, user: User, tool_name: str):
            """Log denied access."""
            self.audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "user": user.id,
                "tool": tool_name,
                "action": "execute",
                "result": "denied"
            })

        def get_audit_log(self, user_id: Optional[str] = None) -> List[Dict]:
            """Get audit log, optionally filtered by user."""
            if user_id:
                return [log for log in self.audit_log if log["user"] == user_id]
            return self.audit_log

    # Test permission system
    perm_manager = PermissionManager()
    executor = SecureToolExecutor(perm_manager)

    # Register tool permissions
    perm_manager.register_tool_permissions("delete_data", {"data:delete", "admin:confirm"})
    perm_manager.register_tool_permissions("analyze_data", {"data:read", "analytics:run"})
    perm_manager.register_tool_permissions("export_report", {"data:read", "report:export"})

    # Create users with different roles
    admin = perm_manager.create_user("admin1", "Admin User", ["admin"])
    developer = perm_manager.create_user("dev1", "Developer User", ["developer"])
    analyst = perm_manager.create_user("analyst1", "Analyst User", ["analyst"])
    viewer = perm_manager.create_user("viewer1", "Viewer User", ["viewer"])

    # Test tool execution with different users
    print("Permission Tests:")
    print("-" * 30)

    test_cases = [
        (admin, "delete_data", {"table": "users"}),
        (developer, "delete_data", {"table": "logs"}),
        (analyst, "analyze_data", {"query": "SELECT *"}),
        (viewer, "export_report", {"format": "pdf"})
    ]

    for user, tool, args in test_cases:
        print(f"\n{user.name} trying to execute {tool}:")
        result = executor.execute(user, tool, args)
        if "error" in result:
            print(f"  ❌ {result['error']}")
            if "required_permissions" in result:
                print(f"     Required: {result['required_permissions']}")
        else:
            print(f"  ✅ {result['result']}")

    # Show available tools for each user
    print("\n" + "-" * 30)
    print("Available Tools by Role:")
    print("-" * 30)

    all_tools = ["delete_data", "analyze_data", "export_report", "view_data"]

    for user in [admin, developer, analyst, viewer]:
        available = perm_manager.get_available_tools(user, all_tools)
        print(f"\n{user.name} ({list(user.roles)[0] if user.roles else 'no role'}):")
        print(f"  Tools: {available}")

    # Show audit log
    print("\n" + "-" * 30)
    print("Audit Log:")
    print("-" * 30)
    for log in executor.get_audit_log()[:5]:
        print(f"  {log['timestamp']}: {log['user']} -> {log['tool']}: {log['result']}")


# ===== Example 5: Usage Analytics =====

def example_5_usage_analytics():
    """Track and analyze tool usage patterns."""
    print("\nExample 5: Usage Analytics")
    print("=" * 50)

    @dataclass
    class ToolMetric:
        """Metrics for a single tool execution."""
        tool_name: str
        user_id: str
        timestamp: datetime
        execution_time: float
        success: bool
        error: Optional[str] = None
        input_size: int = 0
        output_size: int = 0
        memory_usage: float = 0.0

    class AnalyticsEngine:
        """Analyze tool usage patterns."""

        def __init__(self):
            self.metrics: List[ToolMetric] = []
            self.real_time_listeners = []

        def track_execution(self, metric: ToolMetric):
            """Track a tool execution."""
            self.metrics.append(metric)

            # Notify real-time listeners
            for listener in self.real_time_listeners:
                listener(metric)

        def add_listener(self, listener: Callable):
            """Add real-time analytics listener."""
            self.real_time_listeners.append(listener)

        def get_summary(self, time_window: Optional[timedelta] = None) -> Dict:
            """Get usage summary."""
            if time_window:
                cutoff = datetime.now() - time_window
                relevant_metrics = [m for m in self.metrics if m.timestamp > cutoff]
            else:
                relevant_metrics = self.metrics

            if not relevant_metrics:
                return {"message": "No data available"}

            total_executions = len(relevant_metrics)
            successful = sum(1 for m in relevant_metrics if m.success)
            failed = total_executions - successful

            # Calculate averages
            avg_execution_time = sum(m.execution_time for m in relevant_metrics) / total_executions
            avg_input_size = sum(m.input_size for m in relevant_metrics) / total_executions
            avg_output_size = sum(m.output_size for m in relevant_metrics) / total_executions

            return {
                "total_executions": total_executions,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_executions,
                "avg_execution_time": avg_execution_time,
                "avg_input_size": avg_input_size,
                "avg_output_size": avg_output_size
            }

        def get_tool_statistics(self, tool_name: str) -> Dict:
            """Get statistics for a specific tool."""
            tool_metrics = [m for m in self.metrics if m.tool_name == tool_name]

            if not tool_metrics:
                return {"message": f"No data for tool {tool_name}"}

            return {
                "tool": tool_name,
                "total_calls": len(tool_metrics),
                "success_rate": sum(1 for m in tool_metrics if m.success) / len(tool_metrics),
                "avg_execution_time": sum(m.execution_time for m in tool_metrics) / len(tool_metrics),
                "unique_users": len(set(m.user_id for m in tool_metrics)),
                "errors": [m.error for m in tool_metrics if m.error][:5]  # Last 5 errors
            }

        def get_user_statistics(self, user_id: str) -> Dict:
            """Get statistics for a specific user."""
            user_metrics = [m for m in self.metrics if m.user_id == user_id]

            if not user_metrics:
                return {"message": f"No data for user {user_id}"}

            # Tool usage distribution
            tool_usage = defaultdict(int)
            for metric in user_metrics:
                tool_usage[metric.tool_name] += 1

            return {
                "user": user_id,
                "total_calls": len(user_metrics),
                "unique_tools": len(tool_usage),
                "most_used_tools": sorted(
                    tool_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "total_execution_time": sum(m.execution_time for m in user_metrics),
                "error_rate": sum(1 for m in user_metrics if not m.success) / len(user_metrics)
            }

        def get_trending_tools(self, time_window: timedelta = timedelta(hours=1)) -> List[tuple]:
            """Get trending tools in time window."""
            cutoff = datetime.now() - time_window
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]

            tool_counts = defaultdict(int)
            for metric in recent_metrics:
                tool_counts[metric.tool_name] += 1

            return sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)

        def get_error_analysis(self) -> Dict:
            """Analyze error patterns."""
            errors = [m for m in self.metrics if not m.success]

            if not errors:
                return {"message": "No errors found"}

            # Group errors by type
            error_types = defaultdict(list)
            for metric in errors:
                error_type = metric.error.split(":")[0] if metric.error else "Unknown"
                error_types[error_type].append(metric)

            return {
                "total_errors": len(errors),
                "error_rate": len(errors) / len(self.metrics),
                "by_type": {
                    error_type: {
                        "count": len(metrics),
                        "tools": list(set(m.tool_name for m in metrics)),
                        "recent": metrics[-1].timestamp.isoformat() if metrics else None
                    }
                    for error_type, metrics in error_types.items()
                }
            }

        def predict_usage(self, tool_name: str, next_hours: int = 1) -> Dict:
            """Predict future usage (simple prediction)."""
            # Get historical data
            tool_metrics = [m for m in self.metrics if m.tool_name == tool_name]

            if len(tool_metrics) < 10:
                return {"message": "Insufficient data for prediction"}

            # Simple moving average
            recent_window = timedelta(hours=1)
            cutoff = datetime.now() - recent_window
            recent_count = sum(1 for m in tool_metrics if m.timestamp > cutoff)

            predicted_count = recent_count * next_hours

            return {
                "tool": tool_name,
                "predicted_calls": predicted_count,
                "confidence": 0.7,  # Mock confidence
                "based_on": f"Last {recent_window.total_seconds() / 3600} hours"
            }

    # Create analytics engine
    analytics = AnalyticsEngine()

    # Add real-time listener
    def real_time_monitor(metric: ToolMetric):
        """Monitor executions in real-time."""
        if not metric.success:
            print(f"⚠️  Error in {metric.tool_name}: {metric.error}")

    analytics.add_listener(real_time_monitor)

    # Simulate tool executions
    print("Simulating tool executions...")
    print("-" * 30)

    import random

    tools = ["fetch_data", "analyze_data", "generate_report", "send_email"]
    users = ["user1", "user2", "user3"]

    for i in range(50):
        tool = random.choice(tools)
        user = random.choice(users)
        success = random.random() > 0.1  # 90% success rate

        metric = ToolMetric(
            tool_name=tool,
            user_id=user,
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 120)),
            execution_time=random.uniform(0.1, 5.0),
            success=success,
            error=None if success else random.choice(["Timeout", "ValidationError", "NetworkError"]),
            input_size=random.randint(100, 10000),
            output_size=random.randint(100, 50000),
            memory_usage=random.uniform(10, 500)
        )

        analytics.track_execution(metric)

    # Get analytics
    print("\n" + "-" * 30)
    print("Analytics Summary:")
    print("-" * 30)

    summary = analytics.get_summary()
    print(f"\nOverall Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Tool statistics
    print("\n" + "-" * 30)
    print("Tool Statistics:")
    print("-" * 30)

    for tool in tools[:2]:
        stats = analytics.get_tool_statistics(tool)
        print(f"\n{tool}:")
        print(f"  Total calls: {stats.get('total_calls', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"  Avg execution: {stats.get('avg_execution_time', 0):.2f}s")

    # Trending tools
    print("\n" + "-" * 30)
    print("Trending Tools (last hour):")
    print("-" * 30)

    trending = analytics.get_trending_tools(timedelta(hours=2))
    for tool, count in trending[:3]:
        print(f"  {tool}: {count} calls")

    # Error analysis
    print("\n" + "-" * 30)
    print("Error Analysis:")
    print("-" * 30)

    error_analysis = analytics.get_error_analysis()
    if "total_errors" in error_analysis:
        print(f"Total errors: {error_analysis['total_errors']}")
        print(f"Error rate: {error_analysis['error_rate']:.1%}")
        print("By type:")
        for error_type, details in error_analysis.get("by_type", {}).items():
            print(f"  {error_type}: {details['count']} occurrences")

    # Usage prediction
    print("\n" + "-" * 30)
    print("Usage Prediction:")
    print("-" * 30)

    prediction = analytics.predict_usage("fetch_data", next_hours=2)
    if "predicted_calls" in prediction:
        print(f"Predicted calls for fetch_data in next 2 hours: {prediction['predicted_calls']}")
        print(f"Confidence: {prediction['confidence']:.1%}")


# ===== Main Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 09: Tool Ecosystem Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_tool_registry,
        2: example_2_plugin_architecture,
        3: example_3_tool_marketplace,
        4: example_4_permission_system,
        5: example_5_usage_analytics
    }

    if args.all:
        for example in examples.values():
            example()
            print("\n" + "=" * 70 + "\n")
    elif args.example and args.example in examples:
        examples[args.example]()
    else:
        print("Module 09: Tool Ecosystem - Examples")
        print("\nUsage:")
        print("  python tool_ecosystem.py --example N  # Run example N")
        print("  python tool_ecosystem.py --all         # Run all examples")
        print("\nAvailable examples:")
        print("  1: Tool Registry System")
        print("  2: Plugin Architecture")
        print("  3: Tool Marketplace")
        print("  4: Permission System")
        print("  5: Usage Analytics")