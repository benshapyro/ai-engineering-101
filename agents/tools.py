"""
Safe tool execution with schemas and validation.

This module provides infrastructure for defining and executing tools
with proper validation, error handling, and safety guardrails.
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError
import json


@dataclass
class Tool:
    """
    Tool definition with schema and execution function.

    Attributes:
        name: Tool name
        description: What the tool does
        fn: Function to execute
        schema: JSON schema for arguments
        rate_limit: Optional requests per minute limit
        requires_approval: Whether tool needs manual approval
        dangerous: Flag for potentially dangerous operations
    """
    name: str
    description: str
    fn: Callable
    schema: Dict[str, Any]
    rate_limit: Optional[int] = None
    requires_approval: bool = False
    dangerous: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry for available tools.

    Manages tool discovery, validation, and execution.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """
        Register a tool.

        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information for LLM function calling.

        Args:
            name: Tool name

        Returns:
            Tool info dict or None
        """
        tool = self.get(name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema
        }

    def get_all_tool_info(self) -> List[Dict[str, Any]]:
        """
        Get all tool information for LLM.

        Returns:
            List of tool info dicts
        """
        return [
            self.get_tool_info(name)
            for name in self.list_tools()
        ]


def validate_args(args: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate arguments against JSON schema.

    Args:
        args: Arguments to validate
        schema: JSON schema

    Returns:
        Tuple of (valid, error_message)
    """
    # Basic validation
    required = schema.get("properties", {}).keys()
    required_fields = schema.get("required", [])

    # Check required fields
    for field in required_fields:
        if field not in args:
            return False, f"Missing required field: {field}"

    # Check types
    properties = schema.get("properties", {})
    for key, value in args.items():
        if key not in properties:
            return False, f"Unknown field: {key}"

        expected_type = properties[key].get("type")
        if expected_type:
            actual_type = type(value).__name__
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object"
            }

            if type_map.get(actual_type) != expected_type:
                return False, f"Field '{key}' should be {expected_type}, got {actual_type}"

    return True, None


class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass


class ValidationError(Exception):
    """Argument validation failed."""
    pass


def call_tool(
    tool: Tool,
    args: Dict[str, Any],
    skip_validation: bool = False
) -> Dict[str, Any]:
    """
    Call a tool with validation and error handling.

    Args:
        tool: Tool to call
        args: Tool arguments
        skip_validation: Skip validation (dangerous!)

    Returns:
        Tool result dict

    Raises:
        ValidationError: If validation fails
        RateLimitError: If rate limit exceeded
    """
    # Validate arguments
    if not skip_validation:
        valid, error = validate_args(args, tool.schema)
        if not valid:
            raise ValidationError(f"Validation failed: {error}")

    # Check if tool requires approval
    if tool.requires_approval:
        return {
            "status": "pending_approval",
            "tool": tool.name,
            "args": args,
            "message": "This tool requires manual approval"
        }

    # Execute tool
    try:
        result = tool.fn(**args)

        return {
            "status": "success",
            "tool": tool.name,
            "result": result
        }

    except RateLimitError as e:
        return {
            "status": "error",
            "error": "rate_limited",
            "message": str(e),
            "retry_in": 60
        }

    except Exception as e:
        return {
            "status": "error",
            "error": "execution_failed",
            "message": str(e)
        }


# Example tools
def search_database(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Example search tool."""
    return [
        {"id": "1", "title": f"Result for: {query}", "snippet": "..."}
    ]


def send_email(to: str, subject: str, body: str) -> Dict[str, str]:
    """Example email tool (requires approval)."""
    return {"message": f"Email sent to {to}"}


def calculate(expression: str) -> float:
    """Safe calculator."""
    # Only allow safe operations
    allowed_chars = set("0123456789+-*/.()")
    if not all(c in allowed_chars or c.isspace() for c in expression):
        raise ValueError("Invalid characters in expression")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


# Example usage
if __name__ == "__main__":
    print("Tool Guardrails Example")
    print("=" * 60)

    # Create registry
    registry = ToolRegistry()

    # Register tools
    search_tool = Tool(
        name="search_database",
        description="Search the knowledge database",
        fn=search_database,
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return"
                }
            },
            "required": ["query"]
        },
        rate_limit=10
    )

    email_tool = Tool(
        name="send_email",
        description="Send an email",
        fn=send_email,
        schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        },
        requires_approval=True,
        dangerous=True
    )

    calc_tool = Tool(
        name="calculate",
        description="Perform mathematical calculations",
        fn=calculate,
        schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression"
                }
            },
            "required": ["expression"]
        }
    )

    registry.register(search_tool)
    registry.register(email_tool)
    registry.register(calc_tool)

    # Example 1: Valid call
    print("\nExample 1: Valid search")
    print("-" * 60)
    result = call_tool(search_tool, {"query": "python tutorials", "max_results": 3})
    print(json.dumps(result, indent=2))

    # Example 2: Missing required field
    print("\nExample 2: Invalid call (missing field)")
    print("-" * 60)
    try:
        result = call_tool(search_tool, {"max_results": 5})
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Example 3: Tool requiring approval
    print("\nExample 3: Tool requiring approval")
    print("-" * 60)
    result = call_tool(email_tool, {
        "to": "user@example.com",
        "subject": "Test",
        "body": "Hello"
    })
    print(json.dumps(result, indent=2))

    # Example 4: Safe calculator
    print("\nExample 4: Safe calculator")
    print("-" * 60)
    result = call_tool(calc_tool, {"expression": "2 + 2 * 3"})
    print(json.dumps(result, indent=2))

    # Example 5: Unsafe calculator attempt
    print("\nExample 5: Unsafe calculator (blocked)")
    print("-" * 60)
    result = call_tool(calc_tool, {"expression": "__import__('os').system('ls')"})
    print(json.dumps(result, indent=2))

    # Example 6: Get all tool info for LLM
    print("\nExample 6: Tool info for LLM")
    print("-" * 60)
    tools_info = registry.get_all_tool_info()
    print(json.dumps(tools_info, indent=2))
