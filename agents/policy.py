"""
Tool policy enforcement and safety checks.

This module implements allow-lists, usage auditing, and misuse detection.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from collections import defaultdict
import time


class PolicyViolation(Exception):
    """Tool policy violation."""
    pass


@dataclass
class ToolCall:
    """Record of a tool call."""
    timestamp: str
    user_id: str
    tool_name: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    violation: Optional[str] = None


class ToolPolicy:
    """
    Enforce tool usage policies.

    Features:
    - Allow-lists
    - Per-user rate limiting
    - Usage auditing
    - Misuse detection
    """

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        rate_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize policy enforcer.

        Args:
            allowed_tools: Set of allowed tool names (None = all allowed)
            rate_limits: Tool -> calls per minute mapping
        """
        self.allowed_tools = allowed_tools
        self.rate_limits = rate_limits or {}

        # Tracking
        self.call_log: List[ToolCall] = []
        self.user_calls = defaultdict(lambda: defaultdict(list))

        # Violation tracking
        self.violations = defaultdict(int)

    def is_allowed(self, tool_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if tool is in allow-list.

        Args:
            tool_name: Tool name

        Returns:
            Tuple of (allowed, reason)
        """
        if self.allowed_tools is None:
            return True, None

        if tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' not in allow-list"

        return True, None

    def check_rate_limit(
        self,
        user_id: str,
        tool_name: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user has exceeded rate limit for tool.

        Args:
            user_id: User ID
            tool_name: Tool name

        Returns:
            Tuple of (allowed, reason)
        """
        if tool_name not in self.rate_limits:
            return True, None

        limit = self.rate_limits[tool_name]

        # Get recent calls (last minute)
        now = time.time()
        minute_ago = now - 60

        recent_calls = [
            ts for ts in self.user_calls[user_id][tool_name]
            if ts > minute_ago
        ]

        if len(recent_calls) >= limit:
            return False, f"Rate limit exceeded ({limit}/min)"

        return True, None

    def record_call(
        self,
        user_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Record a tool call for auditing.

        Args:
            user_id: User ID
            tool_name: Tool name
            args: Tool arguments
            result: Tool result
        """
        # Record timestamp for rate limiting
        self.user_calls[user_id][tool_name].append(time.time())

        # Record full call for audit
        call = ToolCall(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            tool_name=tool_name,
            args=args,
            result=result,
            success=result.get("status") == "success"
        )

        self.call_log.append(call)

    def record_violation(
        self,
        user_id: str,
        tool_name: str,
        reason: str
    ):
        """
        Record a policy violation.

        Args:
            user_id: User ID
            tool_name: Tool name
            reason: Violation reason
        """
        violation_key = f"{user_id}:{tool_name}:{reason}"
        self.violations[violation_key] += 1

        # Log violation
        call = ToolCall(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            tool_name=tool_name,
            args={},
            result={"status": "blocked"},
            success=False,
            violation=reason
        )

        self.call_log.append(call)

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Stats dictionary
        """
        user_calls = [
            call for call in self.call_log
            if call.user_id == user_id
        ]

        tool_usage = defaultdict(int)
        for call in user_calls:
            tool_usage[call.tool_name] += 1

        violations = [
            call for call in user_calls
            if call.violation is not None
        ]

        return {
            "total_calls": len(user_calls),
            "successful_calls": sum(1 for c in user_calls if c.success),
            "tool_usage": dict(tool_usage),
            "violations": len(violations),
            "violation_details": [
                {"timestamp": v.timestamp, "tool": v.tool_name, "reason": v.violation}
                for v in violations[-10:]  # Last 10 violations
            ]
        }

    def export_audit_log(self, filepath: str):
        """
        Export audit log to file.

        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for call in self.call_log:
                f.write(json.dumps({
                    "timestamp": call.timestamp,
                    "user_id": call.user_id,
                    "tool": call.tool_name,
                    "args": call.args,
                    "success": call.success,
                    "violation": call.violation
                }) + "\n")

    def detect_suspicious_patterns(
        self,
        user_id: str,
        window_minutes: int = 10
    ) -> List[str]:
        """
        Detect suspicious usage patterns.

        Args:
            user_id: User ID
            window_minutes: Time window to analyze

        Returns:
            List of warnings
        """
        warnings = []

        # Get recent calls
        cutoff = time.time() - (window_minutes * 60)
        recent_calls = [
            call for call in self.call_log
            if call.user_id == user_id
            and datetime.fromisoformat(call.timestamp).timestamp() > cutoff
        ]

        if not recent_calls:
            return warnings

        # Check for rapid fire (> 30 calls in window)
        if len(recent_calls) > 30:
            warnings.append(f"Rapid fire detected: {len(recent_calls)} calls in {window_minutes}min")

        # Check for repeated failures
        failures = [c for c in recent_calls if not c.success]
        if len(failures) > 10:
            warnings.append(f"High failure rate: {len(failures)}/{len(recent_calls)} failed")

        # Check for same tool spam
        tool_counts = defaultdict(int)
        for call in recent_calls:
            tool_counts[call.tool_name] += 1

        for tool, count in tool_counts.items():
            if count > 20:
                warnings.append(f"Spamming {tool}: {count} calls")

        return warnings


class SafeToolExecutor:
    """
    Execute tools with policy enforcement.

    Combines ToolRegistry with ToolPolicy for safe execution.
    """

    def __init__(self, registry, policy: ToolPolicy):
        """
        Initialize executor.

        Args:
            registry: ToolRegistry instance
            policy: ToolPolicy instance
        """
        self.registry = registry
        self.policy = policy

    def execute(
        self,
        user_id: str,
        tool_name: str,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tool with policy checks.

        Args:
            user_id: User ID
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Execution result
        """
        # Check allow-list
        allowed, reason = self.policy.is_allowed(tool_name)
        if not allowed:
            self.policy.record_violation(user_id, tool_name, reason)
            return {
                "status": "blocked",
                "reason": reason
            }

        # Check rate limit
        allowed, reason = self.policy.check_rate_limit(user_id, tool_name)
        if not allowed:
            self.policy.record_violation(user_id, tool_name, reason)
            return {
                "status": "rate_limited",
                "reason": reason
            }

        # Get tool
        tool = self.registry.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "reason": f"Tool '{tool_name}' not found"
            }

        # Execute
        from .tools import call_tool
        result = call_tool(tool, args)

        # Record call
        self.policy.record_call(user_id, tool_name, args, result)

        return result


# Example usage
if __name__ == "__main__":
    print("Tool Policy Example")
    print("=" * 60)

    # Create policy
    policy = ToolPolicy(
        allowed_tools={"search_database", "calculate"},
        rate_limits={
            "search_database": 5,  # 5 calls per minute
            "calculate": 10
        }
    )

    # Test allow-list
    print("\nExample 1: Allow-list check")
    print("-" * 60)

    allowed, reason = policy.is_allowed("search_database")
    print(f"search_database: {'ALLOWED' if allowed else f'BLOCKED ({reason})'}")

    allowed, reason = policy.is_allowed("send_email")
    print(f"send_email: {'ALLOWED' if allowed else f'BLOCKED ({reason})'}")

    # Test rate limiting
    print("\nExample 2: Rate limiting")
    print("-" * 60)

    for i in range(7):
        allowed, reason = policy.check_rate_limit("user1", "search_database")
        status = "ALLOWED" if allowed else f"BLOCKED ({reason})"
        print(f"Call {i+1}: {status}")

        if allowed:
            policy.record_call(
                "user1",
                "search_database",
                {"query": f"test {i}"},
                {"status": "success"}
            )

    # Get user stats
    print("\nExample 3: User statistics")
    print("-" * 60)

    stats = policy.get_user_stats("user1")
    print(json.dumps(stats, indent=2))

    # Detect suspicious patterns
    print("\nExample 4: Suspicious pattern detection")
    print("-" * 60)

    # Simulate spam
    for i in range(25):
        policy.record_call(
            "user2",
            "calculate",
            {"expression": "1+1"},
            {"status": "success"}
        )

    warnings = policy.detect_suspicious_patterns("user2", window_minutes=10)
    for warning in warnings:
        print(f"⚠️  {warning}")
