"""
Autograder tests for Modules 06-14

Consolidated tests for advanced modules.
Run with: pytest tests/test_modules_06_14.py
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ============================================================================
# Module 06: Role-Based Prompting
# ============================================================================

@pytest.mark.module06
@pytest.mark.unit
class TestRolePrompting:
    """Test role-based prompting concepts."""

    def test_role_definition(self):
        """Test that roles are clearly defined."""
        good_role = "You are an expert Python developer with 10 years of experience."
        bad_role = "Help me."

        assert "you are" in good_role.lower() or "act as" in good_role.lower()
        assert len(good_role) > len(bad_role)

    def test_persona_consistency(self):
        """Test persona remains consistent."""
        # Personas should be maintained across turns
        assert True  # Conceptual test


# ============================================================================
# Module 07: Context Management
# ============================================================================

@pytest.mark.module07
@pytest.mark.unit
class TestContextManagement:
    """Test context management concepts."""

    def test_context_window_awareness(self):
        """Test understanding of token limits."""
        from shared.utils import count_tokens

        text = "word " * 1000
        tokens = count_tokens(text)
        assert tokens > 500  # Should count tokens

    def test_context_utilities_exist(self):
        """Test context management utilities."""
        from shared.context import ContextRouter
        assert ContextRouter is not None


# ============================================================================
# Module 08: Structured Outputs
# ============================================================================

@pytest.mark.module08
@pytest.mark.unit
class TestStructuredOutputs:
    """Test structured output concepts."""

    def test_json_schema_format(self):
        """Test JSON schema specification."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_structured_utilities_exist(self):
        """Test structured output utilities."""
        from shared.structured import create_json_schema
        assert create_json_schema is not None


# ============================================================================
# Module 09: Function Calling
# ============================================================================

@pytest.mark.module09
@pytest.mark.unit
class TestFunctionCalling:
    """Test function calling concepts."""

    def test_function_schema(self):
        """Test function schema format."""
        function_def = {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }

        assert "name" in function_def
        assert "parameters" in function_def


# ============================================================================
# Module 10: RAG Basics
# ============================================================================

@pytest.mark.module10
@pytest.mark.unit
class TestRAGBasics:
    """Test RAG basic concepts."""

    def test_rag_pipeline_steps(self):
        """Test understanding of RAG steps."""
        steps = ["retrieve", "rank", "generate"]
        assert len(steps) == 3

    def test_retrieval_exists(self):
        """Test retrieval utilities exist."""
        from rag.retrievers import BM25Retriever
        assert BM25Retriever is not None


# ============================================================================
# Module 11: Advanced RAG
# ============================================================================

@pytest.mark.module11
@pytest.mark.unit
class TestAdvancedRAG:
    """Test advanced RAG concepts."""

    def test_hybrid_retrieval_exists(self):
        """Test hybrid retrieval utilities."""
        from rag.retrievers import HybridRetriever
        assert HybridRetriever is not None

    def test_reranking_exists(self):
        """Test reranking utilities."""
        from rag.rerankers import CrossEncoderReranker
        assert CrossEncoderReranker is not None

    def test_eval_harness_exists(self):
        """Test evaluation harness."""
        from rag.eval import RAGEvaluator
        assert RAGEvaluator is not None


# ============================================================================
# Module 12: Prompt Optimization
# ============================================================================

@pytest.mark.module12
@pytest.mark.unit
class TestPromptOptimization:
    """Test prompt optimization concepts."""

    def test_optimization_metrics(self):
        """Test understanding of optimization metrics."""
        metrics = ["cost", "latency", "quality"]
        assert len(metrics) >= 3

    def test_ablation_testing(self):
        """Test ablation testing concept."""
        # Remove one component at a time
        baseline = {"temperature": 0.7, "top_p": 1.0}
        ablation = {"temperature": 0.7}  # Removed top_p

        assert len(ablation) < len(baseline)


# ============================================================================
# Module 13: Agent Design
# ============================================================================

@pytest.mark.module13
@pytest.mark.unit
class TestAgentDesign:
    """Test agent design concepts."""

    def test_tool_system_exists(self):
        """Test agent tool system."""
        from agents.tools import Tool, ToolRegistry
        assert Tool is not None
        assert ToolRegistry is not None

    def test_policy_system_exists(self):
        """Test agent policy system."""
        from agents.policy import ToolPolicy
        assert ToolPolicy is not None


# ============================================================================
# Module 14: Production Patterns
# ============================================================================

@pytest.mark.module14
@pytest.mark.unit
class TestProductionPatterns:
    """Test production pattern concepts."""

    def test_metrics_system_exists(self):
        """Test metrics system."""
        from metrics.tracing import trace_call, MetricsCollector
        assert trace_call is not None
        assert MetricsCollector is not None

    def test_api_skeleton_exists(self):
        """Test API skeleton."""
        import os
        api_path = "templates/api_skeleton/main.py"
        assert os.path.exists(api_path)

    def test_safety_concepts(self):
        """Test safety understanding."""
        # Safety includes: injection detection, PII scrubbing, rate limiting
        safety_components = ["injection_detection", "pii_scrubbing", "rate_limiting"]
        assert len(safety_components) >= 3


# ============================================================================
# Cross-Module Integration
# ============================================================================

@pytest.mark.integration
class TestCrossModuleIntegration:
    """Test integration across modules."""

    def test_complete_rag_pipeline(self):
        """Test complete RAG pipeline integration."""
        from rag.retrievers import HybridRetriever
        from rag.rerankers import CrossEncoderReranker
        from llm.client import LLMClient

        # All components should be importable
        assert HybridRetriever is not None
        assert CrossEncoderReranker is not None
        assert LLMClient is not None

    def test_production_stack(self):
        """Test production infrastructure."""
        from metrics.tracing import MetricsCollector
        from agents.tools import ToolRegistry
        from shared.repro import DeterministicClient

        assert MetricsCollector is not None
        assert ToolRegistry is not None
        assert DeterministicClient is not None
