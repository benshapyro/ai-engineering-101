"""
Module 05: Conditional Flow Patterns

Branching logic, decision trees, and dynamic routing in prompt chains.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.utils import LLMClient
import json
import time
from typing import Dict, Any, Optional, List
from enum import Enum


class DecisionPath(Enum):
    """Possible decision paths in conditional flows."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    URGENT = "urgent"
    STANDARD = "standard"


def example_1_simple_conditional():
    """Simple if-then-else conditional flow."""
    print("=" * 60)
    print("Example 1: Simple Conditional Flow")
    print("=" * 60)

    client = LLMClient("openai")

    # Input message
    message = "The server is down and customers can't access their accounts!"

    print(f"Input: {message}\n")

    # Step 1: Classify severity
    classify_prompt = f"""Classify the severity of this issue:

    Issue: {message}

    Severity (CRITICAL/HIGH/MEDIUM/LOW):"""

    print("STEP 1: Classifying severity...")
    severity = client.complete(classify_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"Severity: {severity}\n")

    # Conditional routing based on severity
    if "CRITICAL" in severity:
        print("→ CRITICAL PATH: Immediate escalation required\n")

        # Critical path: Generate emergency response
        emergency_prompt = f"""Generate emergency response protocol for:

        Critical Issue: {message}

        Emergency Response:
        1. Immediate Actions:"""

        response = client.complete(emergency_prompt, temperature=0.2, max_tokens=200)
        print(f"Emergency Protocol:\n{response.strip()}")

        # Additional critical step: Notify stakeholders
        notify_prompt = f"""Draft urgent notification for stakeholders:

        Critical Issue: {message}

        Notification:"""

        notification = client.complete(notify_prompt, temperature=0.2, max_tokens=150)
        print(f"\nStakeholder Notification:\n{notification.strip()}")

    elif "HIGH" in severity:
        print("→ HIGH PRIORITY PATH: Expedited handling\n")

        # High priority path
        priority_prompt = f"""Create prioritized action plan for:

        High Priority Issue: {message}

        Action Plan:"""

        response = client.complete(priority_prompt, temperature=0.3, max_tokens=200)
        print(f"Action Plan:\n{response.strip()}")

    else:
        print("→ STANDARD PATH: Normal processing\n")

        # Standard path
        standard_prompt = f"""Create standard ticket for:

        Issue: {message}

        Ticket Details:"""

        response = client.complete(standard_prompt, temperature=0.3, max_tokens=150)
        print(f"Ticket Created:\n{response.strip()}")


def example_2_multi_branch_decision_tree():
    """Complex decision tree with multiple branches."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Branch Decision Tree")
    print("=" * 60)

    client = LLMClient("openai")

    # Customer request
    request = """
    We need to analyze our sales data to understand why revenue dropped
    last quarter and create a presentation for the board meeting next week.
    """

    print(f"Request: {request}\n")

    # Level 1: Identify primary need
    primary_prompt = f"""Identify the primary need:

    Request: {request}

    Primary Need (DATA_ANALYSIS/PRESENTATION/INVESTIGATION/PLANNING):"""

    print("DECISION LEVEL 1: Primary Need")
    primary_need = client.complete(primary_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"→ {primary_need}\n")

    # Level 2: Branch based on primary need
    if "DATA_ANALYSIS" in primary_need or "INVESTIGATION" in primary_need:
        print("DECISION LEVEL 2: Analysis Type")

        analysis_prompt = f"""What type of analysis is needed?

        Request: {request}

        Analysis Type (DIAGNOSTIC/PREDICTIVE/DESCRIPTIVE/COMPARATIVE):"""

        analysis_type = client.complete(analysis_prompt, temperature=0.1, max_tokens=20).strip()
        print(f"→ {analysis_type}\n")

        # Level 3: Branch based on analysis type
        if "DIAGNOSTIC" in analysis_type:
            print("DECISION LEVEL 3: Diagnostic Approach\n")

            diagnostic_prompt = f"""Perform root cause analysis:

            Problem: Revenue drop last quarter
            Context: {request}

            Root Cause Analysis:"""

            result = client.complete(diagnostic_prompt, temperature=0.3, max_tokens=200)
            print(f"Diagnostic Output:\n{result.strip()}\n")

            # Continue to presentation branch
            print("→ Continuing to presentation preparation...\n")
            presentation_needed = True
        else:
            presentation_needed = False

    else:
        presentation_needed = True

    # Presentation branch
    if presentation_needed:
        print("DECISION LEVEL 2/3: Presentation Preparation")

        audience_prompt = f"""Identify the audience and their priorities:

        Context: Board meeting presentation about revenue drop

        Audience Analysis:"""

        audience = client.complete(audience_prompt, temperature=0.2, max_tokens=150)
        print(f"Audience Analysis:\n{audience.strip()}\n")

        # Generate presentation outline
        outline_prompt = f"""Create presentation outline:

        Topic: Revenue drop analysis
        Audience: {audience}

        Presentation Outline:"""

        outline = client.complete(outline_prompt, temperature=0.3, max_tokens=200)
        print(f"Presentation Outline:\n{outline.strip()}")


def example_3_dynamic_routing():
    """Dynamic routing based on content analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Dynamic Routing System")
    print("=" * 60)

    client = LLMClient("openai")

    # Various types of inputs to route
    inputs = [
        "How do I reset my password?",
        "I found a bug in the payment processing system",
        "Can we discuss partnership opportunities?",
        "The website is loading very slowly for all users"
    ]

    for input_text in inputs[:2]:  # Process first 2 for demo
        print(f"\nInput: '{input_text}'")
        print("-" * 40)

        # Step 1: Multi-label classification
        classify_prompt = f"""Classify this input (can have multiple labels):

        Input: {input_text}

        Classifications (check all that apply):
        - TECHNICAL: [yes/no]
        - BUSINESS: [yes/no]
        - URGENT: [yes/no]
        - CUSTOMER_FACING: [yes/no]
        - BUG_REPORT: [yes/no]

        Response format: Label1:yes, Label2:no, ..."""

        print("Analyzing input...")
        classifications = client.complete(classify_prompt, temperature=0.1, max_tokens=100).strip()
        print(f"Classifications: {classifications}\n")

        # Parse classifications and route accordingly
        is_technical = "TECHNICAL:yes" in classifications
        is_urgent = "URGENT:yes" in classifications
        is_bug = "BUG_REPORT:yes" in classifications
        is_customer = "CUSTOMER_FACING:yes" in classifications

        # Dynamic routing based on classifications
        routes_taken = []

        if is_bug:
            print("→ ROUTE: Bug Report Handler")
            bug_prompt = f"""Create bug report:

            Description: {input_text}

            Bug Report:
            - Severity:
            - Component:
            - Steps to reproduce:
            - Impact:"""

            bug_report = client.complete(bug_prompt, temperature=0.2, max_tokens=150)
            print(f"Bug Report:\n{bug_report.strip()[:100]}...\n")
            routes_taken.append("bug_handler")

        if is_urgent and is_technical:
            print("→ ROUTE: Emergency Technical Response")
            emergency_prompt = f"""Generate emergency technical response:

            Issue: {input_text}

            Immediate Actions:"""

            emergency = client.complete(emergency_prompt, temperature=0.2, max_tokens=100)
            print(f"Emergency Response:\n{emergency.strip()[:100]}...\n")
            routes_taken.append("emergency_tech")

        elif is_technical and not is_urgent:
            print("→ ROUTE: Standard Technical Support")
            tech_prompt = f"""Provide technical solution:

            Question: {input_text}

            Solution:"""

            tech_solution = client.complete(tech_prompt, temperature=0.3, max_tokens=100)
            print(f"Technical Solution:\n{tech_solution.strip()[:100]}...\n")
            routes_taken.append("standard_tech")

        if is_customer:
            print("→ ROUTE: Customer Communication")
            customer_prompt = f"""Draft customer response:

            Issue: {input_text}

            Customer Response:"""

            customer_response = client.complete(customer_prompt, temperature=0.3, max_tokens=100)
            print(f"Customer Response:\n{customer_response.strip()[:100]}...\n")
            routes_taken.append("customer_comm")

        print(f"Routes Taken: {', '.join(routes_taken)}")


def example_4_nested_conditionals():
    """Nested conditional logic with multiple decision levels."""
    print("\n" + "=" * 60)
    print("Example 4: Nested Conditional Logic")
    print("=" * 60)

    client = LLMClient("openai")

    # Data processing request
    data_request = """
    Process the customer dataset: remove duplicates, fix formatting issues,
    validate email addresses, and prepare for machine learning model training.
    Dataset size: 50,000 records. Contains PII.
    """

    print(f"Data Request: {data_request}\n")

    # Level 1: Check for PII
    pii_prompt = f"""Does this dataset contain PII (Personally Identifiable Information)?

    Request: {data_request}

    Contains PII (YES/NO):"""

    print("LEVEL 1: PII Check")
    has_pii = client.complete(pii_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"→ Contains PII: {has_pii}\n")

    if "YES" in has_pii:
        # Level 2A: PII handling required
        print("LEVEL 2A: PII Handling Protocol")

        anonymize_prompt = f"""Determine anonymization requirements:

        Dataset: {data_request}

        Anonymization Needed For:"""

        anonymization = client.complete(anonymize_prompt, temperature=0.2, max_tokens=100)
        print(f"→ Anonymization: {anonymization.strip()}\n")

        # Level 3A: Check compliance requirements
        print("LEVEL 3A: Compliance Check")

        compliance_prompt = f"""Check compliance requirements:

        Data with PII: {data_request}
        Anonymization plan: {anonymization}

        Compliance Requirements (GDPR/CCPA/HIPAA/Other):"""

        compliance = client.complete(compliance_prompt, temperature=0.1, max_tokens=100)
        print(f"→ Compliance: {compliance.strip()}\n")

        # Level 4A: Generate compliant processing plan
        if "GDPR" in compliance or "CCPA" in compliance:
            print("LEVEL 4A: Generate Compliant Processing Plan")

            compliant_plan = f"""Create GDPR/CCPA compliant processing plan:

            Requirements: {compliance}

            Processing Plan:"""

            plan = client.complete(compliant_plan, temperature=0.3, max_tokens=200)
            print(f"→ Plan: {plan.strip()}\n")

    # Level 2B: Data size check (parallel to PII check)
    print("LEVEL 2B: Data Size Assessment")

    size_prompt = f"""Assess processing requirements for dataset size:

    Request: {data_request}

    Processing Mode (BATCH/STREAM/DISTRIBUTED):"""

    processing_mode = client.complete(size_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"→ Processing Mode: {processing_mode}\n")

    if "DISTRIBUTED" in processing_mode:
        # Level 3B: Distributed processing setup
        print("LEVEL 3B: Distributed Processing Setup")

        distributed_prompt = f"""Design distributed processing architecture:

        Dataset: 50,000 records with PII
        Operations: Remove duplicates, fix formatting, validate emails

        Architecture:"""

        architecture = client.complete(distributed_prompt, temperature=0.3, max_tokens=150)
        print(f"→ Architecture: {architecture.strip()}\n")

    # Final step: Combine all decisions
    print("FINAL: Integrated Processing Pipeline")

    integrate_prompt = f"""Create integrated pipeline considering:

    - PII Handling: {has_pii}
    - Processing Mode: {processing_mode}
    - Original Requirements: {data_request[:100]}...

    Integrated Pipeline:"""

    pipeline = client.complete(integrate_prompt, temperature=0.3, max_tokens=200)
    print(f"Complete Pipeline:\n{pipeline.strip()}")


def example_5_fallback_handling():
    """Conditional flows with fallback options."""
    print("\n" + "=" * 60)
    print("Example 5: Fallback Handling")
    print("=" * 60)

    client = LLMClient("openai")

    # Complex query that might fail primary processing
    query = "Necesito ayuda con el código: def factorial(n): return n * factorial(n-1)"

    print(f"Query: {query}\n")

    # Primary attempt: Detect language
    detect_prompt = f"""Detect the language of this query:

    Query: {query}

    Language:"""

    print("PRIMARY: Language Detection")
    language = client.complete(detect_prompt, temperature=0.1, max_tokens=20).strip()
    print(f"→ Detected: {language}\n")

    # Check if we can handle the detected language
    supported_languages = ["English", "Spanish", "French"]

    if any(lang.lower() in language.lower() for lang in supported_languages):
        print(f"✓ Language supported: {language}\n")

        # Primary path: Process in detected language
        if "Spanish" in language:
            translate_prompt = f"""Translate this Spanish query to English:

            Spanish: {query}

            English:"""

            print("Translating from Spanish...")
            translation = client.complete(translate_prompt, temperature=0.2, max_tokens=100)
            query_english = translation.strip()
            print(f"Translation: {query_english}\n")
        else:
            query_english = query

        # Process the query
        process_prompt = f"""Analyze this code query:

        Query: {query_english}

        Analysis:"""

        print("Processing query...")
        analysis = client.complete(process_prompt, temperature=0.3, max_tokens=200)
        print(f"Analysis:\n{analysis.strip()}")

    else:
        print(f"✗ Language not supported: {language}\n")
        print("FALLBACK 1: Attempt pattern matching")

        # Fallback 1: Try to identify code regardless of language
        code_prompt = f"""Extract and analyze any code in this query (ignore the natural language):

        Query: {query}

        Code Analysis:"""

        code_analysis = client.complete(code_prompt, temperature=0.3, max_tokens=150)

        if "def" in query or "function" in query or "class" in query:
            print("✓ Code detected via pattern matching\n")
            print(f"Code Analysis:\n{code_analysis.strip()}")
        else:
            print("✗ No clear code pattern found\n")
            print("FALLBACK 2: Generic assistance")

            # Fallback 2: Provide generic help
            generic_prompt = f"""Provide general assistance for:

            Query: {query}

            Best effort response:"""

            generic = client.complete(generic_prompt, temperature=0.4, max_tokens=150)
            print(f"Generic Response:\n{generic.strip()}")


def example_6_state_dependent_routing():
    """Routing that depends on accumulated state."""
    print("\n" + "=" * 60)
    print("Example 6: State-Dependent Routing")
    print("=" * 60)

    client = LLMClient("openai")

    # Conversation with accumulating context
    conversation = [
        "I'm having issues with my cloud infrastructure",
        "The costs have increased by 300% this month",
        "We're seeing unusual traffic patterns",
        "Some services are timing out"
    ]

    # Track conversation state
    state = {
        "topics": [],
        "severity": "unknown",
        "category": "unknown",
        "previous_suggestions": []
    }

    print("Conversation Flow:\n")

    for i, message in enumerate(conversation, 1):
        print(f"Turn {i}: '{message}'")
        print("-" * 40)

        # Update state based on current message and history
        state_prompt = f"""Update conversation state:

        Current State:
        - Topics: {state['topics']}
        - Severity: {state['severity']}
        - Category: {state['category']}

        New Message: {message}

        Updated State (as JSON):"""

        print("Updating state...")
        state_update = client.complete(state_prompt, temperature=0.2, max_tokens=150)

        try:
            # Parse updated state (simplified for demo)
            if "cost" in message.lower():
                state["topics"].append("cost_optimization")
            if "traffic" in message.lower():
                state["topics"].append("traffic_anomaly")
            if "timing out" in message.lower():
                state["severity"] = "high"
                state["category"] = "performance"
        except:
            pass

        print(f"State: topics={state['topics']}, severity={state['severity']}\n")

        # Route based on accumulated state
        if len(state["topics"]) >= 3 and state["severity"] == "high":
            print("→ ROUTE: Comprehensive System Analysis")

            comprehensive_prompt = f"""Perform comprehensive analysis:

            Issues identified:
            - Topics: {state['topics']}
            - Recent messages: {conversation[i-2:i] if i >= 2 else conversation[:i]}

            Comprehensive Analysis:"""

            analysis = client.complete(comprehensive_prompt, temperature=0.3, max_tokens=200)
            print(f"Analysis:\n{analysis.strip()}\n")
            break

        elif "cost_optimization" in state["topics"] and "traffic_anomaly" in state["topics"]:
            print("→ ROUTE: Security + Cost Analysis")

            security_prompt = f"""Check for security issues causing cost increase:

            Symptoms: Increased costs, unusual traffic

            Security Check:"""

            security = client.complete(security_prompt, temperature=0.3, max_tokens=150)
            print(f"Security Analysis:\n{security.strip()}\n")

        else:
            print("→ ROUTE: Continue Gathering Information")

            gather_prompt = f"""Ask clarifying question about:

            Context: {message}

            Clarifying Question:"""

            question = client.complete(gather_prompt, temperature=0.3, max_tokens=50)
            print(f"Next Question: {question.strip()}\n")


def example_7_conditional_retry_logic():
    """Conditional retry with different strategies."""
    print("\n" + "=" * 60)
    print("Example 7: Conditional Retry Logic")
    print("=" * 60)

    client = LLMClient("openai")

    # Task that might need retries
    task = "Extract all email addresses and phone numbers from this text: Contact John at johndoe@email or call 555-1234"

    print(f"Task: {task}\n")

    max_attempts = 3
    attempt = 1
    success = False
    result = None

    while attempt <= max_attempts and not success:
        print(f"ATTEMPT {attempt}:")

        if attempt == 1:
            # First attempt: Standard approach
            extract_prompt = f"""{task}

            Extraction:"""

            result = client.complete(extract_prompt, temperature=0.1, max_tokens=100).strip()

        elif attempt == 2:
            # Second attempt: More specific instructions
            print("→ Retry with more specific instructions")

            specific_prompt = f"""Extract contact information with explicit formatting:

            Task: {task}

            Format your response as:
            Emails: [list emails]
            Phone Numbers: [list phone numbers]

            Extraction:"""

            result = client.complete(specific_prompt, temperature=0.1, max_tokens=100).strip()

        else:
            # Third attempt: Break down into steps
            print("→ Retry with step-by-step approach")

            steps_prompt = f"""Follow these steps:

            Step 1: Find all text that looks like email addresses
            Step 2: Find all text that looks like phone numbers
            Step 3: Format the results clearly

            Task: {task}

            Step-by-step extraction:"""

            result = client.complete(steps_prompt, temperature=0.2, max_tokens=150).strip()

        print(f"Result: {result}\n")

        # Validate result
        validate_prompt = f"""Validate this extraction:

        Original task: {task}
        Extraction result: {result}

        Is this extraction complete and correct? (YES/NO and reason):"""

        validation = client.complete(validate_prompt, temperature=0.1, max_tokens=50).strip()
        print(f"Validation: {validation}\n")

        if "YES" in validation:
            success = True
            print("✓ Extraction successful!")
        else:
            attempt += 1
            if attempt <= max_attempts:
                print(f"✗ Validation failed, retrying with different strategy...\n")
            else:
                print("✗ Max attempts reached. Using best effort result.")

    print(f"\nFinal Result after {attempt - 1} attempt(s):")
    print(result)


def run_all_examples():
    """Run all conditional flow examples."""
    examples = [
        example_1_simple_conditional,
        example_2_multi_branch_decision_tree,
        example_3_dynamic_routing,
        example_4_nested_conditionals,
        example_5_fallback_handling,
        example_6_state_dependent_routing,
        example_7_conditional_retry_logic
    ]

    for example in examples:
        try:
            example()
            print("\n" + "=" * 60 + "\n")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 05: Conditional Flows")
    parser.add_argument("--example", type=int, help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if args.all:
        run_all_examples()
    elif args.example:
        examples = {
            1: example_1_simple_conditional,
            2: example_2_multi_branch_decision_tree,
            3: example_3_dynamic_routing,
            4: example_4_nested_conditionals,
            5: example_5_fallback_handling,
            6: example_6_state_dependent_routing,
            7: example_7_conditional_retry_logic
        }
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"Invalid example number. Choose from 1-{len(examples)}")
    else:
        print("Module 05: Conditional Flow Patterns")
        print("\nUsage:")
        print("  python conditional_flows.py --all        # Run all examples")
        print("  python conditional_flows.py --example N  # Run specific example")
        print("\nAvailable examples:")
        print("  1: Simple Conditional")
        print("  2: Multi-Branch Decision Tree")
        print("  3: Dynamic Routing")
        print("  4: Nested Conditionals")
        print("  5: Fallback Handling")
        print("  6: State-Dependent Routing")
        print("  7: Conditional Retry Logic")