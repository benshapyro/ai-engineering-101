# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a comprehensive prompt engineering curriculum with 14 progressive modules teaching LLM interaction from fundamentals to production deployment.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (OpenAI required, Anthropic/Azure optional)
```

### Running Examples
```bash
# Jupyter for interactive experimentation
jupyter notebook

# Python scripts directly
python [module-name]/examples/[example-file].py
```

## Project Architecture

### Module Structure
Each module (01-14) follows a consistent pattern:
- **README.md**: Learning objectives, concepts, and exercises
- **examples/**: Working code demonstrations
- **exercises/**: Practice problems for hands-on learning
- **solutions/**: Reference implementations

### Learning Progression
1. **Foundation (01-03)**: Core prompt engineering, zero-shot, few-shot
2. **Core Techniques (04-07)**: Chain-of-thought, prompt chaining, role-based, context management
3. **Advanced (08-11)**: Structured outputs, function calling, RAG basics and advanced patterns
4. **Expert (12-14)**: Optimization, agent design, production patterns

### Key Dependencies
- **LLM Libraries**: openai, anthropic for model interaction
- **RAG Components**: langchain, chromadb, faiss-cpu for retrieval systems
- **Production Tools**: fastapi, celery for deployment patterns
- **Utilities**: tiktoken for token counting, pydantic for validation

## Development Patterns

### API Configuration
All modules use environment variables from `.env` for API keys and model settings. Default models:
- OpenAI: `gpt-4-turbo-preview`
- Anthropic: `claude-3-opus-20240229`

### Common Utilities Location
Shared code lives in `shared/` directory:
- `utils.py`: Helper functions for API calls, token counting
- `prompts.py`: Reusable prompt templates

### Exercise Workflow
1. Read module README for concepts
2. Run examples to see implementations
3. Complete exercises in order
4. Check solutions only after attempting
5. Build module project to apply concepts