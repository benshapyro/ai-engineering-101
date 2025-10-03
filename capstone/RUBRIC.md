# Capstone Project Grading Rubric

**Total Points**: 100

## 1. Functionality (40 points)

### Core RAG Pipeline (20 points)
- [ ] **Retrieval (8 pts)**: Hybrid retrieval combining dense and sparse methods
  - 8: Both methods implemented and properly fused (RRF or weighted)
  - 6: One method working, attempt at hybrid
  - 4: Basic retrieval only
  - 2: Retrieval incomplete or broken

- [ ] **Reranking (6 pts)**: Cross-encoder or LLM-based reranking
  - 6: Reranker properly integrated, measurable improvement
  - 4: Reranker implemented but not integrated
  - 2: Reranking attempted but not functional

- [ ] **Generation (6 pts)**: LLM generates accurate answers with citations
  - 6: Accurate answers with proper citation formatting
  - 4: Answers generated but citations incomplete
  - 2: Basic generation without citations

### Advanced Features (20 points)
- [ ] **Query Processing (5 pts)**: Expansion, correction, or classification
  - 5: Multiple techniques implemented
  - 3: One technique working
  - 1: Attempted but not functional

- [ ] **Structured Outputs (5 pts)**: JSON format with validation
  - 5: Proper JSON schema, validation, error handling
  - 3: JSON output but incomplete validation
  - 1: Attempted structured output

- [ ] **API Integration (5 pts)**: FastAPI REST endpoints
  - 5: Multiple endpoints, async, proper error handling
  - 3: Basic endpoint working
  - 1: API skeleton only

- [ ] **Evaluation System (5 pts)**: Automated metrics calculation
  - 5: Multiple metrics, comparison, reporting
  - 3: Basic metrics implemented
  - 1: Evaluation attempted

## 2. Code Quality (20 points)

### Organization (8 points)
- [ ] **Structure (4 pts)**: Logical file organization, clear separation of concerns
- [ ] **Naming (2 pts)**: Descriptive variable/function names
- [ ] **Modularity (2 pts)**: Reusable components, minimal duplication

### Documentation (6 points)
- [ ] **Docstrings (3 pts)**: All public functions documented
- [ ] **Comments (2 pts)**: Complex logic explained
- [ ] **README (1 pt)**: Usage instructions clear

### Testing (6 points)
- [ ] **Unit Tests (3 pts)**: Core functions tested
- [ ] **Integration Tests (2 pts)**: End-to-end scenarios
- [ ] **Coverage (1 pt)**: >60% code coverage

## 3. RAG Performance (20 points)

### Retrieval Quality (10 points)
- [ ] **Recall@5 (5 pts)**:
  - 5: >0.90
  - 4: 0.80-0.89
  - 3: 0.70-0.79
  - 2: 0.60-0.69
  - 1: <0.60

- [ ] **MRR (5 pts)**:
  - 5: >0.85
  - 4: 0.75-0.84
  - 3: 0.65-0.74
  - 2: 0.55-0.64
  - 1: <0.55

### Answer Quality (10 points)
- [ ] **Accuracy (5 pts)**: Correct answers on test set
  - 5: >80%
  - 4: 70-79%
  - 3: 60-69%
  - 2: 50-59%
  - 1: <50%

- [ ] **Citation Accuracy (5 pts)**: Valid source attribution
  - 5: >90% of claims cited correctly
  - 4: 80-89%
  - 3: 70-79%
  - 2: 60-69%
  - 1: <60%

## 4. Module Integration (10 points)

### Concept Application (10 points)
- [ ] **Fundamentals (2 pts)**: Clear prompts, delimiters, specificity
- [ ] **Advanced Prompting (2 pts)**: Chain-of-thought, few-shot, role-based
- [ ] **Context Management (2 pts)**: Token counting, truncation
- [ ] **Production Patterns (2 pts)**: Caching, monitoring, safety
- [ ] **Optimization (2 pts)**: Cost tracking, model selection

## 5. Production Readiness (10 points)

### Observability (4 points)
- [ ] **Logging (2 pts)**: Structured logs for debugging
- [ ] **Metrics (2 pts)**: Cost, latency, accuracy tracking

### Robustness (3 points)
- [ ] **Error Handling (2 pts)**: Graceful failures, retry logic
- [ ] **Input Validation (1 pt)**: Sanitization, limits

### Deployment (3 points)
- [ ] **Configuration (1 pt)**: Environment variables, .env
- [ ] **Documentation (1 pt)**: Setup and usage instructions
- [ ] **Testing (1 pt)**: Tests pass, CI ready

---

## Grading Scale

- **90-100**: Excellent - Production-ready system with all advanced features
- **80-89**: Good - Functional RAG system with most features working
- **70-79**: Satisfactory - Core RAG pipeline works, some advanced features
- **60-69**: Needs Improvement - Basic functionality but missing key features
- **Below 60**: Incomplete - Major features missing or broken

## Minimum Requirements (Pass)

To pass (70%), the project MUST include:
1. ✅ Working retrieval (any method)
2. ✅ LLM generation with citations
3. ✅ Basic evaluation (at least one metric)
4. ✅ Documentation (README with usage)
5. ✅ Tests (at least basic unit tests)
6. ✅ Structured outputs (JSON format)

## Bonus Points (up to 10 extra)

- **Advanced Reranking** (+3): Multiple reranking strategies compared
- **Multi-hop Reasoning** (+3): Complex queries requiring chain-of-thought
- **Ablation Studies** (+2): Systematic feature evaluation
- **Optimization Report** (+2): Detailed cost/quality analysis
- **Creative Extension** (+5): Novel feature not in requirements

---

## Self-Assessment Checklist

Before submission, verify:
- [ ] Code runs without errors
- [ ] All dependencies in requirements.txt
- [ ] Environment setup documented
- [ ] Tests pass (pytest tests/)
- [ ] Evaluation produces metrics
- [ ] API responds to sample queries
- [ ] README includes usage examples
- [ ] Code is formatted (black)
- [ ] No API keys committed
- [ ] Citations properly attributed

## Submission

Submit:
1. Complete capstone/ directory
2. Evaluation results (JSON or HTML report)
3. Brief reflection (1-2 pages):
   - Design decisions
   - Challenges encountered
   - Performance analysis
   - What you learned

Good luck! 🚀
