# Module 12: Prompt Optimization

## Learning Objectives
By the end of this module, you will:
- Master techniques for improving prompt effectiveness
- Optimize prompts for cost and performance
- Implement automated prompt testing and refinement
- Understand prompt compression and efficiency strategies
- Build systems for continuous prompt improvement

## Key Concepts

### 1. Optimization Dimensions

```python
class PromptOptimizationGoals:
    """Different aspects to optimize in prompts."""

    QUALITY = "accuracy, completeness, relevance"
    COST = "token usage, API calls, processing time"
    ROBUSTNESS = "consistency, error handling, edge cases"
    LATENCY = "response time, streaming, caching"
    INTERPRETABILITY = "clarity, explainability, debugging"
```

### 2. Prompt Efficiency Techniques

#### Token Reduction
```python
class TokenOptimizer:
    def compress_prompt(self, original_prompt):
        """Reduce tokens while maintaining effectiveness."""
        techniques = {
            'remove_redundancy': self.remove_redundant_words,
            'use_abbreviations': self.abbreviate_common_terms,
            'compact_examples': self.minimize_example_size,
            'optimize_instructions': self.concise_instructions
        }

        optimized = original_prompt
        for technique, method in techniques.items():
            optimized = method(optimized)

        return optimized

    def remove_redundant_words(self, text):
        """Remove unnecessary words."""
        redundant_phrases = {
            "please make sure to": "ensure",
            "in order to": "to",
            "at this point in time": "now",
            "due to the fact that": "because"
        }

        for verbose, concise in redundant_phrases.items():
            text = text.replace(verbose, concise)

        return text
```

#### Instruction Optimization
```python
def optimize_instructions(original_instructions):
    """Make instructions more effective."""
    # Before: Verbose and unclear
    verbose = """
    I would like you to carefully read through the following text
    and then provide me with a summary that captures the main points.
    Please ensure that your summary is comprehensive but also concise.
    """

    # After: Clear and concise
    optimized = """
    Summarize the key points concisely:
    """

    return optimized
```

### 3. Automated Prompt Testing

#### A/B Testing Framework
```python
class PromptABTester:
    def __init__(self, evaluation_metrics):
        self.metrics = evaluation_metrics
        self.results = defaultdict(list)

    def test_prompts(self, prompt_a, prompt_b, test_cases):
        """Compare two prompt versions."""
        for test_case in test_cases:
            # Test prompt A
            response_a = self.llm.generate(prompt_a.format(**test_case))
            score_a = self.evaluate(response_a, test_case['expected'])

            # Test prompt B
            response_b = self.llm.generate(prompt_b.format(**test_case))
            score_b = self.evaluate(response_b, test_case['expected'])

            self.results['prompt_a'].append(score_a)
            self.results['prompt_b'].append(score_b)

        return self.analyze_results()

    def analyze_results(self):
        """Statistical analysis of test results."""
        from scipy import stats

        scores_a = self.results['prompt_a']
        scores_b = self.results['prompt_b']

        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

        return {
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'std_a': np.std(scores_a),
            'std_b': np.std(scores_b),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'winner': 'A' if np.mean(scores_a) > np.mean(scores_b) else 'B'
        }
```

### 4. Prompt Evolution

#### Genetic Algorithm Optimization
```python
class GeneticPromptOptimizer:
    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size
        self.generations = generations

    def evolve_prompt(self, base_prompt, test_cases):
        """Evolve prompt through genetic algorithm."""
        # Initialize population
        population = self.create_initial_population(base_prompt)

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [
                self.evaluate_fitness(prompt, test_cases)
                for prompt in population
            ]

            # Selection
            parents = self.select_parents(population, fitness_scores)

            # Crossover and mutation
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)

            # Create new generation
            population = self.select_next_generation(
                population + offspring,
                fitness_scores
            )

        # Return best prompt
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]

    def mutate(self, prompt):
        """Random modifications to prompt."""
        mutations = [
            self.change_word_order,
            self.add_emphasis,
            self.modify_examples,
            self.adjust_formatting
        ]

        mutation = random.choice(mutations)
        return mutation(prompt)
```

### 5. Cost Optimization

#### Dynamic Prompt Sizing
```python
class CostOptimizer:
    def __init__(self, cost_per_1k_tokens=0.03):
        self.cost_per_1k = cost_per_1k_tokens
        self.token_budget = 1000

    def optimize_for_budget(self, prompt_template, context, budget):
        """Fit prompt within budget constraints."""
        # Calculate base cost
        base_tokens = count_tokens(prompt_template)

        # Available tokens for context
        available_tokens = budget - base_tokens

        # Select most important context
        prioritized_context = self.prioritize_context(
            context,
            available_tokens
        )

        # Build optimized prompt
        return prompt_template.format(context=prioritized_context)

    def progressive_prompting(self, task, max_cost=1.0):
        """Start simple, add complexity if needed."""
        prompts = [
            self.simple_prompt,     # $0.01
            self.detailed_prompt,   # $0.05
            self.expert_prompt     # $0.20
        ]

        for prompt in prompts:
            cost = self.estimate_cost(prompt)
            if cost > max_cost:
                break

            response = self.llm.generate(prompt)

            if self.is_satisfactory(response):
                return response, cost

        return response, cost
```

## Best Practices

### 1. Prompt Templates
```python
class OptimizedPromptTemplate:
    """Reusable, optimized prompt templates."""

    def __init__(self):
        self.templates = {}
        self.performance_data = {}

    def register_template(self, name, template, metadata=None):
        """Register optimized template."""
        self.templates[name] = {
            'template': template,
            'tokens': count_tokens(template),
            'metadata': metadata or {},
            'usage_count': 0,
            'avg_score': 0.0
        }

    def get_best_template(self, task_type):
        """Get highest performing template for task."""
        candidates = [
            (name, data) for name, data in self.templates.items()
            if data['metadata'].get('task_type') == task_type
        ]

        if not candidates:
            return None

        # Return template with highest average score
        best = max(candidates, key=lambda x: x[1]['avg_score'])
        return best[0]
```

### 2. Iterative Refinement
```python
class IterativeOptimizer:
    def refine_prompt(self, initial_prompt, test_cases, iterations=5):
        """Iteratively improve prompt."""
        current_prompt = initial_prompt
        best_score = 0

        for i in range(iterations):
            # Test current prompt
            score = self.evaluate_prompt(current_prompt, test_cases)

            if score > best_score:
                best_prompt = current_prompt
                best_score = score

            # Generate variations
            variations = self.generate_variations(current_prompt)

            # Test variations
            variation_scores = [
                self.evaluate_prompt(var, test_cases)
                for var in variations
            ]

            # Select best variation
            best_idx = np.argmax(variation_scores)
            current_prompt = variations[best_idx]

            print(f"Iteration {i+1}: Score improved to {best_score:.3f}")

        return best_prompt, best_score
```

### 3. Context Window Optimization
```python
def optimize_context_usage(prompt, context, max_tokens=4000):
    """Maximize useful information within token limits."""
    # Calculate token distribution
    prompt_tokens = count_tokens(prompt)
    available_for_context = max_tokens - prompt_tokens - 500  # Reserve for output

    # Rank context by relevance
    ranked_context = rank_by_relevance(context, prompt)

    # Greedily add context
    selected_context = []
    tokens_used = 0

    for item in ranked_context:
        item_tokens = count_tokens(item)
        if tokens_used + item_tokens <= available_for_context:
            selected_context.append(item)
            tokens_used += item_tokens
        else:
            # Try to fit compressed version
            compressed = compress_text(item, available_for_context - tokens_used)
            if compressed:
                selected_context.append(compressed)
                break

    return format_with_context(prompt, selected_context)
```

## Production Optimization

### 1. Performance Monitoring
```python
class PromptPerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def track_execution(self, prompt_id, execution_data):
        """Track prompt performance metrics."""
        self.metrics[prompt_id].append({
            'timestamp': datetime.now(),
            'latency': execution_data['latency'],
            'tokens': execution_data['tokens'],
            'cost': execution_data['cost'],
            'quality_score': execution_data.get('quality_score'),
            'error': execution_data.get('error')
        })

    def analyze_performance(self, prompt_id):
        """Analyze prompt performance over time."""
        data = self.metrics[prompt_id]

        return {
            'avg_latency': np.mean([d['latency'] for d in data]),
            'avg_cost': np.mean([d['cost'] for d in data]),
            'error_rate': sum(1 for d in data if d['error']) / len(data),
            'quality_trend': self.calculate_trend([d['quality_score'] for d in data if d['quality_score']])
        }
```

### 2. Caching Strategy
```python
class PromptCache:
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0

    def get_or_generate(self, prompt_key, generator_func):
        """Cache prompt results for reuse."""
        # Check cache
        if prompt_key in self.cache:
            entry = self.cache[prompt_key]
            if time.time() - entry['timestamp'] < self.ttl:
                self.hit_count += 1
                return entry['result']

        # Generate and cache
        self.miss_count += 1
        result = generator_func()

        self.cache[prompt_key] = {
            'result': result,
            'timestamp': time.time()
        }

        return result

    def optimization_stats(self):
        """Calculate optimization impact."""
        total = self.hit_count + self.miss_count
        return {
            'cache_hit_rate': self.hit_count / total if total > 0 else 0,
            'tokens_saved': self.hit_count * AVG_TOKENS_PER_CALL,
            'cost_saved': self.hit_count * AVG_COST_PER_CALL
        }
```

### 3. Batch Optimization
```python
async def batch_optimize_prompts(prompts, optimization_goals):
    """Optimize multiple prompts in parallel."""
    optimization_tasks = []

    for prompt in prompts:
        task = asyncio.create_task(
            optimize_single_prompt(prompt, optimization_goals)
        )
        optimization_tasks.append(task)

    optimized = await asyncio.gather(*optimization_tasks)

    # Analyze optimization results
    improvements = analyze_improvements(prompts, optimized)

    return optimized, improvements
```

## Advanced Techniques

### 1. Meta-Prompting
```python
def generate_optimized_prompt(task_description, constraints):
    """Use LLM to generate optimized prompts."""
    meta_prompt = f"""Create an optimized prompt for this task:

Task: {task_description}

Constraints:
- Maximum tokens: {constraints['max_tokens']}
- Target accuracy: {constraints['accuracy']}
- Response format: {constraints['format']}

Generate a prompt that is:
1. Clear and unambiguous
2. Token-efficient
3. Likely to produce accurate results

Optimized prompt:"""

    return llm.generate(meta_prompt)
```

### 2. Prompt Compression
```python
class PromptCompressor:
    def compress(self, prompt, target_reduction=0.3):
        """Compress prompt while maintaining meaning."""
        techniques = [
            self.remove_examples,      # If many examples
            self.summarize_context,    # Long context
            self.use_references,       # Repeated information
            self.apply_shorthand      # Common patterns
        ]

        current = prompt
        current_tokens = count_tokens(current)
        target_tokens = current_tokens * (1 - target_reduction)

        for technique in techniques:
            if count_tokens(current) <= target_tokens:
                break

            current = technique(current)

        return current
```

### 3. Adaptive Prompting
```python
class AdaptivePrompt:
    """Prompts that adapt based on model responses."""

    def __init__(self, base_prompt):
        self.base_prompt = base_prompt
        self.adaptation_history = []

    def generate(self, input_data):
        # Start with base prompt
        prompt = self.base_prompt

        # Apply learned adaptations
        for adaptation in self.adaptation_history:
            if adaptation['condition'](input_data):
                prompt = adaptation['modify'](prompt)

        response = llm.generate(prompt.format(**input_data))

        # Learn from response
        if not self.is_satisfactory(response):
            self.learn_adaptation(input_data, response)

        return response

    def learn_adaptation(self, input_data, response):
        """Learn how to adapt prompt for similar inputs."""
        analysis = self.analyze_failure(response)

        adaptation = {
            'condition': lambda x: similarity(x, input_data) > 0.8,
            'modify': self.create_modifier(analysis)
        }

        self.adaptation_history.append(adaptation)
```

## Evaluation Metrics

### 1. Quality Metrics
- **Accuracy**: Correctness of responses
- **Completeness**: Coverage of requirements
- **Relevance**: Focus on asked information
- **Consistency**: Stable outputs across runs

### 2. Efficiency Metrics
- **Token Usage**: Input/output token counts
- **Latency**: Time to first token
- **Cost**: Dollar cost per query
- **Cache Hit Rate**: Reuse effectiveness

### 3. Robustness Metrics
- **Error Rate**: Frequency of failures
- **Edge Case Handling**: Performance on unusual inputs
- **Variance**: Consistency across different inputs
- **Degradation**: Performance under constraints

## Exercises Overview

1. **Optimizer Builder**: Create prompt optimization pipeline
2. **A/B Tester**: Build testing framework
3. **Cost Reducer**: Minimize token usage
4. **Performance Tuner**: Optimize for speed
5. **Evolution Engine**: Implement genetic optimization

## Success Metrics
- **Quality Improvement**: >20% accuracy gain
- **Cost Reduction**: >40% token savings
- **Latency Reduction**: >30% faster responses
- **Robustness**: <5% error rate
- **Consistency**: >90% stable outputs

## Next Steps
After mastering prompt optimization, you'll move to Module 13: Agent Design, where you'll learn to build autonomous AI agents that can plan, execute, and adapt - utilizing optimized prompts for efficient agent operation.