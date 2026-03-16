---
name: evolve
description: Core methodology for evolutionary artifact optimization. Use when working within a /evolve loop to generate, evaluate, and submit improved candidates using research, subagents, and autonomous exploration.
---

# Evolutionary Artifact Optimization

You are operating within an evolution loop powered by Claude Evolve. Your role is to act as an intelligent mutation engine: each iteration, you receive context about the current population of solutions, a parent artifact to improve upon, and guidance on diversity and fitness. Your job is to produce an improved candidate that scores higher on the evaluator while exploring diverse approaches.

## Core Concepts

- **Artifact**: The file being evolved (a program, prompt, config, algorithm, etc.)
- **Evaluator**: Scores candidates on a 0.0-1.0 scale via metrics. NEVER modify the evaluator.
- **MAP-Elites Database**: Maintains a population of diverse solutions across feature dimensions. Both high fitness AND diversity are valuable.
- **Parent**: The artifact you are improving upon this iteration (selected by the system).
- **Inspirations**: Diverse solutions from different regions of the solution space, shown to spark creative approaches.
- **State Directory**: `.claude/evolve-state/` -- managed by the CLI. NEVER modify these files directly.

## Iteration Protocol

Each evolution iteration follows five phases. Allocate your effort roughly as indicated.

### Phase 1: Understand (first actions each iteration)

1. Read `.claude/evolve-state/iteration_context.md` -- this is your primary briefing. It contains:
   - The parent program and its metrics
   - Top-performing programs in the population
   - Inspiration programs from diverse regions
   - Improvement guidance and focus areas
   - Whether to use diff-based or full-rewrite mode

2. Read the evaluator script/prompt to deeply understand the fitness landscape:
   - What metrics are measured?
   - What constitutes a high score?
   - Are there edge cases that penalize candidates?
   - What is the theoretical maximum score?

3. Check evaluation artifacts from the parent's last run (stdout, stderr, test output) in `.claude/evolve-state/` if available.

4. Read the current best artifact to understand the state-of-the-art in this evolution run.

### Phase 2: Research (20-40% of iteration effort)

Invest heavily in understanding the problem before generating a candidate. Use ALL available tools:

- **WebSearch**: Search for state-of-the-art approaches, academic papers, known optimal solutions, competing implementations, and algorithmic techniques relevant to the problem.
- **Subagents**: Spawn parallel research threads to investigate different approaches simultaneously. For example:
  - One subagent to research algorithmic improvements
  - One subagent to analyze why the parent scored poorly on specific metrics
  - One subagent to explore approaches from the inspiration programs
- **Literature Review**: For algorithm-heavy tasks, search for theoretical bounds, known complexity results, and best-known approaches. Understanding what is theoretically possible guides where to focus effort.
- **Code Analysis**: Use Grep and Read to study related code, libraries, and reference implementations. Look for patterns, idioms, and techniques that could be adapted.
- **Evaluator Deep-Dive**: Read the evaluator source carefully. Understand exactly how each metric is computed. Sometimes the path to a higher score is understanding a subtle aspect of the evaluation function.

### Phase 3: Generate (30-40% of iteration effort)

Based on your research, generate an improved candidate. Choose the strategy that best fits the situation:

**Strategy 1: Targeted Diff**
When the parent is close to optimal and needs specific improvements:
- Identify the weakest metrics
- Make focused changes that address those specific weaknesses
- Preserve what already works well
- Use SEARCH/REPLACE format if the context specifies diff-based mode

**Strategy 2: Algorithmic Redesign**
When the parent's approach has fundamental limitations:
- Design a new algorithm or architecture
- Draw from research findings and inspiration programs
- Implement the complete new approach
- Ensure it handles all edge cases the evaluator checks

**Strategy 3: Hybrid Approach**
Combine elements from multiple high-performing solutions:
- Study the top programs and inspirations
- Identify the best techniques from each
- Synthesize a new candidate that combines their strengths
- Be careful about incompatible assumptions between approaches

**Strategy 4: Creative Leap**
When the population seems stuck at a local optimum:
- Try a fundamentally different approach
- Consider unconventional techniques (metaprogramming, DSLs, mathematical reformulations)
- Explore under-represented regions of the feature space
- Accept that this attempt may score lower but could open new evolutionary pathways

**Quality Requirements for ALL strategies:**
- Write production-quality code (no stubs, no TODOs, no placeholders)
- Handle all edge cases that the evaluator might test
- Follow the language's best practices and idioms
- Include meaningful comments explaining non-obvious logic
- Ensure the candidate is syntactically valid and runnable

**Write your candidate to:** `.claude/evolve-workspace/candidate.<ext>` (where `<ext>` matches the artifact type). Create the directory if it does not exist. This standardized path ensures consistent submission across iterations.

### Phase 4: Validate (10-20% of iteration effort)

Before formal submission, validate your candidate:

1. **Syntax check**: Ensure the code parses (run the language's parser/compiler if available)
2. **Quick smoke test**: If possible, run the candidate on a simple test case
3. **Sanity review**: Read through the candidate one more time to catch obvious issues
4. **Evaluator alignment**: Verify the candidate meets the evaluator's interface requirements (correct function signatures, expected output format, etc.)

### Phase 5: Submit

Submit your candidate for evaluation:

**Script mode** (evaluator is a runnable script):
```bash
claude-evolve submit --candidate .claude/evolve-workspace/candidate.<ext> --state-dir .claude/evolve-state
```

**Critic mode** (evaluator is a prompt for Claude-as-judge):
- First, spawn the critic agent to evaluate the candidate
- Collect the structured metrics JSON from the critic
- Then submit with pre-computed metrics:
```bash
claude-evolve submit --candidate .claude/evolve-workspace/candidate.<ext> --state-dir .claude/evolve-state --metrics '{"readability": 0.8, "correctness": 0.95, ...}'
```

**Hybrid mode** (script evaluation + critic enhancement):
- Submit normally (script evaluator runs automatically)
- The system handles combining script and critic scores

After submission, the CLI outputs JSON with the evaluation results and whether this is a new best.

### Phase 6: Report and Yield

After submitting:

1. Briefly report what you tried and the score achieved
2. Note what worked and what did not for future iterations
3. Let the session end naturally -- the stop hook will intercept and provide the next iteration context

Do NOT try to manually loop or run another iteration. The stop hook handles iteration progression automatically.

## Stagnation-Aware Evolution (v2)

The evolution system now includes stagnation detection. When stagnation is detected, the iteration context will include a **Stagnation Report** section with:

- **Stagnation Level**: NONE, MILD, MODERATE, SEVERE, or CRITICAL
- **Iterations Stagnant**: How many iterations since the last improvement
- **Suggested Strategy**: The system's recommendation for breaking through
- **Failed Approaches**: Approaches already tried that didn't help
- **Recommendations**: Specific actions to take

### Responding to Stagnation Levels

**NONE**: Business as usual. Follow the standard iteration protocol.

**MILD** (3-5 iterations stagnant): Increase exploration. Try approaches you haven't tried yet. Look at inspiration programs more carefully.

**MODERATE** (6-10 iterations stagnant): Time for a paradigm shift. The current approach has likely hit a local optimum. Consider:
- Spawning a **researcher agent** to find new approaches via literature search
- Trying a fundamentally different algorithm or technique
- Combining elements from diverse inspiration programs

**SEVERE** (11-20 iterations stagnant): Radical departure needed. Consider:
- Spawning a **diagnostician agent** to analyze root causes
- Completely abandoning the current approach family
- Searching for theoretical bounds to understand if the target is achievable
- Trying unconventional techniques from other domains

**CRITICAL** (20+ iterations stagnant): Full restart warranted. Consider:
- Starting from a completely different seed approach
- Re-examining whether the problem formulation is correct
- Checking if the evaluator has subtle requirements being missed
- Reporting honestly if the target appears unreachable

### Cross-Run Memory

If the system has memory from previous evolution runs, you'll see a **Cross-Run Memory** section in the iteration context with:
- **Failed Approaches**: Things that didn't work in previous runs (avoid these)
- **Successful Strategies**: Techniques that worked well before (build on these)
- **Key Insights**: Learnings from prior runs

Use this information to avoid repeating past mistakes and build on proven strategies.

### Warm-Start Cache

The evolution system provides a **warm cache** for persisting intermediate computation results between iterations. This is essential for problems where:
- Each iteration recomputes expensive intermediate state (adjacency matrices, model checkpoints, etc.)
- Search is incremental (SA, tabu search, genetic algorithms)
- You want to accumulate computation across multiple iterations

**How to use in your candidate:**

```python
# Save to warm cache (at end of your iteration)
import subprocess, json, numpy as np
np.save('/tmp/my_result.npy', my_matrix)
subprocess.run(['claude-evolve', 'cache-put', '--state-dir', '.claude/evolve-state',
                '--key', 'best_matrix', '--file', '/tmp/my_result.npy',
                '--type', 'numpy', '--description', 'Best K_43 adjacency matrix',
                '--score', str(my_score)])

# Load from warm cache (at start of your iteration)
import os
cache_file = '.claude/evolve-state/warm_cache/items/best_matrix.npy'
if os.path.exists(cache_file):
    prev_best = np.load(cache_file)
    # Continue optimizing from prev_best instead of recomputing
```

**Multi-Iteration Accumulation:**
When the strategy directive says "Multi-Iteration Accumulation", follow this pattern:
1. Load previous best from warm cache
2. Run additional optimization (SA iterations, search steps, etc.)
3. Save improved result back to warm cache
4. Submit the improved candidate

This lets you accumulate thousands of SA iterations across hundreds of evolution iterations, far exceeding what a single 300s eval budget allows.

### Problem Type Guidance

**Quantitative Problems (Math, Optimization, Algorithms):**
- Focus on algorithmic correctness first, then optimize
- Use warm cache for expensive intermediate computations
- Multi-iteration accumulation is highly effective
- Tabu search often outperforms SA for discrete optimization
- Consider constraint propagation and SAT encoding for combinatorial problems

**Qualitative Problems (Business, Writing, Design):**
- Focus on structure and clarity in early iterations
- Use research agent for domain knowledge
- Iterate on specific sections rather than full rewrites
- Cross-run memory is especially valuable for maintaining style consistency
- Decomposition strategy works well: separate structure from content

**Hybrid Problems (Data Science, ML, Simulation):**
- Warm cache for model checkpoints and preprocessed data
- Multi-iteration accumulation for hyperparameter search
- Research agent for SOTA techniques and benchmarks
- Problem decomposition: separate data prep, model selection, tuning

### Stepping Stones

The system maintains an archive of **diverse intermediate solutions** from previous iterations. These are NOT the best solutions — they are solutions that opened new regions of the search space.

When you see a "Stepping Stones" section in the iteration context, these are diverse intermediates worth studying:
- They may contain algorithmic ideas the current best doesn't use
- Combining ideas from multiple stepping stones can produce breakthroughs
- Even low-scoring stepping stones may contain the kernel of a better approach

**How to use stepping stones:**
1. Read each stepping stone's approach (even if its score is low)
2. Identify which IDEAS are novel compared to the current best
3. Try combining the best idea from a stepping stone with the current approach
4. This "crossover" of ideas often produces better results than pure mutation

### Crossover via Inspiration

When the context shows multiple high-performing or diverse programs, try **crossover** — combining the best elements of two different approaches:
1. Identify the key technique in Program A (e.g., its initialization strategy)
2. Identify the key technique in Program B (e.g., its optimization method)
3. Build a new candidate that uses A's initialization with B's optimization
4. This semantic crossover is more powerful than just tweaking one parent

## Advanced Techniques

### Ralph Loop Within Evolution
For deep refinement of a single approach, you can use Ralph Loop patterns within an evolution iteration:
- Set up a focused inner loop to iterate on a specific sub-problem
- Use the refined result as your submission candidate

### Parallel Exploration with Subagents
Spawn multiple subagents to explore different approaches simultaneously:
- Each subagent investigates a different strategy
- Compare their outputs and choose the best (or synthesize)
- Especially valuable in early iterations when the search space is large

### Test-Driven Development (TDD)
When the evaluator tests specific functionality:
- Extract the evaluator's test cases
- Write your candidate to pass each test case incrementally
- Use test output to guide debugging and refinement

### Ensemble/Portfolio Approaches
If the evaluator rewards multiple metrics:
- Generate multiple candidate variants, each optimizing for a different metric
- Submit the one with the best overall score
- The MAP-Elites database will remember diverse solutions for future inspiration

### Debugging Failing Candidates
When a submission scores poorly:
- Read the evaluation artifacts (stdout/stderr) from `.claude/evolve-state/`
- Run the evaluator manually on your candidate to see detailed output
- Check for runtime errors, edge cases, and metric-specific failures
- Use the error information to guide the next attempt

## Completion Criteria

Output `<promise>EVOLUTION_TARGET_REACHED</promise>` ONLY when:
- The target score has been genuinely achieved (visible in submission results)
- OR the theoretical maximum has been reached
- OR no further improvement is possible (all metrics at ceiling)

NEVER output the promise tag falsely to exit the loop. The system is designed to continue until genuine completion. If you believe you are stuck, try a different strategy rather than giving up.

## Rules (NEVER violate these)

1. **NEVER modify the evaluator** -- the evaluator defines the fitness landscape. Modifying it is cheating and will invalidate the entire evolution run.
2. **NEVER modify state files directly** -- all files under `.claude/evolve-state/` are managed by the CLI. Direct modification will corrupt the state.
3. **ALWAYS submit through `claude-evolve submit`** -- this ensures proper evaluation, database storage, and iteration tracking.
4. **ALWAYS validate before submitting** -- a syntactically invalid candidate wastes an iteration.
5. **NEVER lie about scores or completion** -- the system relies on honest self-reporting. Fabricated results corrupt the evolutionary process.
6. **Respect the iteration budget** -- if max iterations is set, make each iteration count. Do not waste iterations on trivial changes.
7. **Explore diversity** -- do not always exploit the same approach. The MAP-Elites system rewards diversity across feature dimensions. Sometimes a lower-scoring but novel approach is more valuable than a marginal improvement to the current best.
