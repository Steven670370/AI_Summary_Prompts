# Performance Tests

This folder contains tests and benchmarks to evaluate the token and compute efficiency of the similarity-based routing system.

## Files

- `test_performance.py` - Unit tests for routing logic
- `benchmark.py` - Real-world simulation and cost analysis

## Running Tests

```bash
cd Performance_Tests
python test_performance.py
python benchmark.py
```

## Quick Summary

### Can This System Reduce Token Usage?

**Yes**, in the right scenarios:

| Scenario | Token Reduction | API Call Reduction |
|----------|-----------------|-------------------|
| High repetition (FAQ) | 50-80% | 50-80% |
| Mixed complexity | 20-40% | 20-40% |
| Low repetition (random) | 0-10% | 0-10% |

### When Does It Help?

**Advantages:**

1. **FAQ & Repetitive Queries**
   - Repeated questions (password resets, common issues)
   - Office productivity: "Where is X?", "How do I Y?"
   - Customer support: Standard troubleshooting

2. **High Similarity Matches**
   - When `SIMILARITY_THRESHOLD ≥ 0.90` matches existing DB entries
   - Zero API calls for direct matches
   - 100% token savings on DB hits

3. **Complex Query Decomposition**
   - Long answers (>200 words) split into sub-questions
   - Each sub-answer uses fewer tokens
   - Better for user comprehension

4. **Cost Savings at Scale**
   - 10K queries/day → ~$5-15/day savings
   - 100K queries/day → ~$50-150/day savings

**Disadvantages:**

1. **Cold Start Problem**
   - Requires ≥100 DB records before similarity kicks in
   - Initial deployment has no benefit

2. **Low Repetition Scenarios**
   - Random unique queries
   - Creative tasks
   - Novel domains

3. **Similarity Computation Overhead**
   - ~10-50ms per query for embedding computation
   - O(n) comparison where n = DB size
   - May exceed benefit for very short queries

4. **Decomposition Overhead**
   - Extra API calls for decompose + combine prompts
   - May increase total calls for simple questions
   - Only beneficial when answer is truly complex

## Detailed Analysis

### API Call Reduction

```
Baseline (no routing):     1 API call per query

With similarity routing:
├── DB hit (sim ≥ 0.90):   0 API calls (100% reduction)
├── Complex (len > 200):   3-8 API calls (decomposition)
└── Simple (len ≤ 200):    1 API call (no reduction)
```

### Token Reduction

```
Scenario: Office FAQ (100 queries/day, 50% repeated)

Without routing:
  100 queries × 100 tokens × $0.60/1M = $0.006/day

With routing (50% DB hits):
  50 queries × 100 tokens × $0.60/1M = $0.003/day
  50 queries × 0 tokens (DB) = $0

Daily savings: ~$0.003 (50%)
Monthly savings: ~$0.09
```

### Compute Cost Breakdown

| Component | Time | When |
|-----------|------|------|
| DB check | ~1ms | Always |
| Similarity computation | ~10-50ms | When DB ≥ 100 |
| API call | ~500-2000ms | Per non-DB query |

## Recommendations

1. **Best Fit**: FAQ systems, customer support, repetitive documentation
2. **Avoid For**: Creative writing, one-off questions, novel domains
3. **Threshold Tuning**: Lower threshold (0.80) = more hits but less accurate
4. **DB Quality**: Higher rated responses (≥4) give better matches

## Monitoring Metrics

Track these to evaluate effectiveness:

- `db_direct_rate`: % of queries answered from DB
- `decomposed_rate`: % of queries decomposed
- `avg_similarity`: Average similarity score of matches
- `token_savings`: Estimated tokens saved vs baseline
