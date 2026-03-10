# Session 16 Handoff (2026-03-10)

## Critical Bug Found & Fixed

**MHA silently ignored RoPE from BlockConfig** (issue #227, fixed in PR #228).

`ConfigurableBlock._build_attention()` passed `rope=block_config.rope` to GQA
but NOT to MHA. All "rope+mha" experiments had NO positional encoding.

The 10.9x "super-additive synergy" between RoPE and GQA was an **artifact**
of this confound. With the corrected code:

| Variant | BPB | vs baseline |
|---------|-----|-------------|
| rope+gqa | 3.140 | -12.5% (best) |
| rope+mha | 3.191 | -11.0% |
| alibi+gqa | 3.237 | -9.8% |
| alibi+mha | 3.283 | -8.5% |
| none+gqa | 3.572 | -0.4% |
| none+mha | 3.587 | baseline |
| learned+gqa | 3.583 | -0.1% |
| learned+mha | 3.602 | +0.4% (worst) |

**RoPE is the dominant factor (+11%).** GQA adds <2%. Synergy is ~1.1x (additive).

## Merged This Session
- PR #226: Self-repair strategy added to HumanEval experiment
- PR #228: RoPE/ALiBi support for MHA, ALiBi for GQA, 14 new tests

## Open Issues
- **#219** (RoPE+GQA): Updated with correction. Deposit pattern needs re-test.
- **#218** (Variance decomposition): Untouched. Good research question.
- **#30** (GitHub Pages): Long-term.

## Next Steps (Priority Order)

### 1. Re-test deposit pattern (high value, ~2 hours CPU)
The mechanistic analysis (`experiments/rope_gqa_mechanism.py`) ran with buggy
code. The deposit pattern (12.3x head concentration in rope+gqa) might also
be an artifact. Re-run with corrected MHA to check if rope+mha shows it too.

### 2. Close #219 with honest write-up
Summarize findings, corrections, and lessons learned.

### 3. Variance decomposition (#218, ~6 hours CPU)
Factorial: 5 seeds x 5 data orderings x 5 HP configs = 125 runs.
Novel research question at toy scale.

### 4. GPU eval of HumanEval (needs GPU)
164 problems x 5 strategies. Script ready: `experiments/humaneval_scaling.py`.

### 5. New research: Why RoPE >> ALiBi?
Both encode relative position. RoPE is multiplicative (rotation), ALiBi is
additive (bias). Does the multiplicative structure matter? Cleaner story than
the confounded synergy narrative.

## Key Files
- `experiments/rope_gqa_pe_ablation.py` -- corrected 8-variant experiment
- `experiments/rope_gqa_mechanism.py` -- deposit pattern analysis (needs re-run)
- `experiments/humaneval_scaling.py` -- HumanEval with 5 strategies
- `experiments/results/rope_gqa_pe_ablation.json` -- results (gitignored)

## Lesson Learned
Always verify the treatment is actually applied. Silent no-ops in config-driven
systems produce results that look reasonable but are meaningless. Be suspicious
of huge effect sizes (10.9x) -- they often indicate a bug, not a discovery.
