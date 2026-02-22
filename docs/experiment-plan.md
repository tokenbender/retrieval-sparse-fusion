# Experiment Plan: Baseline vs Isolations vs Combination

This plan operationalizes `docs/notes.md` into four controlled experiment arms.

Implementation baseline code path: `baselines/nanogpt/`.

## Objective

Measure whether Paper 1 and Paper 2 mechanisms are complementary under matched controls:

- **Paper 1 (`2602.11374`)**: retrieval-aware head selection in hybrid Transformer-SSM (`k_h`) with optional state-size sensitivity (`d_state`).
- **Paper 2 (`2512.02556`)**: DeepSeek Sparse Attention token selection (`k_t`) with context-management policy effects that must be tracked separately.

This plan focuses on mechanism-level effects, not full-pipeline leaderboard comparison.

## Arms

1. **Baseline**
   - No Paper 1 or Paper 2 mechanism.
   - Use shared control path and fixed evaluation protocol.

2. **Paper 1 Isolation**
   - Enable head selection only (`k_h` sweep).
   - Keep token selection disabled.

3. **Paper 2 Isolation**
   - Enable token selection only (`k_t` sweep).
   - Keep head-selection mechanism disabled.
   - Report context-policy variant per run (for example, summary/discard variants if applicable).

4. **Combination (Paper 1 + Paper 2)**
   - Enable both `k_h` and `k_t`.
   - Test complementarity and tradeoffs.

## Controlled Variables (Must Match Across Arms)

- Random seed policy (fixed set of seeds across all arms)
- Train/validation/test splits
- Data preprocessing/tokenization
- Compute budget (same training steps/epochs and comparable hardware budget)
- Evaluation prompts/protocol
- Overlapping benchmark slice definition for cross-arm comparison

## Tunable Variables

- **Paper 1 knob**: `k_h`
- **Paper 2 knob**: `k_t`
- Optional Paper 1 sensitivity knob: `d_state`
- Sequence length setting(s): `L`
- Context-management policy label (reported, not hidden)

## Metrics to Report Per Run

- Retrieval/coverage metrics used in notes (`COV`, `KV-Ret`, optional `SWDE`)
- Memory usage (MB) and token cost
- Context-management outcomes (score and steps where relevant)
- Task scores used by the selected overlapping benchmark slice

Always pair quality metrics with cost metrics.

## Fairness Checklist

- Report mean and variance across multiple seeds.
- Keep the same tuning budget across arms.
- Avoid direct claims from non-overlapping benchmarks.
- Separate mechanism effects from full pipeline effects in conclusions.
- Mark any non-comparable result explicitly in tables and conclusions.

## Minimal Run Matrix

- Baseline: 1-2 seed runs
- Paper 1 only: `k_h` sweep x seeds
- Paper 2 only: `k_t` sweep x seeds
- Combined: `(k_h, k_t)` grid x seeds

Suggested first pass (grounded to `docs/notes.md` variables):

- Paper 1 sweeps: include `k_h=0` control and at least one retrieval-preserving `k_h` setting.
- Paper 2 sweeps: include low/medium/high `k_t` settings with the same evaluation protocol.
- Combined sweeps: reuse the Paper 1/Paper 2 candidate settings before widening the grid.

Use `experiments/run-manifest.template.yaml` to register every run.

Baseline entry command:

`python baselines/nanogpt/train.py baselines/nanogpt/config/train_fineweb10B.py`
