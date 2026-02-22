# Experiment Plan: Baseline vs Isolations vs Combination

This plan operationalizes `docs/notes.md` into four controlled experiment arms.

Implementation baseline code path: `baselines/nanogpt/`.

## Objective

Measure whether RAD-TSH and DSA are complementary under matched controls:

- **RAD-TSH (`2602.11374`)**: retrieval-aware head selection in hybrid Transformer-SSM (`k_h`) where non-retained heads are replaced by SSM recurrence; include optional state-size sensitivity (`d_state`).
- **DSA (`2512.02556`)**: token-level sparse attention (`k_t`) with context-management policy effects that must be tracked separately.

Scope note: we include the DSA mechanism from `2512.02556`, not to claim full-system parity against the entire source-paper training pipeline.

This plan focuses on mechanism-level effects, not full-pipeline leaderboard comparison.

## Arms

1. **Baseline**
   - No RAD-TSH or DSA mechanism.
   - Use shared control path and fixed evaluation protocol.

2. **RAD-TSH Isolation**
   - Enable RAD-TSH head-selection path (`k_h` sweep).
   - Split runs into `rad_mode=selection_only` vs `rad_mode=full_ssm` to isolate SSM-replacement effect.
   - Keep token selection disabled.

3. **DSA Isolation**
   - Enable token selection only (`k_t` sweep).
   - Keep head-selection mechanism disabled.
   - Report context-policy variant per run (for example, summary/discard variants if applicable).

4. **Combination (RAD-TSH + DSA)**
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

- **RAD-TSH knob**: `k_h`
- **DSA knob**: `k_t`
- Optional RAD-TSH sensitivity knob: `d_state`
- RAD-TSH mode label: `rad_mode` in `{selection_only, full_ssm}`
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
- RAD-TSH only: `k_h` sweep x `rad_mode` x seeds
- DSA only: `k_t` sweep x seeds
- Combined: `(k_h, k_t)` grid x seeds

Suggested first pass (grounded to `docs/notes.md` variables):

- RAD-TSH sweeps: include `k_h=0` control and at least one retrieval-preserving `k_h` setting for both `selection_only` and `full_ssm`.
- DSA sweeps: include low/medium/high `k_t` settings with the same evaluation protocol.
- Combined sweeps: reuse the RAD-TSH/DSA candidate settings before widening the grid.

Use `experiments/run-manifest.template.yaml` to register every run.

Baseline entry command:

`python baselines/nanogpt/train.py baselines/nanogpt/config/train_fineweb10B.py`

Implementation note:

- `baselines/nanogpt/` is baseline-only and has no SSM recurrence path.
- RAD-TSH `full_ssm` runs require adding a Transformer-SSM hybrid implementation track.
