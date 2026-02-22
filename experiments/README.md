# Experiments

This directory tracks run definitions and outputs for four study arms:

1. `baseline`
2. `paper1_isolation`
3. `paper2_isolation`
4. `combination`

The goal is mechanism isolation and complementarity testing, not direct full-pipeline leaderboard claims.

Baseline training code for all arms lives in `baselines/nanogpt/`.

## Run Naming Convention

Use this run id format:

`<arm>__kh-<value_or_na>__kt-<value_or_na>__ds-<value_or_na>__cp-<policy_or_na>__seed-<n>__date-YYYYMMDD`

Examples:

- `baseline__kh-na__kt-na__ds-na__cp-na__seed-1__date-20260222`
- `paper1_isolation__kh-10__kt-na__ds-64__cp-na__seed-1__date-20260222`
- `paper2_isolation__kh-na__kt-256__ds-na__cp-discard-all__seed-1__date-20260222`
- `combination__kh-10__kt-256__ds-64__cp-discard-all__seed-1__date-20260222`

## What to Log Per Run

- Arm, seed, and mechanism toggles
- `k_h`, `k_t`, `L`, optional `d_state`, and context-policy label
- Cost metrics: memory/tokens/latency
- Quality metrics: retrieval/coverage and benchmark slice metrics
- Notes on deviations from planned controls
- Explicit label if a run is non-comparable to another run

Use `run-manifest.template.yaml` for every run to keep reports comparable.
