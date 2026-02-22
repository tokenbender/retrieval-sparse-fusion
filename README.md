# retrieval-sparse-fusion

This repository is a study workspace for mechanism-level long-context experiments across two complementary papers documented in `docs/notes.md`:

- Paper 1 (`2602.11374`): retrieval-aware head selection in Transformer-SSM hybrids; retain critical heads (`k_h`) and replace others with SSM recurrence.
- Paper 2 (`2512.02556`): DeepSeek Sparse Attention (DSA) token selection with top-`k_t` routing (plus broader training/runtime pipeline context in the paper).

The core goal is to run and compare four conditions under a fair protocol:

1. Baseline shared control (no Paper 1/Paper 2 selection mechanism enabled).
2. Paper 1 isolation (head selection only, sweep `k_h`, optional `d_state`).
3. Paper 2 isolation (token selection only, sweep `k_t`, include context-policy reporting).
4. Paper 1 + Paper 2 combination (joint `k_h` and `k_t` to test complementarity).

## Repository Layout

- `docs/notes.md`: comparative technical study and evidence from both papers.
- `docs/experiment-plan.md`: baseline vs isolation vs combination protocol.
- `experiments/README.md`: run organization and naming conventions.
- `experiments/run-manifest.template.yaml`: per-run metadata template.
- `baselines/nanogpt/`: pure non-character NanoGPT baseline training path.

## Baseline Training

Baseline training entrypoint:

`baselines/nanogpt/train.py config/train_fineweb10B.py`

See `baselines/nanogpt/README.md` for run commands and dataset download.

## Experiment Intent

This repo is not trying to declare raw leaderboard winners between unrelated full pipelines.
It is designed to isolate mechanism effects (head selection vs token selection), pair quality with cost, and test whether the two mechanisms are complementary under matched controls.

Direct benchmark-level claims across papers are treated as non-comparable unless training budget, benchmark slices, and evaluation protocol are harmonized.
