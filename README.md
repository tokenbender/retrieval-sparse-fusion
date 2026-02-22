# retrieval-sparse-fusion

This repository is a study workspace for mechanism-level long-context experiments across two complementary methods documented in `docs/notes.md`:

- **RAD-TSH** (Bick et al., arXiv:2602.11374): retrieval-aware head selection in Transformer-SSM hybrids; retain critical heads (`k_h`) and replace non-retained heads with SSM recurrence.
- **DSA** (arXiv:2512.02556): token-level sparse attention with top-`k_t` routing.

Why include DSA here: this repo is not reproducing the full source-paper system recipe. It uses DSA as a complementary token-selection axis against RAD-TSH head/SSM selection.

The core goal is to run and compare four conditions under a fair protocol:

1. Baseline shared control (no RAD-TSH or DSA mechanism enabled).
2. RAD-TSH isolation (explicit SSM track: compare `selection_only` vs `full_ssm`; sweep `k_h`, optional `d_state`).
3. DSA isolation (token selection only, sweep `k_t`, include context-policy reporting).
4. Combined run (RAD-TSH + DSA; joint `k_h` and `k_t` to test complementarity).

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

## RAD-TSH SSM Angle

For RAD-TSH, the mechanism is two-part:

- retain retrieval-critical heads (`k_h`)
- replace non-retained heads with SSM recurrence

In this repo today, the SSM angle is tracked in experiment docs/manifests; the current `baselines/nanogpt/` path is a pure GPT baseline and does not yet implement the Transformer-SSM hybrid path.

## Experiment Intent

This repo is designed to isolate mechanism effects (head selection vs token selection), pair quality with cost, and test whether the two mechanisms are complementary under matched controls.