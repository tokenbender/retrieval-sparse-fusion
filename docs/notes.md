# Retrieval-Aware Distillation vs DSA: A Comparative Study

**RAD-TSH:** `2602.11374.pdf` - *Retrieval-Aware Distillation for Transformer-SSM Hybrids*  
**DSA:** `2512.02556.pdf` - *Token-level sparse attention mechanism study*  
**arXiv:** https://arxiv.org/abs/2602.11374 and https://arxiv.org/abs/2512.02556

---

## TL;DR

- **Problem:** both papers address long-context efficiency limits, but at different layers of the stack.
- **RAD-TSH idea:** keep only retrieval-critical attention heads (`k_h`) and replace other heads with SSM recurrence.
- **DSA idea:** use DSA to select top-`k_t` tokens per query and run sparse attention there.
- **Evidence:** RAD-TSH reports retrieval-heavy coverage jump 49.2 -> 95.0 at `k_h=10`; DSA reports MMLU-Pro 85.0, GPQA 82.4, AIME 93.1, SWE-Verified 73.1.
- **Efficiency signal:** RAD-TSH reports 0.8/5.7/11.0 MB memory at L=128/2048/4096; DSA reports context-management gains on BrowseComp from 51.4 to 67.6.
- **Caveat:** direct leaderboard-style comparison is invalid because training pipelines, benchmarks, and scope differ (retrieval-focused hybrid distillation vs broad end-to-end open-model system).

---

## Beginner's Guide

### Setup

These papers answer different versions of one question: how can we avoid paying dense-attention cost everywhere while keeping useful long-context behavior?

- **RAD-TSH (head-level selection):** preserve a few attention heads that do retrieval, convert the rest to recurrent heads.
- **DSA (token-level selection):** keep attention, but only on selected tokens via an indexer and top-`k_t` filtering.

### Core equation / intuition

Symbols used in this note:

- `L`: sequence length
- `k_h`: number of retained attention heads (RAD-TSH)
- `k_t`: number of selected tokens per query (DSA)
- `I_{t,s}`: index score of token `s` for query `t`

DSA's selection form:

```math
I_{t,s}=\sum_{j=1}^{H_I} w^I_{t,j}\,\mathrm{ReLU}\left((q^I_{t,j})^\top k^I_s\right),
\quad
u_t = \mathrm{Attn}\left(h_t,\{c_s\mid I_{t,s}\in \mathrm{Top}\text{-}k_t(I_{t,:})\}\right)
```

RAD-TSH intuition: select important **heads** once (architecture-level).  
DSA intuition: select important **tokens** dynamically per query.

### Why the naive method fails

- Uniform head placement in hybrids can miss retrieval-critical heads (RAD-TSH).
- Dense attention over all history tokens is expensive for long contexts and agent trajectories (DSA).

### Toy example

For a 100K-token reasoning trace, most past tokens are irrelevant at each step. Token selection (DSA) lowers compute. If retrieval function is concentrated in a few heads, head selection (RAD-TSH) can keep retrieval with minimal attention budget.

---

## Paper Summary

### Core Problem (Quick Recap)

- **RAD-TSH:** SSM-heavy hybrids are efficient but underperform on retrieval-heavy behavior unless key attention structure is preserved.
- **DSA:** open models need long-context efficiency plus stronger reasoning/agentic behavior under realistic deployment constraints.

### Key Contributions

1. **RAD-TSH:** retrieval-aware head ranking by synthetic KV-retrieval ablation.
2. **RAD-TSH:** selective hybrid distillation retaining only top `k_h` heads.
3. **DSA:** DSA (indexer + top-`k_t` selection under MLA/MQA).
4. **DSA:** large-scale continued pretraining + RL + synthetic agentic task pipeline.

### Results

- **RAD-TSH:** retrieval-heavy COV 49.2 (`k_h=0`) -> 95.0 (`k_h=10`) on Llama hybrid; Qwen trend 52.1 -> 96.4.
- **RAD-TSH:** memory row reported as 0.8/5.7/11.0 MB (L=128/2048/4096) for retrieval-aware setting.
- **DSA:** source paper reports MMLU-Pro 85.0, GPQA 82.4, AIME 93.1, SWE-Verified 73.1, MCP-Universe 45.9.
- **DSA:** context-management section reports BrowseComp branch from 51.4 to 67.6.

### Experiments in the Papers

| Method | Experiment | Tasks/Benchmarks | Models/Baselines | Metrics | Key outcome |
|---|---|---|---|---|---|
| RAD-TSH | Retained-head sweep (Table 1) | retrieval-heavy + knowledge-focused groups | Hybrid-Llama, Hybrid-Qwen vs teachers | KV-Ret, SWDE, COV | retrieval-heavy COV rises sharply by `k_h=10` |
| RAD-TSH | Memory analysis (Table 2) | sequence lengths 128/2048/4096 | retrieval-aware vs layer-wise 25/50% | memory (MB) | retrieval-aware shows lowest memory row |
| RAD-TSH | State-size ablation (Table 4) | retrieval-heavy group with top-20 heads fixed | `d_state` in {64,8,4} | COV | 95.8 -> 90.0 -> 81.0 trend |
| RAD-TSH | Head localization (Table 6/5.5) | single-head ablation in student | retained attention vs SSM heads | drop sensitivity | retrieval burden concentrated in retained attention heads |
| DSA | Main benchmark table (Table 2) | reasoning/coding/agentic suite | source model vs closed/open models | EM/Pass@1/rating/resolved/success | strong broad benchmark profile |
| DSA | Efficiency table (Table 3) | reasoning-heavy benchmarks | source model vs Speciale and peers | score + token count | better scores often come with higher token use |
| DSA | Competition outcomes (Table 4) | IMO/CMO/IOI/ICPC | Speciale | score + medal | reported gold-level outcomes |
| DSA | Synthetic task transfer (Table 5/Fig 5) | synthesized general-agent tasks | source-model experiments + stronger proprietary baselines | Pass@K + transfer behavior | synthetic tasks are hard; RL transfer improvements reported |
| DSA | Context management (Sec 4.4/Fig 6) | BrowseComp-style search-agent regime | Summary/Discard-75/Discard-all/parallel baseline | score + steps | 67.6 branch via discard-all strategy |

### Side-by-Side Canonical Comparison

| Axis | RAD-TSH | DSA |
|---|---|---|
| Selection unit | heads | tokens per query |
| Main knob | `k_h` | `k_t` |
| Architectural target | retrieval-preserving hybridization | sparse long-context attention |
| Optimization coupling | distillation-centric | pretraining + RL + data synthesis |
| Evaluation center | retrieval + memory | broad capability + agentic benchmarks |

---

## Deep Dive

### Key definitions

- **G&A heads (RAD-TSH):** heads implicated in gather-and-aggregate retrieval behavior.
- **Retrieval-critical heads:** top-ranked by ablation impact on synthetic KV retrieval.
- **DSA indexer (DSA):** lightweight scoring mechanism for token selection.
- **Context management (DSA):** test-time policy for overflow trajectories (Summary/Discard variants).

### Derivations (if any)

RAD-TSH base operators:

```math
\mathrm{Attn}(X)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\quad
h_t=A_t h_{t-1}+B_t x_t,
\quad
y_t=C_t h_t + D_t x_t
```

RAD-TSH distillation view (compressed):

```math
\min \|M_T^{(l)}-M_S^{(l)}\|_F,
\quad
\min \mathcal{L}_{CE}(\text{teacher},\text{student})
```

DSA token-selection equations:

```math
I_{t,s}=\sum_{j=1}^{H_I} w^I_{t,j}\,\mathrm{ReLU}\left((q^I_{t,j})^\top k^I_s\right),
\quad
u_t = \mathrm{Attn}\left(h_t,\{c_s\mid I_{t,s}\in \mathrm{Top}\text{-}k_t(I_{t,:})\}\right)
```

DSA alignment + RL objective family (compressed):

```math
\mathcal{L}_I=\sum_t D_{KL}(p_{t,:}\|\mathrm{Softmax}(I_{t,:})),
\quad
\mathcal{J}_{GRPO}(\theta)=\text{clipped policy objective} - \beta\,D_{KL}
```

### RAD-TSH training flow

This is the exact flow for RAD-TSH and where distillation enters:

1. **Start with a pretrained Transformer teacher.**
2. **Rank attention heads by retrieval importance** using synthetic KV-retrieval ablation, then sort heads by performance drop.
3. **Build a hybrid student:** keep top `k_h` heads as attention and replace non-retained heads with SSM heads (basis: DISCRETEMAMBA-2 in the paper text).
4. **Run distillation training:**
   - Original MOHAWK has 3 stages (matrix orientation -> hidden-state alignment -> weight transfer + KD).
   - RAD-TSH adaptation skips matrix orientation because critical heads are copied from teacher.
   - Training proceeds with hidden-state alignment and knowledge distillation.
5. **Sweep and evaluate** (`k_h`, optional `d_state`) on retrieval/cost metrics under matched controls.

DSA flow is separate: index tokens -> select Top-`k_t` -> sparse attention -> broader pretraining/RL pipeline.

### Evidence from RAD-TSH paper

- **Head ranking (Sec 4.1):** "ablate each attention head ... and measure ... drop ... This provides a sorted list of heads, from most to least critical".
- **Hybrid construction (Sec 4.2):** "Retained heads ... kept unchanged" and "Replaced heads ... replaced ...".
- **Feature alignment (Sec 4.2):** "Chunks assigned to retained heads are processed by their original attention operators, while all other chunks are processed by the SSM operators ... producing a single mixing output with the same interface as the teacher."
- **SSM basis (Sec 3.3):** "DISCRETEMAMBA-2 ... we use it as the basis for our SSM replacements."
- **MOHAWK stages (Sec 3.3):** "The pipeline optimizes the student parameters ... across three stages" followed by "Matrix Orientation", "Hidden-State Alignment", and "Weight Transfer & Knowledge Distillation".
- **RAD-TSH adaptation (Sec 5.3):** "Since critical heads are copied from teacher, matrix orientation is skipped; training proceeds via hidden-state alignment and knowledge distillation."
- **Related-work boundary:** Zamba and Jamba appear in the references list, not as the method backbone declaration.

### Pseudocode (optional)

```python
# RAD-TSH: head selection + hybrid build + distill
scores = ablate_each_head_and_measure_drop(teacher)
keep_heads = top_k(scores, k_h)
student_a = build_hybrid(keep_heads=keep_heads)
distill(student_a, teacher)

# DSA: token selection
for query in sequence:
    idx_scores = indexer(query, history)
    sel = top_k(idx_scores, k_t)
    out = sparse_attention(query, history[sel])
train_with_pretraining_and_grpo(student_b)
```

## Zero-Ambiguity Execution Contract

### Scope

- This repo compares **mechanisms**, not full training pipelines.
- Methods in scope: `RAD-TSH` (head retention + SSM replacement) and `DSA` (token top-`k_t` routing).
- Source of truth for method/evidence: this file and `docs/experiment-plan.md`.

### Required Experiment Arms

| Arm | `rad_mode` | DSA path | Purpose |
|---|---|---|---|
| `baseline` | `off` | `off` | shared control |
| `rad_tsh_isolation` | `selection_only` or `full_ssm` | `off` | isolate head/SSM effects |
| `dsa_isolation` | `off` | `on` | isolate token-routing effects |
| `combination` | `full_ssm` (or tested RAD mode) | `on` | test complementarity |

### Required Knobs and Meanings

- `k_h`: retained attention head count for RAD-TSH.
- `d_state`: SSM state size for RAD-TSH replaced heads.
- `k_t`: selected token count per query for DSA.
- `L`: evaluated sequence length setting.
- `context_policy`: explicit label for context-management policy in DSA-related runs.

### RAD-TSH Distillation Flow (non-negotiable)

1. Rank heads by synthetic KV-retrieval ablation (Sec 4.1).
2. Keep top `k_h` attention heads and replace non-retained heads with SSM heads (Sec 4.2).
3. Use DiscreteMamba-2 basis for SSM replacements (Sec 3.3).
4. Train with MOHAWK adaptation: matrix orientation skipped; hidden-state alignment + knowledge distillation are used (Sec 5.3).

### Comparison Validity Rules

- Do not claim raw leaderboard superiority across methods without harmonized budgets and benchmark overlap.
- Pair every quality metric claim with cost metrics (memory/tokens/compute).
- Mark non-comparable results explicitly.

### Definition of Done for First Pilot

- Run at least one successful seed for each arm.
- Include `k_h=0` and one retrieval-preserving `k_h` setting in RAD-TSH runs.
- Include low/medium/high `k_t` in DSA runs.
- Log all runs in `experiments/run-manifest.template.yaml` fields.

### Code-Level Mapping (what must be implemented where)

- `baselines/nanogpt/model.py`
  - add `rad_mode` switch and routing logic for retained heads vs replaced heads.
  - `selection_only`: non-retained head slots are no-op/zeroed.
  - `full_ssm`: non-retained head slots use SSM heads with `d_state`.
  - output tensor shape and residual interface must remain identical to baseline.
- `baselines/nanogpt/train.py`
  - expose and log run knobs: `rad_mode`, `k_h`, `d_state`, `k_t`, `L`, `context_policy`.
  - keep optimizer/scheduler/eval protocol matched across arms.
- `baselines/nanogpt/config/*.py`
  - define arm-specific configs only through documented knobs.
  - avoid hidden architecture/training changes across arms.
- `experiments/run-manifest.template.yaml`
  - each run must record arm, mechanism toggles, knobs, controls, and quality+cost metrics.

### Hard Invariants (must stay true)

1. `baseline` must be behavior-identical to pure baseline path (`rad_mode=off`, DSA off).
2. `rad_tsh_isolation` must not enable DSA.
3. `dsa_isolation` must not enable RAD head/SSM routing.
4. `combination` is the only arm allowed to enable both RAD-TSH and DSA.
5. Any result without matched controls is marked non-comparable.

### Explicitly Out of Scope (to prevent drift)

- Reproducing full source-paper RL/post-training pipelines for DSA.
- Cross-paper leaderboard claims without harmonized benchmarks and budgets.
- Untracked architectural edits not captured in run manifest knobs/notes.

---

## Critical Analysis

### Strongest objection

This comparison can falsely imply parity between mechanisms when DSA gains are produced by a much broader system recipe (architecture + large RL/data/training choices), while RAD-TSH is a narrower architecture-distillation intervention.

### When it's wrong / when it's right

- **Right:** mechanism-level comparison (selection axis, memory/compute behavior, failure modes).
- **Wrong:** raw leaderboard comparison without harmonized training budget and benchmark overlap.

### Failure modes

1. Interpreting `k_h` and `k_t` as interchangeable controls.
2. Attributing all DSA gains to DSA instead of full pipeline.
3. Ignoring synthetic-probe dependency in RAD-TSH head ranking.
4. Ignoring context-policy dependence for DSA search-agent outcomes.

### Baseline vs weighting vs selection effects

Both papers are primarily **selection** mechanisms. Weighting/regularization terms exist, but the core intervention is what gets attention budget (heads vs tokens).

### Experiments to validate/refute

1. Keep training budget fixed and compare head-selection-only vs token-selection-only interventions.
2. Replace RAD-TSH's single synthetic retrieval probe with multiple retrieval probes and check retained-head stability.
3. Evaluate DSA with frozen RL/data recipe to isolate architecture contribution.
4. Use shared benchmark slices with matched compute and prompt protocol.

### Mitigations

- Pair every score claim with a cost claim (tokens/memory/compute).
- Separate mechanism claims from system-level claims.
- Mark non-comparable metrics explicitly in comparative sections.

### Author-stated limitations

- **RAD-TSH:** <3B evaluation scope, synthetic-probe dependence, incomplete retrieval/SSM decoupling.
- **DSA:** knowledge breadth gap to frontier closed models, token-efficiency gap, and remaining deficits on hardest tasks.

---

## Learning Path

- **Prereqs:** attention internals, SSM recurrence, sparse attention, distillation, policy-gradient RL.
- **Learn next:**
  1. MOHAWK distillation and hybrid model construction details.
  2. MLA/MQA implementation mechanics for DSA.
  3. Cost-aware test-time compute scaling and context-management policy design.
- **Diagnostics:**
  1. Track retrieval-heavy COV vs `k_h` and `d_state` in RAD-TSH.
  2. Track benchmark score vs token output and context policy in DSA.
  3. Keep direct-comparison claims restricted to harmonized settings.

---

## Key Equations Summary

| Method | Equation | Meaning |
|---|---|---|
| RAD-TSH | `Attn(X)=softmax(QK^T/sqrt(d_k))V` | global token mixing via attention |
| RAD-TSH | `h_t=A_t h_{t-1}+B_t x_t`, `y_t=C_t h_t + D_t x_t` | recurrent state update/output |
| RAD-TSH | `min ||M_T^{(l)}-M_S^{(l)}||_F` | mixer alignment objective |
| RAD-TSH | `min L_CE(teacher, student)` | final distillation objective |
| DSA | `I_{t,s}=sum_j w^I_{t,j}ReLU((q^I_{t,j})^T k^I_s)` | indexer score per query-token pair |
| DSA | `u_t=Attn(h_t,{c_s | I_{t,s} in Top-k_t(I_{t,:})})` | sparse attention over selected tokens |
| DSA | `L_I=sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))` | indexer alignment loss |
| DSA | `J_GRPO = clipped objective - beta*KL` | RL post-training control objective |

---

## References

1. Bick, A., Xing, E. P., Gu, A. *Retrieval-Aware Distillation for Transformer-SSM Hybrids*. arXiv:2602.11374 (2026). https://arxiv.org/abs/2602.11374
2. Source-paper authors et al. arXiv:2512.02556 (2025). https://arxiv.org/abs/2512.02556
3. Local PDF artifacts for both arXiv IDs used in this note.
4. OCR extraction artifacts for both arXiv IDs used for evidence anchoring.
