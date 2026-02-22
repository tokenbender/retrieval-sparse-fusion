# Retrieval-Aware Distillation vs DeepSeek Sparse Attention: A Comparative Study

**Paper A:** `2602.11374.pdf` - *Retrieval-Aware Distillation for Transformer-SSM Hybrids*  
**Paper B:** `2512.02556.pdf` - *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*  
**arXiv:** https://arxiv.org/abs/2602.11374 and https://arxiv.org/abs/2512.02556

---

## TL;DR

- **Problem:** both papers address long-context efficiency limits, but at different layers of the stack.
- **Paper A idea:** keep only retrieval-critical attention heads (`k_h`) and replace other heads with SSM recurrence.
- **Paper B idea:** use DSA to select top-`k_t` tokens per query and run sparse attention there.
- **Evidence:** Paper A reports retrieval-heavy coverage jump 49.2 -> 95.0 at `k_h=10`; Paper B reports MMLU-Pro 85.0, GPQA 82.4, AIME 93.1, SWE-Verified 73.1.
- **Efficiency signal:** Paper A reports 0.8/5.7/11.0 MB memory at L=128/2048/4096; Paper B reports context-management gains on BrowseComp from 51.4 to 67.6.
- **Caveat:** direct leaderboard-style comparison is invalid because training pipelines, benchmarks, and scope differ (retrieval-focused hybrid distillation vs broad end-to-end open-model system).

---

## Beginner's Guide

### Setup

These papers answer different versions of one question: how can we avoid paying dense-attention cost everywhere while keeping useful long-context behavior?

- **Paper A (head-level selection):** preserve a few attention heads that do retrieval, convert the rest to recurrent heads.
- **Paper B (token-level selection):** keep attention, but only on selected tokens via an indexer and top-`k_t` filtering.

### Core equation / intuition

Symbols used in this note:

- `L`: sequence length
- `k_h`: number of retained attention heads (Paper A)
- `k_t`: number of selected tokens per query (Paper B)
- `I_{t,s}`: index score of token `s` for query `t`

Paper B's selection form:

```math
I_{t,s}=\sum_{j=1}^{H_I} w^I_{t,j}\,\mathrm{ReLU}\left((q^I_{t,j})^\top k^I_s\right),
\quad
u_t = \mathrm{Attn}\left(h_t,\{c_s\mid I_{t,s}\in \mathrm{Top}\text{-}k_t(I_{t,:})\}\right)
```

Paper A intuition: select important **heads** once (architecture-level).  
Paper B intuition: select important **tokens** dynamically per query.

### Why the naive method fails

- Uniform head placement in hybrids can miss retrieval-critical heads (Paper A).
- Dense attention over all history tokens is expensive for long contexts and agent trajectories (Paper B).

### Toy example

For a 100K-token reasoning trace, most past tokens are irrelevant at each step. Token selection (Paper B) lowers compute. If retrieval function is concentrated in a few heads, head selection (Paper A) can keep retrieval with minimal attention budget.

---

## Paper Summary

### Core Problem (Quick Recap)

- **Paper A:** SSM-heavy hybrids are efficient but underperform on retrieval-heavy behavior unless key attention structure is preserved.
- **Paper B:** open models need long-context efficiency plus stronger reasoning/agentic behavior under realistic deployment constraints.

### Key Contributions

1. **Paper A:** retrieval-aware head ranking by synthetic KV-retrieval ablation.
2. **Paper A:** selective hybrid distillation retaining only top `k_h` heads.
3. **Paper B:** DeepSeek Sparse Attention (indexer + top-`k_t` selection under MLA/MQA).
4. **Paper B:** large-scale continued pretraining + RL + synthetic agentic task pipeline.

### Results

- **Paper A:** retrieval-heavy COV 49.2 (`k_h=0`) -> 95.0 (`k_h=10`) on Llama hybrid; Qwen trend 52.1 -> 96.4.
- **Paper A:** memory row reported as 0.8/5.7/11.0 MB (L=128/2048/4096) for retrieval-aware setting.
- **Paper B:** DeepSeek-V3.2 reports MMLU-Pro 85.0, GPQA 82.4, AIME 93.1, SWE-Verified 73.1, MCP-Universe 45.9.
- **Paper B:** context-management section reports BrowseComp branch from 51.4 to 67.6.

### Experiments in the Papers

| Paper | Experiment | Tasks/Benchmarks | Models/Baselines | Metrics | Key outcome |
|---|---|---|---|---|---|
| A | Retained-head sweep (Table 1) | retrieval-heavy + knowledge-focused groups | Hybrid-Llama, Hybrid-Qwen vs teachers | KV-Ret, SWDE, COV | retrieval-heavy COV rises sharply by `k_h=10` |
| A | Memory analysis (Table 2) | sequence lengths 128/2048/4096 | retrieval-aware vs layer-wise 25/50% | memory (MB) | retrieval-aware shows lowest memory row |
| A | State-size ablation (Table 4) | retrieval-heavy group with top-20 heads fixed | `d_state` in {64,8,4} | COV | 95.8 -> 90.0 -> 81.0 trend |
| A | Head localization (Table 6/5.5) | single-head ablation in student | retained attention vs SSM heads | drop sensitivity | retrieval burden concentrated in retained attention heads |
| B | Main benchmark table (Table 2) | reasoning/coding/agentic suite | DeepSeek-V3.2 vs closed/open models | EM/Pass@1/rating/resolved/success | strong broad benchmark profile |
| B | Efficiency table (Table 3) | reasoning-heavy benchmarks | V3.2 vs Speciale and peers | score + token count | better scores often come with higher token use |
| B | Competition outcomes (Table 4) | IMO/CMO/IOI/ICPC | Speciale | score + medal | reported gold-level outcomes |
| B | Synthetic task transfer (Table 5/Fig 5) | synthesized general-agent tasks | V3.2-Exp + stronger proprietary baselines | Pass@K + transfer behavior | synthetic tasks are hard; RL transfer improvements reported |
| B | Context management (Sec 4.4/Fig 6) | BrowseComp-style search-agent regime | Summary/Discard-75/Discard-all/parallel baseline | score + steps | 67.6 branch via discard-all strategy |

### Side-by-Side Canonical Comparison

| Axis | Paper A | Paper B |
|---|---|---|
| Selection unit | heads | tokens per query |
| Main knob | `k_h` | `k_t` |
| Architectural target | retrieval-preserving hybridization | sparse long-context attention |
| Optimization coupling | distillation-centric | pretraining + RL + data synthesis |
| Evaluation center | retrieval + memory | broad capability + agentic benchmarks |

---

## Deep Dive

### Key definitions

- **G&A heads (Paper A):** heads implicated in gather-and-aggregate retrieval behavior.
- **Retrieval-critical heads:** top-ranked by ablation impact on synthetic KV retrieval.
- **DSA indexer (Paper B):** lightweight scoring mechanism for token selection.
- **Context management (Paper B):** test-time policy for overflow trajectories (Summary/Discard variants).

### Derivations (if any)

Paper A base operators:

```math
\mathrm{Attn}(X)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\quad
h_t=A_t h_{t-1}+B_t x_t,
\quad
y_t=C_t h_t + D_t x_t
```

Paper A distillation view (compressed):

```math
\min \|M_T^{(l)}-M_S^{(l)}\|_F,
\quad
\min \mathcal{L}_{CE}(\text{teacher},\text{student})
```

Paper B token-selection equations:

```math
I_{t,s}=\sum_{j=1}^{H_I} w^I_{t,j}\,\mathrm{ReLU}\left((q^I_{t,j})^\top k^I_s\right),
\quad
u_t = \mathrm{Attn}\left(h_t,\{c_s\mid I_{t,s}\in \mathrm{Top}\text{-}k_t(I_{t,:})\}\right)
```

Paper B alignment + RL objective family (compressed):

```math
\mathcal{L}_I=\sum_t D_{KL}(p_{t,:}\|\mathrm{Softmax}(I_{t,:})),
\quad
\mathcal{J}_{GRPO}(\theta)=\text{clipped policy objective} - \beta\,D_{KL}
```

### Implementation sketch

1. **A:** ablate heads -> rank -> keep top `k_h` -> replace others with SSM heads -> distill.
2. **B:** index all prior tokens -> pick top `k_t` -> sparse attention -> continue pretraining and RL.
3. For fair comparison, isolate mechanism effects from training-pipeline effects.

### Pseudocode (optional)

```python
# A: head selection
scores = ablate_each_head_and_measure_drop(teacher)
keep_heads = top_k(scores, k_h)
student_a = build_hybrid(keep_heads=keep_heads)
distill(student_a, teacher)

# B: token selection
for query in sequence:
    idx_scores = indexer(query, history)
    sel = top_k(idx_scores, k_t)
    out = sparse_attention(query, history[sel])
train_with_pretraining_and_grpo(student_b)
```

---

## Critical Analysis

### Strongest objection

This comparison can falsely imply parity between mechanisms when Paper B gains are produced by a much broader system recipe (architecture + large RL/data/training choices), while Paper A is a narrower architecture-distillation intervention.

### When it's wrong / when it's right

- **Right:** mechanism-level comparison (selection axis, memory/compute behavior, failure modes).
- **Wrong:** raw leaderboard comparison without harmonized training budget and benchmark overlap.

### Failure modes

1. Interpreting `k_h` and `k_t` as interchangeable controls.
2. Attributing all Paper B gains to DSA instead of full pipeline.
3. Ignoring synthetic-probe dependency in Paper A head ranking.
4. Ignoring context-policy dependence for Paper B search-agent outcomes.

### Baseline vs weighting vs selection effects

Both papers are primarily **selection** mechanisms. Weighting/regularization terms exist, but the core intervention is what gets attention budget (heads vs tokens).

### Experiments to validate/refute

1. Keep training budget fixed and compare head-selection-only vs token-selection-only interventions.
2. Replace Paper A's single synthetic retrieval probe with multiple retrieval probes and check retained-head stability.
3. Evaluate Paper B DSA with frozen RL/data recipe to isolate architecture contribution.
4. Use shared benchmark slices with matched compute and prompt protocol.

### Mitigations

- Pair every score claim with a cost claim (tokens/memory/compute).
- Separate mechanism claims from system-level claims.
- Mark non-comparable metrics explicitly in comparative sections.

### Author-stated limitations

- **Paper A:** <3B evaluation scope, synthetic-probe dependence, incomplete retrieval/SSM decoupling.
- **Paper B:** knowledge breadth gap to frontier closed models, token-efficiency gap, and remaining deficits on hardest tasks.

---

## Learning Path

- **Prereqs:** attention internals, SSM recurrence, sparse attention, distillation, policy-gradient RL.
- **Learn next:**
  1. MOHAWK distillation and hybrid model construction details.
  2. MLA/MQA implementation mechanics for DSA.
  3. Cost-aware test-time compute scaling and context-management policy design.
- **Diagnostics:**
  1. Track retrieval-heavy COV vs `k_h` and `d_state` in A.
  2. Track benchmark score vs token output and context policy in B.
  3. Keep direct-comparison claims restricted to harmonized settings.

---

## Key Equations Summary

| Paper | Equation | Meaning |
|---|---|---|
| A | `Attn(X)=softmax(QK^T/sqrt(d_k))V` | global token mixing via attention |
| A | `h_t=A_t h_{t-1}+B_t x_t`, `y_t=C_t h_t + D_t x_t` | recurrent state update/output |
| A | `min ||M_T^{(l)}-M_S^{(l)}||_F` | mixer alignment objective |
| A | `min L_CE(teacher, student)` | final distillation objective |
| B | `I_{t,s}=sum_j w^I_{t,j}ReLU((q^I_{t,j})^T k^I_s)` | indexer score per query-token pair |
| B | `u_t=Attn(h_t,{c_s | I_{t,s} in Top-k_t(I_{t,:})})` | sparse attention over selected tokens |
| B | `L_I=sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))` | indexer alignment loss |
| B | `J_GRPO = clipped objective - beta*KL` | RL post-training control objective |

---

## References

1. Bick, A., Xing, E. P., Gu, A. *Retrieval-Aware Distillation for Transformer-SSM Hybrids*. arXiv:2602.11374 (2026). https://arxiv.org/abs/2602.11374
2. DeepSeek-AI et al. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556 (2025). https://arxiv.org/abs/2512.02556
3. Local PDF artifacts: `bick-deepseek/2602.11374.pdf`, `bick-deepseek/2512.02556.pdf`
4. OCR sources used for evidence anchoring: `bick-deepseek/2602.11374.ocr.feedback-loop.txt`, `bick-deepseek/2512.02556.ocr.feedback-loop.txt`
