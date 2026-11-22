# Results (22 November 2026)

Everything here was measured, not estimated. "Tricky" is the hard-negative Q&A test (gold answer against
its own up-to-10 Gemini wrong answers; CLM-8B publishes 69.2%). "General" is in-batch top-1 retrieval over
1,024-pair blocks with same-document pairs masked. Raw numbers: `results/innovations/`.

## 1. Checking the eval code against the authors' heads

Our `evals.py` on the published vectors (first 20,000 rows of `shard_00/0028`, 20,480 rows of `shard_05/0083`):

| Authors' head | Tricky (ours) | Tricky (published) | General |
|---|---|---|---|
| pre-trained | 51.3% | 52.1% | 83.6% |
| mid-trained | 68.8% | 69.2% | 83.9% |
| post-trained | 68.2% | 68.5% | 83.9% |

## 2. Quantization (idea 3)

**Backbone latency**, batch 1, one L40S, vLLM 0.11, mean of 30 runs (`backbone_latency_l40s.json`):

| Backbone | 64 tokens | 256 tokens | 1024 tokens |
|---|---|---|---|
| Qwen3-8B bf16 (CLM) | 21.1 ms | 22.0 ms | 24.3 ms |
| Qwen3-8B FP8 | 16.2 ms | 16.8 ms | 19.1 ms |
| Qwen3-8B AWQ 4-bit | **13.2 ms** | **13.8 ms** | **15.5 ms** |

**Does CLM's head survive a quantized backbone?** The authors' mid-trained head, unchanged, scored on our
rebuilt 12k-pair test slice (contexts cut to 512 tokens, so absolute numbers sit below the published 69%):

| Vectors from | Tricky | General |
|---|---|---|
| Qwen3-8B bf16 | 66.4% | 81.8% |
| Qwen3-8B FP8 | 65.6% | 81.7% |

FP8 costs 0.7 points tricky for a 22 to 24% faster backbone. AWQ accuracy: not yet measured.

**Heads and action cache**, the authors' mid-trained head, CPU (`quant_heads_mid.json`):

| Variant | Tricky | Head size | Cache for 10k actions |
|---|---|---|---|
| fp32 | 68.8% | 75.6 MB | 20.5 MB |
| bf16 | 68.9% | 37.8 MB | 10.2 MB |
| int8 weights, per channel | 68.8% | 19.0 MB | 20.5 MB |
| int8 dynamic (weights and activations) | **49.1%** | 18.9 MB | 20.5 MB |
| int8 action cache | 68.0% | 75.6 MB | 5.2 MB |
| 1-bit action cache | 58.2% | 75.6 MB | 0.6 MB |

Weight-only int8 and bf16 are free. Quantizing activations breaks the head (Qwen's hidden states have large
outlier dimensions), so keep activations in 16-bit.

## 3. Head layout: embedding dimensions and a shared space (ideas 2 and 4)

CPU, from scratch, matched budget: 12 published chunks (1.2M pairs), batch 1,024, 3,000 steps, seed 1234.
This is far from convergence (the authors' head reaches general loss 0.63, these sit near 1.4 to 1.7), so it
ranks layouts by how fast they learn, not by their final ceiling.

| Layout | Params | General | Tricky |
|---|---|---|---|
| Baseline: 2 separate heads, 1536 wide, 512 out | 18.9M | 56.5% | 34.2% |
| **Shared head** (one head for states and actions) | **9.4M** | **64.8%** | 35.8% |
| Shared trunk + 64-rank adapter per side | 9.8M | 64.2% | **36.1%** |
| Separate, 768 wide | 8.3M | 59.1% | 34.9% |
| Separate, 2 layers (no hidden block) | 14.2M | 57.3% | 32.8% |
| Separate, 128 out | 17.7M | 58.0% | 34.9% |
| Matryoshka 1024 out, read at 64 / 128 / 256 / 512 / 1024 | 20.5M | 58.7% at every size | 35.6% to 35.9% |

What this says, at this scale:
- **A shared embedding space learns fastest**, with half the parameters: +8 points general and +1.6 to
  1.9 tricky over the baseline at the same step count.
- **Output size barely matters**: a Matryoshka head is as good at 64 dims as at 1024. A 64-dim action
  cache is 8x smaller than CLM's 512-dim one.
- Depth matters more than width: dropping the hidden block hurts tricky; halving the width does not.

Open question: does the shared head's lead survive full training, or is it only faster early on? That is the
first thing to check at full scale (`modal_train.py --tower shared_adapter`, about $10 on the published vectors).

## 4. New backbones (idea 1)

Same head trained from scratch on each backbone's vectors of the same 96k-pair slice, 3,000 steps
(`bb_*`). The slice is tiny and both heads overfit it, so this is a screening signal, not a final score.

| Backbone | Hidden | Tricky | General |
|---|---|---|---|
| Qwen3-8B | 4096 | 34.2% | 48.3% |
| **Muse Glimmer 30B** | 6656 | **36.9%** | **53.3%** |
| Inkling 2 Mini (B200 only) | 2304 | 31.8% | 40.1% |

Inkling 2 Mini needs a Blackwell GPU in vLLM's nightly image and a loader patch: see `inkling_attempts.md`.
After the same post-training, Muse is best on ARC, CSQA and Q&A and Qwen on intent routing.

## 5. Post-training (idea 5)

Full table, T-Rex runs and costs: `results/innovations/slice_run_2026-11-22.md`.

Listwise post-training from the authors' mid-trained head: 400 steps, sibling options as hard negatives,
and 40% replay of published Q&A vectors **with** their hard negatives. Compared with the authors'
post-trained head, on Qwen3-8B vectors:

| Test | CLM post head | Ours (`post_choice_hn`) |
|---|---|---|
| Tricky Q&A | 66.1% | **68.7%** |
| General Q&A | **81.8%** | 80.2% |
| Tool choice (Glaive) | 56.0% | **97.5%** |
| Ticket routing (Banking77) | 22.1% | **83.4%** |
| Intent routing (CLINC) | 29.0% | **86.1%** |
| Label reading | 32.9% | **50.8%** |
| ARC / CommonsenseQA | 30.6% / 30.7% | **42.3% / 46.0%** |
| **T-Rex courses survived (shield on)** | **5/5** | 0/5 |

- Replay without hard negatives cost 5.3 points tricky; with them it gains 2.6. That matches the baseline's own lesson.
- The choice gains are in-distribution (test rows come from the training sources).
- **T-Rex is lost.** The authors' mid head also survives 0/5, so T-Rex skill comes only from their agent
  post-training. All these variants wash it out, including a gentle run starting from their post head.
  Section 6 fixes this.

## 6. Keeping T-Rex: the best head so far

Full table, every attempt and costs: `results/innovations/day2_2026-11-22.md`.

**`wise_0.15`** is a weight blend with no extra training: 85% the authors' post head and 15%
`post_choice_hn_frompost` (`python -m innovations.interpolate --alpha 0.15`). Compared with the authors' post head:

| Test | CLM post head | Ours (`wise_0.15`) |
|---|---|---|
| **T-Rex, shield on** (5 seeds: survived, deaths, score) | 5/5, 0, 697 | **5/5, 0, 697** |
| T-Rex, shield off (deaths, score) | **22, 221** | 45, 55 |
| Tricky Q&A | 66.1% | **66.6%** |
| General Q&A | 81.8% | 81.7% |
| Tool choice (Glaive) | 56.0% | **71.7%** |
| Ticket routing (Banking77) | 22.1% | **26.9%** |
| Intent routing (CLINC) | 29.0% | **37.4%** |
| Label reading | 32.9% | **35.3%** |
| ARC / CommonsenseQA | 30.6% / 30.7% | **33.0% / 33.6%** |
| DeepSWE (38 tasks, seed 1234) | 31/38 | 30/38 |

- It ties CLM on T-Rex with the shield on (the published setting) and wins every listwise test and tricky
  Q&A. General Q&A is 0.1 points lower. DeepSWE is one task lower, inside the authors' own 29 to 32 seed range.
- Not yet a clean win: with the shield off CLM still plays T-Rex far better, and the choice gains are in-distribution.
- What failed: blending in more of our head (0.25: 4/5, 0.5: 0/5); distillation to the post head (0/5);
  distillation with a strong L2 anchor (5/5 but no gains at all); our ADP agent-data mix, 118k steps from
  34 configs (2/5 with distillation, 0/5 without). The authors' T-Rex skill sits in a small, fragile
  weight change that only a light blend preserves.
- Next, all cheap: blends between 0.15 and 0.25, DeepSWE over 5 seeds, and a choice test from held-out sources.

Total Modal spend for both days: $63.15 (`modal billing summary`).
