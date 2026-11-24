# Results (22 and 23 November 2026)

Everything here was measured, except where it says estimate or guess. "Tricky" is the hard-negative Q&A test
(gold answer against its own up-to-10 Gemini wrong answers; CLM publishes 69.2% after mid-training and 68.5% for
the released head). "General" is in-batch top-1 retrieval over 1,024-pair blocks with same-document pairs masked.
Raw numbers: `results/innovations/`.

## 1. Checking the eval code against the authors' heads

Our `evals.py` on the published vectors (the 19,136 rows with hard negatives among the first 20,000 of
`shard_00/0028`, and 20,480 rows of `shard_05/0083`). A CPU rerun on 24 November gave the same numbers
and is saved in `results/innovations/eval_check/`:

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
| Qwen3-8B AWQ 4-bit | 61.9% | 79.7% |

FP8 is the practical choice: it costs 0.7 points tricky for 21 to 23% lower backbone latency. AWQ cuts latency
by 36 to 37% but costs 4.5 points tricky and 2.1 general (`auth_awq.json`). The two released heads were not
scored on quantized vectors.

**Heads and action cache**, the authors' mid-trained head, CPU (`quant_heads_mid.json`):

| Variant | Tricky | Head size | Cache for 10k actions |
|---|---|---|---|
| fp32 | 68.8% | 75.6 MB | 20.5 MB |
| bf16 | 68.9% | 37.8 MB | 10.2 MB |
| int8 weights, per channel | 68.8% | 19.0 MB | 20.5 MB |
| int8 dynamic (weights and activations) | **49.1%** | 18.9 MB | 20.5 MB |
| int8 action cache | 68.0% | 75.6 MB | 5.2 MB |
| 1-bit action cache | 58.2% | 75.6 MB | 0.6 MB |

Weight-only int8 and bf16 cost nothing on tricky Q&A, the one test run for this. Quantizing activations breaks
the head, probably because Qwen's hidden states have large outlier dimensions (not measured), so keep
activations in 16-bit.

## 3. Head layout: embedding dimensions and a shared space (ideas 2 and 4)

CPU, from scratch, matched budget: 12 published chunks (1.2M pairs), batch 1,024, 3,000 steps, one seed (1234).
This is far from convergence (the authors' head reaches general loss 0.63, `eval_check/`; these sit near 1.4
to 1.7), so it ranks layouts by how fast they learn, not by their final ceiling.

| Layout | Params | General | Tricky |
|---|---|---|---|
| Baseline: 2 separate heads, 1536 wide, 512 out | 18.9M | 56.5% | 34.2% |
| **Shared head** (one head for states and actions) | **9.4M** | **64.8%** | 35.8% |
| Shared trunk + 64-rank adapter per side | 9.8M | 64.2% | **36.1%** |
| Separate, 768 wide | 8.3M | 59.1% | 34.9% |
| Separate, 2 layers (no hidden block) | 14.2M | 57.3% | 32.8% |
| Separate, 128 out | 17.7M | 58.0% | 34.9% |
| Matryoshka 1024 out, read at 64 / 128 / 256 / 512 / 1024 | 20.5M | 58.7% to 58.8% | 35.6% to 35.9% |

What this says, at this scale:
- **A shared embedding space learns fastest**, with half the parameters: +8 points general and +1.6 to
  1.9 tricky over the baseline at the same step count.
- **Output size barely matters**: a Matryoshka head at 64 dims is within 0.3 points of its score at 1024.
  A 64-dim action cache is 8x smaller than CLM's 512-dim one.
- Depth matters more than width: dropping the hidden block hurts tricky; halving the width does not.

Open question: does the shared head's lead survive full training, or is it only faster early on? That is the
first thing to check at full scale (`modal_train.py --tower shared_adapter`; our guess is about $10 on the
published vectors, which is a guess and not a measurement).

## 4. New backbones (idea 1)

The same head design trained from scratch on each backbone's vectors of the same 96k-pair slice, 3,000 steps
(`bb_*`). The input width follows the backbone, so the heads have 18.9M, 26.8M and 13.4M parameters. The slice
is tiny and all three heads overfit it, so this is a screening signal, not a final score.

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

- Replay and hard negatives. A run that replayed our own slice vectors with no hard negatives (`post_choice`)
  scored 60.9% tricky. A run that replayed CLM's published vectors with their hard negatives (`post_choice_hn`)
  scored 68.7%. Both start from the authors' mid-trained head (66.4% on this test). The two runs differ in
  replay source as well as in negatives. The Vizuara baseline had already found that replay needs hard negatives.
- The Q&A test slice is not fully held out for the runs that replay published chunks: 424 of its 11,998 test
  pairs (53 of 1,500 documents, 3.5%) are in the replayed chunks `shard_00/0` and `shard_00/1`
  (`data_checks.json`).
- The choice gains are in-distribution (test rows come from the training sources). About 4 in 10 Glaive states
  carry earlier dialogue turns and an end-of-text marker (510 of 1,129 test rows, 23,618 of 58,871 training
  rows), because the builder keeps everything before the first function call.
- The label-reading set is 38,838 training rows and 1,162 test rows built from the multiple-choice questions:
  one random option carries a positive tag such as Correct or Safe, the others the opposite tag, and the state
  asks for the tagged one. It was meant to teach the label reading that T-Rex needs. The head improved on it
  (32.9% to 50.8%, chance 24.6%) and still lost T-Rex.
- **T-Rex is lost.** The authors' mid head also survives 0/5, so T-Rex skill comes only from their agent
  post-training. All these variants wash it out, including a gentle run starting from their post head.
  Section 6 gets survival back with the shield on, and not the skill with the shield off.

## 6. Keeping T-Rex: the launch head

Full tables, every attempt and costs: `results/innovations/day2_2026-11-22.md` and `day3_2026-11-23.md`.

**`wise_0.15`** (released as S1-Tweaks v0.1) is a weight blend with no extra training: 85% the authors' post head
and 15% `post_choice_hn_frompost` (`python -m innovations.interpolate --alpha 0.15`). Compared with the authors' post head:

| Test | CLM post head | Ours (`wise_0.15`) |
|---|---|---|
| **T-Rex, shield on**, second run, seeds 0 to 14 (survived, deaths, mean best score) | 15/15, 0, 697.0 | 14/15, 1, 685.8 |
| T-Rex, shield on, first run, seeds 0 to 4 | 5/5, 0, 697 | 5/5, 0, 697 |
| T-Rex planner agreement, shield on, seeds 0 to 14 | **64.8%** | 54.3% |
| T-Rex shield interventions, shield on, seeds 0 to 14 | 15,668 in 50,696 decisions | 21,022 in 51,524 decisions |
| T-Rex, shield off, 5 seeds (survived, deaths, score) | **0/5, 22, 221** | 0/5, 45, 55 |
| Tricky Q&A | 66.1% | **66.6%** |
| General Q&A | 81.8% | 81.7% |
| Tool choice (Glaive) | 56.0% | **71.7%** |
| Ticket routing (Banking77) | 22.1% | **26.9%** |
| Intent routing (CLINC) | 29.0% | **37.4%** |
| Label reading | 32.9% | **35.3%** |
| ARC / CommonsenseQA | 30.6% / 30.7% | **33.0% / 33.6%** |
| DeepSWE after CLM's own fine-tune, 38 tasks, mean of 5 seeds | 31.0 | 30.8 |

- **What T-Rex measures.** The model never sees the game. A physics planner labels each of three actions Safe
  or Unsafe, and one of them Best, and the model picks from that text in real time. With the shield on, an
  unsafe answer is replaced, so survival measures the model and the shield together. Planner agreement and
  the intervention counts measure the model.
- **T-Rex is a near-tie, not a match.** Over the two runs `wise_0.15` survives 19 of 20 games against 20 of 20 for
  the authors' head. The 20 games cover 15 distinct courses. Seeds 0 to 4 were played twice, and the first run is
  the one the 0.15 blend was picked on. Its one death is seed 0 of the second run, a seed it survived in the first
  run, so real-time play is not fully deterministic. On the ten new seeds both heads survive 10 of 10.
- **It leans on the shield more.** Over seeds 0 to 14 the harness counted 21,022 shield interventions against
  15,668 (`trex_15seeds/`). That count is the sum of three counters: answers that the shield's check changed when
  they arrived (19,014 against 14,710), top picks that the planner had labelled unsafe and were replaced (1,995
  against 941), and emergency moves (13 against 17).
- With the shield off the authors' head is far better, and no head survives a course: the authors' head survives
  0 of 5 with 22 deaths. Every blend dies 45 times in 5 seeds. That is 9 deaths in every course, and it is what
  every failing head gets, including the authors' mid-trained head, which has no agent training. A round-trip
  copy of the authors' head died 23 times, so the 22 is not exact either.
- DeepSWE is a tie within seed noise: per seed 31, 32, 30, 31, 31 for the authors' head and 30, 32, 30, 31, 31 for ours
  (seeds 1234, 1, 2, 3, 4). Each head is first fine-tuned on DeepSWE steps with CLM's own `finetune.py`, then
  scored, so the row tests the head as a starting point for that fine-tune and not a skill of the shipped head.
  The authors' head's five values equal the baseline's values of 6 November in `results/results.json`.
  <!-- PENDING: whether the authors' head was rerun on DeepSWE on 22 and 23 November for all five seeds, or its values were taken from results/results.json -->
- The choice tests in this table share sources with our training data. Section 7 is the held-out check.
- Larger blends: 0.175 survives 3/5, 0.2 4/5, 0.25 4/5, 0.5 0/5 (5 seeds each). 0.15 is the largest blend that
  survived all five courses in its 5-seed run. 0.175, 0.2 and 0.25 lost two, one and one of five, which 5 seeds
  cannot tell apart from 0.15 (it lost one of 20). The authors' own head lost one of five in the baseline's run of
  6 November. 0.5 clearly fails: 0 of 5 and 41 deaths. Planner agreement drops at the first step (65.4%, 55.1%,
  49.3%, 50.8%, 42.9% at 0, 0.15, 0.175, 0.2, 0.25) and is no clean gauge either: the 0.5 blend agrees 62.6% of
  the time and still dies 41 times.
- What else failed: distillation to the post head (0/5); distillation with a strong L2 anchor (5/5 but no gain
  above 0.4 points on any test); our ADP agent-data mix, 118k steps from 34 configs (2/5 with distillation, 0/5
  without). The authors' T-Rex skill sits in a small, fragile weight change. A light blend keeps survival with
  the shield on and loses the skill with the shield off.
- No matched control: we did not run a training recipe as light as the 0.15 blend. At similar in-distribution
  gains, the gentle training run kept more of T-Rex than the 0.5 blend (7 deaths and score 398 against 41 deaths
  and score 67, both 0 of 5). So this shows that a blend worked, not that blending beats light training.

## 7. Held-out choice test, built by us from public sets

13,194 multiple-choice rows built by `innovations/build_heldout_choice.py` from eight public sets: seven sources
that none of our training data touches, plus the MMLU test split. We trained on MMLU's `auxiliary_train` set,
which is built from other datasets, so that row is a held-out split and not a new source. The rows use the same
option templates as training. These are our builds and not the benchmarks' own metrics: the BFCL row is a
10-option tool choice made from BFCL v3 files, not the BFCL score that CLM reports. Rows whose question text
also appears in our training file were dropped (34 in total). OpenBookQA is excluded because MMLU's
auxiliary_train, which we trained on, contains it. Top-1 accuracy, `heldout.json`. Chance is a random pick among
the options (`heldout_paired.json`, `data_checks.json`):

| Source (rows) | Kind | Options, chance | CLM post head | `wise_0.15` | Full post-train (`post_choice_hn_frompost`) |
|---|---|---|---|---|---|
| BFCL v3, our build (1,252) | tool choice | 10, 10.0% | 65.4% | 68.7% | 76.8% |
| MASSIVE (2,951) | intent routing | 10, 10.0% | 26.3% | 31.7% | 60.7% |
| HWU64 (1,072) | intent routing | 10, 10.0% | 29.5% | 34.4% | 57.3% |
| Bitext (2,000) | support routing | 10, 10.0% | 16.1% | 17.9% | 39.8% |
| SciQ (1,000) | science questions | 4, 25.0% | 56.5% | 60.3% | 69.8% |
| QASC (923) | science questions | 8, 12.5% | 36.4% | 39.0% | 48.6% |
| MMLU test (1,997) | mixed knowledge | 4, 25.0% | 28.9% | 29.2% | 31.6% |
| MedMCQA (1,999) | medical questions | 4, 25.0% | 29.0% | 29.1% | 30.0% |

Paired test on the same rows (`modal run innovations/heldout_paired.py`, `heldout_paired.json`): change against the
authors' head in points, 95% bootstrap interval, and exact McNemar p-value.

| Source | `wise_0.15` | Net rows gained | `post_choice_hn_frompost` |
|---|---|---|---|
| BFCL v3 | +3.27 (2.32 to 4.31), p 1.0e-11 | +41 of 1,252 | +11.42 (9.11 to 13.74) |
| MASSIVE | +5.42 (4.54 to 6.34), p 2.8e-34 | +160 of 2,951 | +34.40 (32.43 to 36.36) |
| HWU64 | +4.94 (3.54 to 6.44), p 1.7e-12 | +53 of 1,072 | +27.80 (24.44 to 31.25) |
| Bitext | +1.85 (1.20 to 2.50), p 2.5e-08 | +37 of 2,000 | +23.70 (21.20 to 26.25) |
| SciQ | +3.80 (2.40 to 5.30), p 1.4e-07 | +38 of 1,000 | +13.30 (10.30 to 16.40) |
| QASC | +2.60 (1.08 to 4.23), p 0.0018 | +24 of 923 | +12.24 (8.78 to 15.71) |
| MMLU test | +0.25 (-0.75 to 1.30), p 0.71 | +5 of 1,997 | +2.60 (0.30 to 4.81), p 0.025 |
| MedMCQA | +0.05 (-0.85 to 0.90), p 1.0 | +1 of 1,999 | +1.00 (-1.05 to 3.15), p 0.37 |
| The six sources that moved, pooled (9,198) | +3.84 (3.39 to 4.29) | +353 | +23.66 (22.53 to 24.79) |
| All eight, pooled (13,194) | +2.72 (2.35 to 3.11) | +359 | +17.04 (16.13 to 17.94) |

- **The gains carry over on six of eight sources.** `wise_0.15` is ahead of the authors' head on tools (+3.3),
  routing (+1.8 to +5.4) and science questions (+2.6, +3.8), and each of these six intervals is above zero. All
  six also rise at every blend weight we scored. MMLU and MedMCQA are flat, and every head is within 7 points of
  chance on them.
- The gains are smaller than on the in-distribution tests of section 6 (tool choice +15.7 there, +3.3 here).
- The option wording is ours (`name: description`, and for routing `name: Route the ticket to name.`). The
  authors' head never saw it and we did not test other wordings, so part of the gain may be the heads learning
  our template. The intervals cover the rows of this one build, not a second build or a second training run.
- The fully post-trained head carries over much more (MASSIVE 26.3% to 60.7%, Bitext 16.1% to 39.8%, BFCL 65.4%
  to 76.8%), but it survives 0/5 T-Rex courses. On MMLU and MedMCQA its lead is small. It is the better pick when
  the job is routing or tool choice and nothing like T-Rex is needed.

Open questions, all cheap to test: a blend that keeps T-Rex with the shield off, agent data that restores the
authors' T-Rex skill, and the shared head (section 3) at full scale.

Total Modal spend for all three rounds: about $69 (`modal billing summary` read $68.20 before the last runs were metered).
<!-- PENDING: the final Modal bill, read after the last runs were metered -->
