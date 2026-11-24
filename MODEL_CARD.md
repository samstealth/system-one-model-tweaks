---
license: apache-2.0
base_model:
  - Contrastive-LM/CLM-v0.1-8B
  - Qwen/Qwen3-8B
library_name: pytorch
language:
  - en
tags:
  - system-one
  - decision-model
  - contrastive
  - clm
  - ranking
  - routing
  - tool-selection
  - weight-interpolation
  - wise-ft
---

# S1-Tweaks v0.1 and S1-Router v0.1

Two small decision heads for the frozen Qwen3-8B backbone. They do not write text. You give one a
situation (the state) and a short list of options. It returns a probability for each option in one pass.

Both are derived from the official
[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) head from Stanford and NVIDIA Research,
and both use CLM's own checkpoint layout, so the official CLM engine serves them with `--ckpt`.

| | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|
| What it is | A weight blend: 85% CLM's head + 15% S1-Router. No training after the blend. | CLM's head, post-trained for 400 steps on tool choice and routing data. |
| Pick it when | You want to stay close to CLM (85% of its weights) and gain a few points on tool choice and routing. It survives 19 of 20 T-Rex games with the safety shield on and loses that skill with the shield off, so it is not for T-Rex-like use without a safety check. | The job is routing or tool choice and nothing like the T-Rex game matters. |
| File | `s1_tweaks_v0.1.pt` (run `wise_0.15`) | `s1_router_v0.1.pt` (run `post_choice_hn_frompost`) |
| Size | 75,557,108 bytes | 75,557,108 bytes |

```text
sha256  c0a32217d1288d8452046bda9bcb045b53b052b943ef7de42652691e057fd9d9  s1_tweaks_v0.1.pt
sha256  d9f22c9fe0a52d4bd32fe1a6c965b3900fc854c44f1c6267f23bf6e897911b18  s1_router_v0.1.pt
```

Download: https://github.com/samstealth/system-one-model-tweaks/releases/tag/v0.1 .
Code and every measurement: https://github.com/samstealth/system-one-model-tweaks .
Backbone (not included): [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), frozen. Licence: Apache 2.0.

## Results against CLM-v0.1-8B

All heads were scored with the same code on the same Qwen3-8B vectors, on 22 and 23 November 2026.
Bold marks the best head in a row where the lead is clear.

### Held-out tests, built by us from public sets

13,194 multiple-choice rows that we built from eight public sets: seven sources that none of our
training data touches, plus the MMLU test split (we trained on MMLU's `auxiliary_train` set, which is
built from other datasets, so that row is a held-out split and not a new source). Top-1 accuracy.
Chance is a random pick among the options: 10 options for the first four rows, 8 for QASC and 4 for
SciQ, MMLU and MedMCQA. These are our builds, not the benchmarks' own metrics.

| Source (rows) | Kind | Chance | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|---|---|
| BFCL v3, our 10-option build (1,252) | tool choice | 10.0% | 65.4% | 68.7% | **76.8%** |
| MASSIVE (2,951) | intent routing | 10.0% | 26.3% | 31.7% | **60.7%** |
| HWU64 (1,072) | intent routing | 10.0% | 29.5% | 34.4% | **57.3%** |
| Bitext (2,000) | support routing | 10.0% | 16.1% | 17.9% | **39.8%** |
| SciQ (1,000) | science questions | 25.0% | 56.5% | 60.3% | **69.8%** |
| QASC (923) | science questions | 12.5% | 36.4% | 39.0% | **48.6%** |
| MMLU test (1,997) | mixed knowledge | 25.0% | 28.9% | 29.2% | 31.6% |
| MedMCQA (1,999) | medical questions | 25.0% | 29.0% | 29.1% | 30.0% |

Paired bootstrap over the same rows, 95% interval of the change against CLM in points
([`heldout_paired.json`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/heldout_paired.json)):

| Source | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|
| BFCL v3 | +3.27 (2.32 to 4.31) | +11.42 (9.11 to 13.74) |
| MASSIVE | +5.42 (4.54 to 6.34) | +34.40 (32.43 to 36.36) |
| HWU64 | +4.94 (3.54 to 6.44) | +27.80 (24.44 to 31.25) |
| Bitext | +1.85 (1.20 to 2.50) | +23.70 (21.20 to 26.25) |
| SciQ | +3.80 (2.40 to 5.30) | +13.30 (10.30 to 16.40) |
| QASC | +2.60 (1.08 to 4.23) | +12.24 (8.78 to 15.71) |
| MMLU test | +0.25 (-0.75 to 1.30) | +2.60 (0.30 to 4.81) |
| MedMCQA | +0.05 (-0.85 to 0.90) | +1.00 (-1.05 to 3.15) |

### Agent benchmarks

| Test | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|
| T-Rex game, safety shield on, second run, seeds 0 to 14: survived, deaths, mean best score | **15/15, 0, 697.0** | 14/15, 1, 685.8 | not run |
| T-Rex game, safety shield on, first run, seeds 0 to 4: survived, deaths, score | **5/5, 0, 697** | **5/5, 0, 697** | 0/5, 17, 198 |
| T-Rex agreement with the reference planner, shield on, seeds 0 to 14 | **64.8%** | 54.3% | not run |
| T-Rex shield interventions, shield on, seeds 0 to 14 | 15,668 | 21,022 | not run |
| T-Rex game, safety shield off, 5 seeds: survived, deaths, score | **0/5, 22, 221** | 0/5, 45, 55 | 0/5, 45, 46 |
| DeepSWE verifier after CLM's own fine-tune, 38 held-out tasks, mean of 5 seeds | 31.0 | 30.8 | not run |

### In-distribution tests

These test rows come from the same sources as the post-training data (held-out rows, not held-out
sources), so read them as an upper bound.

| Test | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|
| Tricky Q&A (gold answer against up to 10 hard negatives, 7,848 rows) | 66.1% | 66.6% | **68.2%** |
| General Q&A (in-batch retrieval) | **81.8%** | 81.7% | 79.9% |
| Tool choice (Glaive function calling), chance 10.0% | 56.0% | 71.7% | **97.5%** |
| Ticket routing (Banking77), chance 10.0% | 22.1% | 26.9% | **82.8%** |
| Intent routing (CLINC), chance 10.0% | 29.0% | 37.4% | **85.7%** |
| Label reading (pick the option that carries a tag such as Correct or Safe), chance 24.6% | 32.9% | 35.3% | **51.9%** |
| ARC, chance 25.0% | 30.6% | 33.0% | **42.1%** |
| CommonsenseQA, chance 20.0% | 30.7% | 33.6% | **46.3%** |

About 4 in 10 Glaive states carry earlier turns of the dialogue and an end-of-text marker (510 of the
1,129 test rows), because the builder keeps everything before the first function call. The Q&A test
slice shares 424 of its 11,998 pairs (3.5%) with the two published chunks that our post-training
replayed ([`data_checks.json`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/data_checks.json)).

**Limits of these numbers. Please read them before quoting the tables.**

1. **T-Rex is a near-tie for S1-Tweaks, not a match.** Over the two shield-on runs it survives 19 of
   20 games against 20 of 20 for CLM. The 20 games cover 15 distinct courses. Seeds 0 to 4 were
   played twice, and the first 5 games are the ones the blend was picked on. Its one death is on
   seed 0, which it survived in the first run, so real-time play is not fully deterministic. On the
   ten new seeds both heads survive 10 of 10.
2. **T-Rex survival measures the model and the shield together.** The model never sees the game. A
   physics planner labels each of three actions Safe or Unsafe, and one of them Best, and the model
   picks from that text. With the shield on, an unsafe answer is replaced. S1-Tweaks agrees with the
   planner less often (54.3% against 64.8%) and leans on the shield more: over seeds 0 to 14 the
   harness counted 21,022 shield interventions in 51,524 decisions against 15,668 in 50,696 for CLM.
   Top picks that the planner had labelled unsafe were replaced 1,995 times against 941.
3. **Without the T-Rex safety shield, CLM is much better** than either head (score 221 against 55 and
   46), and no head survives a course: CLM survives 0 of 5 with 22 deaths. 45 deaths is 9 in every
   course. It is what every failing head gets, including CLM's mid-trained head from before its
   agent training. So S1-Tweaks keeps T-Rex survival with the shield on and loses the skill with it off.
4. **S1-Router gives up T-Rex completely** (0/5 courses with the shield on). Its 15-seed T-Rex run
   and DeepSWE were not run.
5. **Held-out gains are real but smaller than in-distribution ones, and the tests are our builds.**
   For S1-Tweaks, tool choice is +15.7 points on Glaive (a training source) and +3.3 on BFCL (never
   seen). The option wording is ours (`name: description`, and for routing
   `name: Route the ticket to name.`). CLM never saw it and no other wording was tested, so part of
   the gain may be the heads learning our template. MMLU and MedMCQA are flat and near chance for
   every head. On those two, S1-Router's lead is small, and on MedMCQA its interval includes zero.
6. **This is slice-scale work, not a full retrain.** The post-training saw 233k choice rows for 400
   steps. The Q&A tests use a rebuilt 12k-pair slice with contexts cut to 512 tokens, so they sit
   below CLM's published 68.5% for this head.
7. **DeepSWE is a tie after fine-tuning, within seed noise.** Each head is first fine-tuned on
   DeepSWE steps with CLM's own script, then scored, so the row tests the head as a starting point
   for that fine-tune. Per seed: 31, 32, 30, 31, 31 for CLM and 30, 32, 30, 31, 31 for S1-Tweaks.

Raw numbers: [`results/innovations/`](https://github.com/samstealth/system-one-model-tweaks/tree/HEAD/results/innovations)
(`heldout.json`, `heldout_paired.json`, `wise_0.15.json`, `post_choice_hn_frompost.json`, `auth_post.json`, `trex.json`,
`trex_15seeds/`, `deepswe.json`, `data_checks.json`).
Full write-up: [`innovations/RESULTS.md`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/innovations/RESULTS.md).

## What it is

| Part | Detail |
|---|---|
| Backbone | Qwen/Qwen3-8B, frozen, not part of this file |
| Input to the head | hidden state of the last token of the final layer, L2 normalized, 4096 numbers |
| State head and action head | each 4096 -> 1536 -> 1536 (LayerNorm) -> 512, GELU |
| Parameters | 18,887,680 in the two heads (9.44M each) |
| Score | cosine of the two 512-number vectors times a learned scale (capped at 100), then a softmax over the options |
| Text layout | state = state, a blank line, then the instructions; option = `name: description` (how our choice data writes options) |

The architecture and the checkpoint layout are CLM's. Only the weight values differ. Speed is the
same as CLM-8B: the T-Rex runs measured a median of 16 to 17 ms per decision for every head on one
L40S, with 6 requests in flight.

## How they were made

1. **Start from CLM's released head.** `CLM_v0.1-8B.pt` is the authors' post-trained head.
2. **Post-train a copy on choice data: this is S1-Router v0.1.** Run `post_choice_hn_frompost`: 400
   steps, batch 4,096, learning rate 0.0003, starting from CLM's head. Each training row is one state
   with the sibling options of the same question as hard negatives. 40% of every batch is replay of
   CLM's published Q&A vectors with their hard negatives. The choice data is 233k rows built by
   `innovations/build_choice_data.py` from Banking77, CLINC, Glaive function calling, ARC,
   CommonsenseQA, the MMLU auxiliary set and a label-reading set derived from them (about 39k rows
   where one random option carries a tag such as Correct or Safe and the state asks for it).
3. **Blend: this is S1-Tweaks v0.1.** `python -m innovations.interpolate --alpha 0.15` replaces every
   weight tensor of both heads, and the log scale, by `0.85 * CLM + 0.15 * S1-Router`. This is plain
   weight interpolation in the style of WiSE-FT (Wortsman et al., 2022). No gradient step follows it.

Why blend at all: S1-Router is far better at tool choice and routing, but it loses CLM's T-Rex skill
completely. The blend weight trades one for the other:

| Share of S1-Router in the blend | T-Rex courses survived, shield on, 5 seeds | BFCL (held-out) | MASSIVE (held-out) | Tool choice (Glaive) | Tricky Q&A |
|---|---|---|---|---|---|
| 0 (CLM-v0.1-8B) | 5/5 (15/15 in the second run) | 65.4% | 26.3% | 56.0% | 66.1% |
| **0.15 (S1-Tweaks v0.1)** | 5/5 (14/15 in a second run over seeds 0 to 14) | 68.7% | 31.7% | 71.7% | 66.6% |
| 0.175 | 3/5 | 69.5% | 32.2% | 73.2% | 66.8% |
| 0.2 | 4/5 | 69.7% | 33.3% | 75.0% | 67.0% |
| 0.25 | 4/5 | 70.5% | 35.5% | 78.0% | 66.9% |
| 0.5 | 0/5 | 74.8% | 45.8% | 90.3% | 67.5% |
| 1 (S1-Router v0.1) | 0/5 | 76.8% | 60.7% | 97.5% | 68.2% |

0.15 is the largest blend that survived all five courses in its 5-seed run. 0.175, 0.2 and 0.25
lost two, one and one of five, which 5 seeds cannot tell apart from 0.15 (it lost one of 20). 0.5
clearly fails: 0 of 5 and 41 deaths. Agreement with the planner drops at the first step (65.4%,
55.1%, 49.3%, 50.8% and 42.9% at 0, 0.15, 0.175, 0.2 and 0.25) and is no clean gauge either: the
0.5 blend agrees 62.6% of the time and still dies 41 times.

Other ways to keep T-Rex were tried and failed: distillation toward CLM's head (0/5), distillation
with a strong L2 anchor (5/5 but no gain above 0.4 points on any test), and mixing in 118k agent
steps (2/5 with distillation, 0/5 without). We did not run a training recipe as light as the 15%
blend, so none of this shows that blending beats light training. Details are in
[`results/innovations/day2_2026-11-22.md`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/day2_2026-11-22.md)
and [`day3_2026-11-23.md`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/day3_2026-11-23.md).
All the experiments in the repo together cost about $69 on Modal.
<!-- PENDING: the final Modal bill, read after the last runs were metered -->

## Intended use

- Ranking a short list of options against a situation: which tool an agent should call, which team
  or intent a message belongs to, which candidate answer fits.
- A fast first decision before, or in place of, a call to a generative model.
- Research on System One models, weight blending and forgetting.

Not intended for:

- Writing text. The model only scores options you supply.
- Decisions with real consequences without a check around it. In the T-Rex test the safety shield
  matters a lot, for these heads more than for CLM.
- Use with any backbone other than Qwen3-8B with last-token pooling. The head reads that model's vectors only.

## How to use it

### 1. Score vectors on a CPU

```bash
git clone https://github.com/samstealth/system-one-model-tweaks
cd system-one-model-tweaks
python -m venv .venv && source .venv/bin/activate
pip install -e .
# scores 60 real decisions with CLM's head and both of these
python examples/demo_cpu.py
```

```python
import numpy as np
from innovations.scorer import load_head

z = np.load("examples/data/sample_decisions.npz")   # the 60 shipped decisions
state_vec, option_vecs = z["q"][0], z["opts"][0][: z["n_opts"][0]]

head = load_head("s1-tweaks")                # or "s1-router", or a local path to a .pt file
probs = head.probs(state_vec, option_vecs)   # Qwen3-8B vectors: (4096,), (K, 4096) -> (K,) probabilities

clm = load_head("clm")                       # CLM's official head from Hugging Face, for comparison
```

The loader reads the file with `torch.load(weights_only=True)`, which limits loading to tensors and
plain Python types, and it checks each download against the sha256 above. Use torch 2.6 or newer:
older versions had a flaw in that mode. On the demo's 60 decisions CLM's head gets 29 right,
S1-Tweaks 34 and S1-Router 47
([`demo_sample.json`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/demo_sample.json)).
A sample that small is for trying things, not for quoting.

### 2. Text in, with a GPU

```bash
pip install "transformers>=4.51" accelerate
python examples/route_text.py --head s1-tweaks \
    --state "Customer: my invoice was charged twice and nobody answers the phone!" \
    --instructions "Which team should handle this?" \
    --option "billing: Charges, invoices, refunds" --option "technical: Bugs and outages"
```

This embeds the texts with Qwen/Qwen3-8B through transformers and needs a GPU with 24 GB or more.

Checked on one L40S with the pinned Qwen3-8B snapshot: the vectors this script computes match the
vLLM vectors behind our results (cosine above 0.9997 on all 60 states and 600 options), and 59 of
60 picks of CLM's head are the same
([`route_text_check.json`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/route_text_check.json)).

### 3. Serve a head with the official CLM engine

The file has CLM's layout (`state_head`, `action_head`, `logit_scale`, `cfg`), so
[the CLM engine](https://github.com/Contrastive-LM/CLM) serves it unchanged:

```bash
pip install contrastive-lm

# 1. encoder (Qwen3-8B embeddings)
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &

# 2. CLM API on :8700, with this head in place of the reference head
clm-serve --ckpt s1_tweaks_v0.1.pt        # or s1_router_v0.1.pt
```

```python
from clm import CLMClient, Choice

client = CLMClient()                          # http://127.0.0.1:8700 by default
r = client.system_one(
    state="User request: The checkout page has been down for 20 minutes and I cannot fix it myself. Get the on-call engineer to look at it.",
    questions={"tool": Choice(
        instructions="Which tool should the agent call next?",
        criteria={"create_ticket": "create_ticket: Open a ticket for the on-call engineer with a title and a description.",
                  "restart_service": "restart_service: Restart a named service on the staging cluster.",
                  "search_logs": "search_logs: Search application and job logs by time range and keyword.",
                  "no_tool": "no_tool: Answer directly without calling a tool."})},
)
print(r.answers["tool"].choice, r.answers["tool"].probabilities)
```

Two notes on serving:

- The engine sends each option's description to the encoder as written (`src/clm/schema.py`). The
  choice data behind these heads writes options as `name: description`, so repeat the name inside the
  description, as above.
- Our T-Rex runs served these heads through `clm-serve --ckpt`, with `vllm==0.11.0`, `transformers<5`
  and CLM at commit `d5f9ef0`. An unpinned install failed in the baseline's tests.

## Limitations

- **T-Rex.** S1-Tweaks is a near-tie with CLM with the safety shield on (19 of 20 games against 20
  of 20) and much worse with it off. S1-Router cannot play it at all. CLM's T-Rex skill sits in a
  small, fragile weight change. A light blend keeps survival with the shield on. With the shield
  off it dies as often as a head with no agent training. Nothing we tried keeps all of the skill.
- **S1-Tweaks' gains on unseen sources are modest**: +1.8 to +5.4 points on tools, routing and
  science questions, and nothing on MMLU or MedMCQA. In absolute terms it is still a weak router
  (31.7% on MASSIVE with 10 options, where chance is 10%). The held-out tests are our own
  multiple-choice builds with our option wording. If you need routing and do not need CLM's agent
  skill, use S1-Router.
- **S1-Router was tested less.** No DeepSWE run and no 15-seed T-Rex run. General Q&A retrieval
  drops 1.9 points against CLM (79.9% against 81.8%).
- **Slice-scale.** No full-scale retrain was run. Results may change with more data and more steps.
- **Quantized backbones were measured with CLM's mid-trained head only**: FP8 vectors cost 0.7 points
  on tricky Q&A, AWQ 4-bit vectors cost 4.5. Both released heads were only run on bf16 vectors.
- **English only.** All post-training and test data is English. Other languages were not tested.
- **Probabilities are not calibrated.** We measured top-1 accuracy, not calibration.

## Credits, licence and changes

- **Derived from CLM-v0.1-8B.** The weights are a modification of
  [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (`CLM_v0.1-8B.pt`),
  released under Apache 2.0 by Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré and Mirhoseini. The
  architecture, the checkpoint layout and the serving engine are theirs.
- **Changes made.** S1-Router v0.1: CLM's head after 400 further training steps on choice data, as
  described above. S1-Tweaks v0.1: every weight tensor of the state head and the action head, and the
  log scale, replaced by `0.85 * CLM + 0.15 * S1-Router`. Nothing else changed.
- **Baseline training code.** The repo builds on
  [VizuaraAI/system-one-from-scratch](https://github.com/VizuaraAI/system-one-from-scratch) (Apache 2.0),
  a rebuild of CLM-8B's training made for a live workshop.
- **Backbone.** Qwen/Qwen3-8B is not included in this file. See its model card for its licence.
- **Post-training data.** Banking77, CLINC, Glaive function calling v2, ARC, CommonsenseQA, the MMLU
  auxiliary set, and CLM's published Q&A vectors with hard negatives. The table below lists the
  licence tag on each dataset card.
- **These files** are released under Apache 2.0 by the GitHub user `samstealth`. They are not
  official CLM releases and the CLM authors have not reviewed them. The project is not affiliated
  with Vizuara AI Labs.

Licences of the post-training data, as tagged on each dataset card on Hugging Face (read on 24
November 2026):

| Data | Dataset | Licence tag on the card |
|---|---|---|
| Ticket routing | `PolyAI/banking77` | CC BY 4.0 |
| Intent routing | `clinc/clinc_oos` | CC BY 3.0 |
| Tool choice | `glaiveai/glaive-function-calling-v2` | Apache 2.0 |
| Science questions | `allenai/ai2_arc` | CC BY-SA 4.0 |
| Commonsense questions | `tau/commonsense_qa` | MIT |
| Mixed questions | `cais/mmlu`, `auxiliary_train` | MIT. That set is built from ARC, MC_TEST, OBQA and RACE, and we did not check the terms of each. |
| Q&A replay | `Contrastive-LM/CLM-v0.1-Pretrain-Nemotron` | no licence tag |
| Hard negatives for the replay | `jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings` | other: NVIDIA Open Data License |

Apache 2.0 covers our changes to CLM's Apache 2.0 weights. We make no statement on whether a
training set's licence reaches the weights. Check the list for your use.

## Citation

If you use these heads, please cite CLM, which they are built on:

```bibtex
@misc{kwok2026contrastivelanguagemodels,
  title={Contrastive Language Models},
  author={Jacky Kwok and Hangoo Kang and Tarun Suresh and
          Jon Saad-Falcon and Marco Pavone and
          Christopher Ré and Azalia Mirhoseini},
  year={2026},
  note={Notion Blog},
  url={https://contrastive-lm.notion.site}
}
```

To point at these exact heads, link the repository and the release:
https://github.com/samstealth/system-one-model-tweaks (release `v0.1`, files `s1_tweaks_v0.1.pt` and `s1_router_v0.1.pt`).
