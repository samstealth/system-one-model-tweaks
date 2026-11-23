---
license: apache-2.0
base_model: Qwen/Qwen3-8B
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
| Pick it when | You want CLM's behaviour, including most of its agent-game skill, with better tool choice and routing. | The job is routing or tool choice and nothing like the T-Rex game matters. |
| File | `s1_tweaks_v0.1.pt` (run `wise_0.15`) | `s1_router_v0.1.pt` (run `post_choice_hn_frompost`) |
| Size, sha256 | 75,557,108 bytes, `c0a32217d1288d8452046bda9bcb045b53b052b943ef7de42652691e057fd9d9` | 75,557,108 bytes, `d9f22c9fe0a52d4bd32fe1a6c965b3900fc854c44f1c6267f23bf6e897911b18` |

Download: https://github.com/samstealth/system-one-model-tweaks/releases/tag/v0.1 .
Code and every measurement: https://github.com/samstealth/system-one-model-tweaks .
Backbone (not included): [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), frozen. Licence: Apache 2.0.

## Results against CLM-v0.1-8B

All heads were scored with the same code on the same Qwen3-8B vectors, on 22 and 23 November 2026.
Bold marks the best head in a row.

### Held-out sources: nothing here was trained on

13,194 rows from eight public sources that none of our training data touches. Top-1 accuracy with up
to 10 options per question.

| Source (rows) | Kind | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|---|
| BFCL v3 (1,252) | tool choice | 65.4% | 68.7% | **76.8%** |
| MASSIVE (2,951) | intent routing | 26.3% | 31.7% | **60.7%** |
| HWU64 (1,072) | intent routing | 29.5% | 34.4% | **57.3%** |
| Bitext (2,000) | support routing | 16.1% | 17.9% | **39.8%** |
| SciQ (1,000) | science questions | 56.5% | 60.3% | **69.8%** |
| QASC (923) | science questions | 36.4% | 39.0% | **48.6%** |
| MMLU test (1,997) | mixed knowledge | 28.9% | 29.2% | **31.6%** |
| MedMCQA (1,999) | medical questions | 29.0% | 29.1% | **30.0%** |

### Agent benchmarks

| Test | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|
| T-Rex game, safety shield on, 15 fresh seeds: survived, deaths, mean best score | **15/15, 0, 697.0** | 14/15, 1, 685.8 | not run |
| T-Rex game, safety shield on, first 5 seeds: survived, deaths, score | **5/5, 0, 697** | **5/5, 0, 697** | 0/5, 17, 198 |
| T-Rex agreement with the reference planner, shield on, 15 seeds | **64.8%** | 54.3% | not run |
| T-Rex game, safety shield off, 5 seeds: deaths, score | **22, 221** | 45, 55 | 45, 46 |
| DeepSWE verifier, 38 held-out tasks, mean of 5 seeds | 31.0 | 30.8 | not run |

### In-distribution tests

These test rows come from the same sources as the post-training data (held-out rows, not held-out
sources), so read them as an upper bound.

| Test | CLM-v0.1-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|
| Tricky Q&A (gold answer against up to 10 hard negatives, 7,848 rows) | 66.1% | 66.6% | **68.2%** |
| General Q&A (in-batch retrieval) | **81.8%** | 81.7% | 79.9% |
| Tool choice (Glaive function calling) | 56.0% | 71.7% | **97.5%** |
| Ticket routing (Banking77) | 22.1% | 26.9% | **82.8%** |
| Intent routing (CLINC) | 29.0% | 37.4% | **85.7%** |
| Label reading | 32.9% | 35.3% | **51.9%** |
| ARC | 30.6% | 33.0% | **42.1%** |
| CommonsenseQA | 30.7% | 33.6% | **46.3%** |

**Limits of these numbers. Please read them before quoting the tables.**

1. **T-Rex is a near-tie for S1-Tweaks, not a match.** Over the two shield-on runs it survives 19 of
   20 games against 20 of 20 for CLM. Its one death is a seed it survived in the earlier run, so
   real-time play is not fully deterministic. It agrees with the reference planner less often
   (54.3% against 64.8%) and leans on the shield more.
2. **Without the T-Rex safety shield, CLM is much better** than either head (score 221 against 55 and 46).
3. **S1-Router gives up T-Rex completely** (0/5 courses with the shield on). Its 15-seed T-Rex run
   and DeepSWE were not run.
4. **Held-out gains are real but smaller than in-distribution ones.** For S1-Tweaks, tool choice is
   +15.7 points on Glaive (a training source) and +3.3 on BFCL (never seen). MMLU and MedMCQA are flat.
5. **This is slice-scale work, not a full retrain.** The post-training saw 233k choice rows for 400
   steps. The Q&A tests use a rebuilt 12k-pair slice with contexts cut to 512 tokens, so they sit
   below CLM's published 69%.
6. **DeepSWE is a tie within seed noise**: per seed 31, 32, 30, 31, 31 for CLM and 30, 32, 30, 31, 31
   for S1-Tweaks.

Raw numbers: [`results/innovations/`](https://github.com/samstealth/system-one-model-tweaks/tree/HEAD/results/innovations)
(`heldout.json`, `wise_0.15.json`, `post_choice_hn_frompost.json`, `auth_post.json`, `trex.json`, `deepswe.json`).
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
same as CLM-8B: the T-Rex runs measured a median of 16 to 17 ms per decision for every head on one L40S.

## How they were made

1. **Start from CLM's released head.** `CLM_v0.1-8B.pt` is the authors' post-trained head.
2. **Post-train a copy on choice data: this is S1-Router v0.1.** Run `post_choice_hn_frompost`: 400
   steps, batch 4,096, learning rate 0.0003, starting from CLM's head. Each training row is one state
   with the sibling options of the same question as hard negatives. 40% of every batch is replay of
   CLM's published Q&A vectors with their hard negatives. The choice data is 233k rows built by
   `innovations/build_choice_data.py` from Banking77, CLINC, Glaive function calling, ARC,
   CommonsenseQA, the MMLU auxiliary set and a label-reading set derived from them.
3. **Blend: this is S1-Tweaks v0.1.** `python -m innovations.interpolate --alpha 0.15` replaces every
   weight tensor of both heads, and the log scale, by `0.85 * CLM + 0.15 * S1-Router`. This is plain
   weight interpolation in the style of WiSE-FT (Wortsman et al., 2022). No gradient step follows it.

Why blend at all: S1-Router is far better at tool choice and routing, but it loses CLM's T-Rex skill
completely. The blend weight trades one for the other, and the trade is steep:

| Share of S1-Router in the blend | T-Rex courses survived, shield on, 5 seeds | BFCL (held-out) | MASSIVE (held-out) | Tool choice (Glaive) | Tricky Q&A |
|---|---|---|---|---|---|
| 0 (CLM-v0.1-8B) | 5/5 | 65.4% | 26.3% | 56.0% | 66.1% |
| **0.15 (S1-Tweaks v0.1)** | **5/5** (14/15 on 15 more seeds) | 68.7% | 31.7% | 71.7% | 66.6% |
| 0.175 | 3/5 | 69.5% | 32.2% | 73.2% | 66.8% |
| 0.2 | 4/5 | 69.7% | 33.3% | 75.0% | 67.0% |
| 0.25 | 4/5 | 70.5% | 35.5% | 78.0% | 66.9% |
| 0.5 | 0/5 | 74.8% | 45.8% | 90.3% | 67.5% |
| 1 (S1-Router v0.1) | 0/5 | 76.8% | 60.7% | 97.5% | 68.2% |

Other ways to keep T-Rex were tried and failed: distillation toward CLM's head (0/5), distillation
with a strong L2 anchor (5/5 but no gains at all), and mixing in 118k agent steps (2/5 with
distillation, 0/5 without). Details are in
[`results/innovations/day2_2026-11-22.md`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/day2_2026-11-22.md)
and [`day3_2026-11-23.md`](https://github.com/samstealth/system-one-model-tweaks/blob/HEAD/results/innovations/day3_2026-11-23.md).
All the experiments in the repo together cost about $69 on Modal.

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
pip install -e .
python examples/demo_cpu.py        # scores 60 real decisions with CLM's head and both of these
```

```python
from innovations.scorer import load_head

head = load_head("s1-tweaks")                # or "s1-router", or a local path to a .pt file
probs = head.probs(state_vec, option_vecs)   # Qwen3-8B vectors: (4096,), (K, 4096) -> (K,) probabilities

clm = load_head("clm")                       # CLM's official head from Hugging Face, for comparison
```

The loader reads the file with `torch.load(weights_only=True)`, so loading it cannot run code, and
it checks each download against the sha256 above. On the demo's 60 decisions CLM's head gets 29
right, S1-Tweaks 34 and S1-Router 47. A sample that small is for trying things, not for quoting.

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
vLLM vectors behind our results (cosine 0.9998 or higher on 60 states and 600 options), and 59 of
60 picks are the same.

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
  small, fragile weight change. A light blend keeps most of it and nothing we tried keeps all of it.
- **S1-Tweaks' gains on unseen sources are modest**: +1.8 to +5.4 points on tools, routing and
  science questions, and nothing on MMLU or MedMCQA. In absolute terms it is still a weak router
  (31.7% on MASSIVE with 10 options). If you need routing and do not need CLM's agent skill, use S1-Router.
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
  auxiliary set, and CLM's published Q&A vectors with hard negatives. See each dataset card for its licence.
- **These files** are released under Apache 2.0 by the GitHub user `samstealth`. They are not
  official CLM releases and the CLM authors have not reviewed them.

## Citation

If you use these heads, please cite CLM, which they are built on:

```bibtex
@misc{kwok2026contrastivelanguagemodels,
  title={Contrastive Language Models},
  author={Jacky Kwok and Hangoo Kang and Tarun Suresh and Jon Saad-Falcon and Marco Pavone and Christopher Ré and Azalia Mirhoseini},
  year={2026},
  note={Notion Blog},
  url={https://contrastive-lm.notion.site}
}
```

To point at these exact heads, link the repository and the release:
https://github.com/samstealth/system-one-model-tweaks (release `v0.1`, files `s1_tweaks_v0.1.pt` and `s1_router_v0.1.pt`).
