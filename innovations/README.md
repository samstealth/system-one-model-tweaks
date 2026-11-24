# Innovations on top of the System One rebuild

The workshop's last chapter ("Innovations you can build on top") suggests five experiments.
This folder holds the code for each. All five have been run, on 22 and 23 November 2026: three
sweeps on a CPU and the rest on Modal GPUs. Everything reuses the baseline's data, held-out chunks
and test definitions.

- Numbers and their limits: [`RESULTS.md`](RESULTS.md). Raw result files: [`../results/innovations/`](../results/innovations/).
- Exact commands, hardware and cost for every run: [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).
- The two released heads, S1-Tweaks v0.1 and S1-Router v0.1: [`../MODEL_CARD.md`](../MODEL_CARD.md).

| # | Idea | Status | Code |
|---|---|---|---|
| 1 | Swap Qwen3-8B for a newer open model | Run as a screen: three backbones, same head, one small slice. No full-scale retrain. [RESULTS section 4](RESULTS.md#4-new-backbones-idea-1) | `build_qa_text.py`, `modal_embed.py --job qa`, `modal_train.py` |
| 2 | Tune the embedding dimensions | Run on a CPU at small scale. The full-scale run is scripted, not run. [Section 3](RESULTS.md#3-head-layout-embedding-dimensions-and-a-shared-space-ideas-2-and-4) | `heads.py`, `train.py`, `sweep_cpu.sh` |
| 3 | Quantize for speed | Run. Heads and action cache on a CPU. bf16, FP8 and AWQ backbones timed and scored on one L40S. [Section 2](RESULTS.md#2-quantization-idea-3) | `quantize_heads.py`, `modal_embed.py --job latency`, `modal_embed.py --job qa` |
| 4 | Shared embedding space | Run on a CPU at small scale. The full-scale run is scripted, not run. [Section 3](RESULTS.md#3-head-layout-embedding-dimensions-and-a-shared-space-ideas-2-and-4) | `heads.py`, `train.py --tower shared / shared_adapter` |
| 5 | Better agentic post-training | Run at slice scale: listwise post-training, T-Rex, DeepSWE, weight blends and a held-out test. Two heads released. [Sections 5 to 7](RESULTS.md#5-post-training-idea-5) | `build_choice_data.py`, `modal_embed.py --job choice`, `train.py --hard-negs --replay-*`, `interpolate.py`, `build_heldout_choice.py` |

## What is in this folder

| File | What it does |
|---|---|
| `scorer.py` | Public API: `load_head`, `Scorer`, `blend`. Scores options against a state on a CPU. |
| `heads.py` | The two-tower head with knobs for width, depth, output size, shared towers and Matryoshka. Reads and writes CLM's checkpoint layout. |
| `train.py` | One trainer for CPU and GPU: pre-training style runs, hard negatives, replay, distillation, an L2 anchor, eval only with `--steps 0`. |
| `data.py`, `evals.py` | Chunk reader and the two Q&A tests (tricky and general). |
| `interpolate.py` | Weight blend of two heads. S1-Tweaks v0.1 is made with it. |
| `quantize_heads.py` | bf16 and int8 heads, int8 and 1-bit action cache, on a CPU. |
| `sweep_cpu.sh` | The head layout sweep on a CPU. |
| `build_qa_text.py` | Rebuilds Q&A text from the public Qwen token ids, so any backbone can embed it. |
| `fetch_choice_src.py`, `build_choice_data.py` | Downloads the public sources and builds the listwise choice training and test rows. |
| `build_heldout_choice.py` | Builds the held-out choice test: seven sources no head here was trained on, plus the MMLU test split. |
| `heldout_paired.py` | Paired bootstrap and McNemar test of two heads on the held-out rows (Modal job). |
| `modal_embed.py`, `modal_train.py` | The GPU side on Modal: embed text with any backbone, time it, run the trainer. |
| `export_demo_sample.py` | Exports the 60 decisions behind `examples/demo_cpu.py`. |

## 1. A different backbone

What the video names, checked on Hugging Face on 22 November 2026:

| Model | Hub id | Size | Hidden | Licence | Fits one GPU? |
|---|---|---|---|---|---|
| Inkling 2 Small | `thinkingmachines/Inkling-2-Small` | 594B MoE, 17B active | 4608 | Apache 2.0 | No, about 500 GB VRAM |
| Inkling 2 Mini | `thinkingmachines/Inkling-2-Mini` | 26.4B MoE, 3.4B active, looped | 2304 | see model card | Yes, but so far only on a B200. See below. |
| Muse Spark Open | `meta-models/Muse-Spark-Open-816B-A42B` | 816B MoE, 42B active | 8192 | Muse licence, gated | No, about 700 GB VRAM |
| Muse Glimmer | `meta-models/Muse-Glimmer-30B` | 30B dense | 6656 | Apache 2.0 | Yes (H100 80 GB, bf16) |

A System One model runs the backbone on every decision, so latency is set by the active
parameters. Inkling 2 Mini, with 3.4B active, looked like the most promising swap on paper. In the
screen it scored lowest of the three and Muse Glimmer 30B scored highest
([RESULTS section 4](RESULTS.md#4-new-backbones-idea-1)). Latency was timed for the Qwen variants
only, so we do not know yet whether either new model is faster per decision. The two large models
are research curiosities for this use.

Only Qwen3-8B vectors are published, so the comparison is controlled: rebuild Q&A pairs as text
from the public Qwen token ids (`build_qa_text.py`, with a doc-disjoint holdout that keeps its
Gemini hard negatives), embed them with each backbone using the same last-token recipe, train
identical heads, and compare. The recorded run used a slice of 96k training pairs and 12k test
pairs, so it is a screening signal and not a final score.

```bash
python -m innovations.build_qa_text --out qa_text --train-docs 12000 --test-docs 1500 --max-context-tokens 512   # CPU
modal volume put clm-data qa_text /qa_text
modal run --detach innovations/modal_embed.py --job qa --backbone qwen3-8b
CLM_NEW_STACK=1 CLM_GPU=H100 modal run --detach innovations/modal_embed.py --job qa --backbone muse-glimmer-30b
CLM_IMAGE=vllm/vllm-openai:inkling-2-mini CLM_GPU=B200 modal run --detach innovations/modal_embed.py --job qa --backbone inkling-2-mini
for b in qwen3-8b muse-glimmer-30b inkling-2-mini; do
  modal run --detach innovations/modal_train.py --args "--root /data/emb/$b --out /data/runs/bb_$b --steps 3000 --batch 4096 --eval-every 1000"
done
```

Both new architectures are newer than the `vllm==0.11.0` the repo pins. Muse Glimmer ran on the
unpinned stack (`CLM_NEW_STACK=1`) on an H100. Inkling 2 Mini needed three more things: vLLM's own
Inkling image (`CLM_IMAGE`), a small loader patch that lives in `modal_embed.py`, and a B200,
because its custom kernel has no H100 build. The five attempts are written up in
[`../results/innovations/inkling_attempts.md`](../results/innovations/inkling_attempts.md).
`--engine hf` is a plain transformers fallback for models that vLLM cannot pool. Inkling 2 Mini
runs its middle layers three times; we pool the final pass.

## 2. Embedding dimensions

`HeadConfig` makes width, depth and output size knobs. One Matryoshka run (`--matryoshka
64,128,256,512,1024`) trains a single 1024-d head whose first k numbers are a usable k-d embedding,
so a single run gives the whole accuracy-versus-size curve. Smaller output vectors make the cached
action table smaller and the scoring step cheaper, which matters for a router over 10,000 tools.

Run so far: a CPU sweep at small scale ([RESULTS section 3](RESULTS.md#3-head-layout-embedding-dimensions-and-a-shared-space-ideas-2-and-4)).
The full-scale command is in the docstring of `modal_train.py`. Nobody has run it yet.

## 3. Quantization

The backbone dominates latency, and the heads are small. On a CPU we measured the heads in bf16
and int8, and the cached action vectors in int8 and 1 bit. For the backbone, Qwen publishes
official FP8 and AWQ 4-bit checkpoints. `modal_embed.py --job latency` times one decision for
each. `--job qa --backbone qwen3-8b-fp8` (or `qwen3-8b-awq`) re-embeds the test slice, so the
authors' unchanged head can be scored on quantized vectors. All of it has been run:
[RESULTS section 2](RESULTS.md#2-quantization-idea-3). In short, FP8 costs little accuracy and AWQ
costs more.

## 4. Shared embedding space

`--tower shared` uses one head for states and actions (half the parameters). `--tower shared_adapter`
shares the trunk and adds a 64-rank adapter per side (396,288 extra parameters in all, about 0.2M
per side), so the two sides can still differ a little.

Run so far: the same CPU sweep as idea 2 ([RESULTS section 3](RESULTS.md#3-head-layout-embedding-dimensions-and-a-shared-space-ideas-2-and-4)).
Whether the shared head keeps its lead after full training is an open question. A shared head
also cannot be exported to CLM's checkpoint layout, which has two separate heads.

## 5. Post-training

Where this started: the baseline's T-Rex gap (10% planner agreement against 66.8% for the authors'
head). At serving time the engine compares one state with the options of the same question. The
baseline's reconstructed post-training only ever contrasted an agent action with random actions
from other tasks, so the head never learned to separate sibling options. `build_choice_data.py`
builds 233k listwise training rows and 13k test rows with up to 10 options each, where the other
options of the same question are the hard negatives:

| Source | Train | Test | What it teaches |
|---|---|---|---|
| banking77 | 10,003 | 3,080 | ticket routing, 77 teams |
| clinc_oos | 15,250 | 5,500 | intent routing, 150 intents |
| glaive function calling | 58,871 | 1,129 | which tool to call, incl. "no tool" |
| ARC, CommonsenseQA, MMLU aux | 109,579 | 2,393 | multiple choice reasoning |
| labeled options | 38,838 | 1,162 | reading inline labels such as "Recommended" or "Unsafe" |
| Total | 232,541 | 13,264 | |

These counts are what the builder prints (and saves as `counts.json`) with its default seed. They
were checked by rebuilding on 23 November 2026, and they match the 233k/13k in the run record
(`../results/innovations/slice_run_2026-11-22.md`). An earlier version of this file said 236k: that
was before the builder started keeping every question of the labeled test split out of training.

**Option text.** The builder writes every option as `name: description`. That prefix is this
dataset's own convention. The CLM engine sends an option's description to the encoder exactly as
the caller wrote it, with no key in front (CLM `src/clm/schema.py`). So when you serve a head
trained on these rows through the engine, put `name: description` in the option's description.

**What the test rows are.** The test rows above are held-out rows from the same sources as the
training rows, so scores on them are in-distribution. `build_heldout_choice.py` builds a second,
test-only set from eight public sets: seven sources that no head here was trained on, plus the
MMLU test split (training used MMLU's `auxiliary_train` set). `--choice-eval` reports top-1 per
source for either set. The labeled source never uses T-Rex text, but it teaches the
skill T-Rex's labeled prompt tests, so a T-Rex score after training on it should be reported as
in-domain.

**What the runs showed.** The diagnosis was only half right. Listwise post-training lifts tool
choice and routing by a lot, but it erases the authors' T-Rex skill. That skill comes from their
agent post-training, not from telling sibling options apart. A light weight blend
(`interpolate.py`) keeps T-Rex survival with the safety shield on (19 of 20 games against 20 of
20). With the shield off it dies as often as a head with no agent training. The label-reading
rows did not help either: the post-trained head reads tags better and still lost the game. The
numbers, the failed attempts and the limits are in
[RESULTS sections 5 to 7](RESULTS.md#5-post-training-idea-5).

The recipe behind the two released heads, in short. The full version, with hardware and cost, is
in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

```bash
python -m innovations.fetch_choice_src --out choice_src
python -m innovations.build_choice_data --src choice_src --out choice
modal volume put clm-data choice /choice
modal run --detach innovations/modal_embed.py --job choice --backbone qwen3-8b
modal run innovations/modal_embed.py --job heads                      # the authors' heads
modal run innovations/modal_embed.py --job published --chunks 0,1     # replay vectors with hard negatives

# S1-Router v0.1 (run post_choice_hn_frompost): post-train the authors' post head
modal run --detach innovations/modal_train.py --args "--root /data/emb/qwen3-8b --chunks shard_10/0,shard_10/1,shard_10/2 \
  --hard-negs --init /data/heads/qwen3_8b_posttrained_head.pt --replay-root /data --replay-chunks shard_00/0,shard_00/1 \
  --replay-negs --replay-frac 0.4 --choice-eval /data/emb/qwen3-8b --lr 0.0003 --steps 400 --batch 4096 --eval-every 100 \
  --out /data/runs/post_choice_hn_frompost"

# S1-Tweaks v0.1 (run wise_0.15): 85% the authors' post head + 15% S1-Router, no training
modal volume get clm-data /heads/qwen3_8b_posttrained_head.pt qwen3_8b_posttrained_head.pt
modal volume get clm-data /runs/post_choice_hn_frompost/final_authors_layout.pt post_choice_hn_frompost.pt
python -m innovations.interpolate --a qwen3_8b_posttrained_head.pt --b post_choice_hn_frompost.pt --alpha 0.15 --out wise_0.15.pt
modal volume put clm-data wise_0.15.pt /runs/wise_0.15/final_authors_layout.pt

modal run pipeline/run_trex.py --tag wise_0.15 --ckpt /data/runs/wise_0.15/final_authors_layout.pt
```

## Tests

`python -m pytest -q tests` runs every code path on synthetic vectors in under a minute on a
laptop CPU. No GPU, no data and no network are needed.
