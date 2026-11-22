# Innovations on top of the System One rebuild

The workshop's last chapter ("Innovations you can build on top") suggests five experiments.
This folder holds the code for each, what we measured on a CPU, and the exact commands for the
GPU runs. Everything reuses the baseline's data, held-out chunks and test definitions, so
numbers are comparable with the main README.

| # | Idea | Status | Code |
|---|---|---|---|
| 1 | Swap Qwen3-8B for a newer open model | GPU scripts ready, not run | `build_qa_text.py`, `modal_embed.py --job qa`, `modal_train.py` |
| 2 | Tune the embedding dimensions | CPU sweep done at small scale; full run scripted | `heads.py`, `train.py`, `sweep_cpu.sh` |
| 3 | Quantize for speed | Heads and action cache measured on CPU; backbone FP8/AWQ scripted | `quantize_heads.py`, `modal_embed.py --job latency` |
| 4 | Shared embedding space | CPU sweep done at small scale; full run scripted | `heads.py --tower shared / shared_adapter` |
| 5 | Better agentic post-training | Data built (236k listwise rows); GPU embed and train scripted | `build_choice_data.py`, `modal_embed.py --job choice`, `train.py --hard-negs --replay-*` |

Results are in `RESULTS.md` and `../results/innovations/`.

## 1. A different backbone

What the video names, checked on Hugging Face on 22 November 2026:

| Model | Hub id | Size | Hidden | Licence | Fits one GPU? |
|---|---|---|---|---|---|
| Inkling 2 Small | `thinkingmachines/Inkling-2-Small` | 594B MoE, 17B active | 4608 | Apache 2.0 | No, about 500 GB VRAM |
| Inkling 2 Mini | `thinkingmachines/Inkling-2-Mini` | 26.4B MoE, 3.4B active, looped | 2304 | see model card | Yes (H100 80 GB) |
| Muse Spark Open | `meta-models/Muse-Spark-Open-816B-A42B` | 816B MoE, 42B active | 8192 | Muse licence, gated | No, about 700 GB VRAM |
| Muse Glimmer | `meta-models/Muse-Glimmer-30B` | 30B dense | 6656 | Apache 2.0 | Yes (H100 80 GB, bf16) |

A System One model runs the backbone on every decision, so latency is set by the active
parameters. Inkling 2 Mini, with 3.4B active, is the most promising swap: it could be faster
than Qwen3-8B and is newer. The two large models are research curiosities for this use.

Only Qwen3-8B vectors are published, so the comparison is controlled: rebuild 320k Q&A pairs as text
from the public Qwen token ids (`build_qa_text.py`, doc-disjoint 32k-pair holdout with Gemini hard
negatives), embed them with each backbone using the same last-token recipe, train identical heads,
and compare tricky-question accuracy and latency.

```bash
python -m innovations.build_qa_text --out qa_text            # CPU, about 10 minutes; copy to the volume
modal volume put clm-data qa_text /qa_text
modal run --detach innovations/modal_embed.py --job qa --backbone qwen3-8b
CLM_NEW_STACK=1 CLM_GPU=H100 modal run --detach innovations/modal_embed.py --job qa --backbone inkling-2-mini
CLM_NEW_STACK=1 CLM_GPU=H100 modal run --detach innovations/modal_embed.py --job qa --backbone muse-glimmer-30b --engine hf
for b in qwen3-8b inkling-2-mini muse-glimmer-30b; do
  modal run --detach innovations/modal_train.py --args "--root /data/emb/$b --out /data/runs/bb_$b --steps 4000 --batch 4096"
done
```

Both new architectures are newer than the `vllm==0.11.0` the repo pins, so they use an unpinned
stack (`CLM_NEW_STACK=1`) and fall back to plain transformers (`--engine hf`) if the pooling runner
rejects them. Inkling 2 Mini runs its middle layers three times; we pool the final pass.

## 2. Embedding dimensions

`HeadConfig` makes width, depth and output size knobs. One Matryoshka run (`--matryoshka
64,128,256,512,1024`) trains a single 1024-d head whose first k numbers are a usable k-d embedding,
so a single run gives the whole accuracy-versus-size curve. Smaller output vectors make the cached
action table smaller and the scoring step cheaper, which matters for a router over 10,000 tools.

## 3. Quantization

The backbone dominates latency, and the heads are small. We measured both parts we could on a CPU:
the heads in bf16 and int8, and the cached action vectors in int8 and 1 bit. For the backbone,
Qwen publishes official FP8 and AWQ 4-bit checkpoints. `modal_embed.py --job latency` times
one decision for each, and `--job qa --backbone qwen3-8b-fp8` re-embeds the holdout so the
authors' unchanged head can be scored on quantized vectors (does it still reach 69%?).

## 4. Shared embedding space

`--tower shared` uses one head for states and actions (half the parameters). `--tower shared_adapter`
shares the trunk and adds a 64-rank adapter per side (about 0.2M extra parameters), so the two
sides can still differ a little.

## 5. Post-training

Our diagnosis of the T-Rex gap (10% planner agreement against 66.8%): at serving time the engine
compares one state with the 3 to 10 options of the same question, written as `key: description`.
Our reconstructed post-training only ever contrasted an agent action with random actions from other
tasks, so the head never learned to separate sibling options. `build_choice_data.py` builds 236k
listwise rows in the engine's exact format, where the other options are the hard negatives:

| Source | Train | Test | What it teaches |
|---|---|---|---|
| banking77 | 10,003 | 3,080 | ticket routing, 77 teams |
| clinc_oos | 15,250 | 5,500 | intent routing, 150 intents |
| glaive function calling | 58,871 | 1,129 | which tool to call, incl. "no tool" |
| ARC, CommonsenseQA, MMLU aux | 112,953 | 2,393 | multiple choice reasoning |
| labeled options | 38,838 | 1,162 | reading inline labels such as "Recommended" or "Unsafe" |

The labeled source never uses T-Rex text, but it teaches the skill T-Rex's labeled prompt tests,
so a T-Rex score after training on it should be reported as in-domain. The held-out decision sets
(banking77, clinc, ARC, CSQA, tools) are a new benchmark for System One as an enterprise router:
`--choice-eval` reports top-1 per source.

```bash
python -m innovations.build_choice_data --src choice_src --out choice
modal volume put clm-data choice /choice
modal run --detach innovations/modal_embed.py --job choice --backbone qwen3-8b
modal run --detach innovations/modal_train.py --args "--root /data/emb/qwen3-8b --chunks shard_10/0,shard_10/1,shard_10/2 \
  --hard-negs --init /data/heads/qwen3_8b_midtrained_head.pt --replay-root /data \
  --replay-chunks shard_00/0,shard_00/1,shard_00/2 --replay-negs --replay-frac 0.4 \
  --choice-eval /data/emb/qwen3-8b --lr 0.0003 --steps 400 --batch 4096 --out /data/runs/post_choice"
modal run run_trex.py --tag choice --ckpt /data/runs/post_choice/final_authors_layout.pt
```

## Tests

`python -m pytest -q tests` runs every code path on synthetic vectors in about 15 seconds.
