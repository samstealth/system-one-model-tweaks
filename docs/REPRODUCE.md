# Reproduce every result

Every measured number in this repo comes from one of the commands below and is saved in a result
file or a dated run note under `results/`. Guesses are marked as guesses. For each command this
guide gives the hardware, the wall-clock time where it was recorded, and the cost. It goes from
free to paid:

1. [Free, on a CPU](#1-free-on-a-cpu): the tests, the demo, a check of our eval code, head
   quantization, the head layout sweep and the data builders.
2. [GPU jobs on Modal](#2-gpu-jobs-on-modal): latency, embedding, the backbone screen,
   post-training, T-Rex and DeepSWE.
3. [Rebuild both released heads](#3-rebuild-both-released-heads-from-clms-public-head) from CLM's public head.
4. [Rerun the held-out test, T-Rex and DeepSWE](#4-rerun-the-held-out-test-t-rex-and-deepswe-on-any-head) on any head.

What the numbers mean and where they stop is in [`innovations/RESULTS.md`](../innovations/RESULTS.md).
The raw files are in [`results/innovations/`](../results/innovations/).

## How far to trust each command

Each command below carries one of three marks.

| Mark | Meaning |
|---|---|
| **recorded** | Copied from a run record in the repo: `day3_2026-11-23.md`, `inkling_attempts.md`, a script docstring or the note in a result file. |
| **rebuilt** | Written down after the run, from the run tables, the settings stored in each run's `result.json` (steps, batch, learning rate, number of chunks, hard negatives) and the run plan. The flags are right as far as we can tell. Nobody saved the exact line that was typed. |
| **checked** | Rerun on 23 November 2026 for this guide, on a machine with 4 CPU cores, 15 GB of RAM and no GPU. The output matched the committed result file. |

No GPU job was rerun for this guide. If you rerun one, please open a "Result replication" issue
and say what you got, matching or not.

## At a glance

| Step | Hardware | Wall-clock | Cost |
|---|---|---|---|
| [Tests](#11-tests) | any CPU | 28 s (checked) | free |
| [No-GPU demo](#12-the-no-gpu-demo) | any CPU | about 3 s after the first download (checked) | free |
| [Eval code against CLM's heads](#14-check-the-eval-code-against-clms-three-heads) | CPU, 11.5 GB of vectors | 17 to 32 s per head (checked) | free |
| [Head and cache quantization](#15-quantize-the-heads-and-the-action-cache) | CPU, same vectors | 2 min 30 s (checked) | free |
| [Head layout sweep](#16-head-layout-sweep) | CPU, 31 GB of vectors | 2 h 17 min for the seven recorded runs | free |
| [Data builders](#17-build-the-data-sets) | any CPU | 13 s, 16 s and 1 min 33 s (checked) | free |
| [Backbone latency](#22-backbone-latency) | one L40S per backbone | not recorded | day 1 |
| [Embed the Q&A slice](#23-embed-the-qa-slice-with-each-backbone) | L40S. H100 for Muse Glimmer. B200 for Inkling 2 Mini | Inkling: about 20 B200-minutes. Others not recorded | day 1 |
| [Backbone screen](#25-backbone-screen) | L40S | 1.2 to 2.9 trainer minutes per backbone | day 1 |
| [Post-training](#26-choice-embeddings-and-post-training) | L40S | 1.3 to 4.4 trainer minutes per run | day 1 |
| [T-Rex](#27-t-rex) | L40S | 60 s of play per seed and shield mode, plus start-up. Total not recorded | days 1 to 3 |
| [Blends and other attempts](#28-day-2-blends-distillation-anchor-agent-data) | CPU for the blend, L40S for the rest | 2.4 to 3.4 trainer minutes per training run | day 2 |
| [DeepSWE](#29-deepswe) | A10G | not recorded | days 2 and 3 |
| [Held-out test](#4-rerun-the-held-out-test-t-rex-and-deepswe-on-any-head) | CPU to build, L40S to embed and score | not recorded | day 3 |

"Trainer minutes" is the `minutes` field of a run's `result.json`: the training loop with its
evals, without container start-up.

### What it cost

Only day totals were recorded, from Modal's billing page.

| Round | What ran | Modal spend |
|---|---|---|
| Day 1, 22 Nov (`slice_run_2026-11-22.md`) | latency, embeddings, backbone screen, FP8 accuracy, post-training variants, first T-Rex runs | $42.33 metered, of which $30 was covered by included credits |
| Day 2, 22 Nov (`day2_2026-11-22.md`) | blends, distillation, anchor, agent data, T-Rex with the shield on and off, first DeepSWE seed | about $21 (the meter went from $42.33 to $63.15) |
| Day 3, 23 Nov (`day3_2026-11-23.md`) | held-out test, two more blends, T-Rex over 15 seeds, four more DeepSWE seeds, AWQ accuracy, demo sample | about $5 (the meter read $68.09 after the last job) |
| All three | | about $68 |

Cost and wall-clock per Modal job were not recorded. Only the totals per round above were.

The CPU steps cost nothing.

## 0. Set up

```bash
git clone https://github.com/samstealth/system-one-model-tweaks
cd system-one-model-tweaks
python -m venv .venv && source .venv/bin/activate

# Optional on Linux: the CPU build of torch is far smaller than the default wheel.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Python 3.10 or newer. Run every command in this guide from the repo root.

## 1. Free, on a CPU

### 1.1 Tests

```bash
python -m pytest -q tests
```

**checked**: 28 tests pass in 28 s, in a fresh venv with Python 3.11, torch 2.15.0, numpy 2.4.6 and
pyarrow 26.0.1. The tests use synthetic vectors. They need no GPU, no data and no network. CI runs
the same three install and test commands (`.github/workflows/ci.yml`).

### 1.2 The no-GPU demo

```bash
python examples/demo_cpu.py
```

It scores 60 real decisions whose Qwen3-8B vectors were computed in advance
(`examples/data/sample_decisions.npz` and `.jsonl`, 12 decisions each from `glaive_fc`,
`banking77`, `clinc_oos`, `heldout_bfcl` and `heldout_massive`). The first run downloads CLM's head
from Hugging Face.

- **checked**: CLM's head gets 29 of 60. Scoring takes about a second or less on an idle CPU and
  the whole script about 3 s.
- **recorded** (`results/innovations/demo_sample.json`): S1-Tweaks v0.1 gets 34 of 60 and S1-Router
  v0.1 gets 47, with the hits per source for each head.
- `python examples/demo_cpu.py --json my_demo.json` writes the hits per head and per source of your
  run in the same layout, so you can compare the two files.
- **recorded** (`results/innovations/demo_output.txt`): the full output with all three heads.
- If a release file cannot be downloaded, the demo says so and runs with the heads it has.

A sample of 60 is for trying things, not for quoting. The sample was exported with
`modal run innovations/export_demo_sample.py` (**recorded**). That needs the vectors on the Modal
volume, see section 4.

### 1.3 Get the published vectors

Sections 1.4 to 1.6 read CLM's published Qwen3-8B vectors and the authors' three heads. This
downloads the two test chunks and the heads into `./data`, about 11.7 GB:

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download

QA = ("Contrastive-LM/CLM-v0.1-Pretrain-Nemotron", "05ca7d05c043350326c5e86c589b9542962747b4")
NEG = ("jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings", None)
PAIRS = ("q.npy", "a.npy", "meta.parquet")


def fetch(src, kind, shard, idx, suffixes):
    repo, rev = src
    for suf in suffixes:
        hf_hub_download(repo, f"{shard}/chunk_{idx:04d}_{suf}", repo_type="dataset", revision=rev, local_dir=f"data/{kind}")


for f in ("qwen3_8b_pretrained_head.pt", "qwen3_8b_midtrained_head.pt", "qwen3_8b_posttrained_head.pt"):
    hf_hub_download("jackyk02/qwen3_8b_head", f, local_dir="data/heads")
fetch(QA, "qa", "shard_00", 28, PAIRS)                # tricky test: states and answers, 1.6 GB
fetch(NEG, "neg", "shard_00", 28, ("n.npy", "ncnt.npy"))  # its hard negatives, 8.2 GB
fetch(QA, "qa", "shard_05", 83, PAIRS)                # general test, 1.6 GB
# Only for the layout sweep in 1.6: twelve training chunks, 19.7 GB more.
# for i in range(12):
#     fetch(QA, "qa", "shard_05", i, PAIRS)
EOF
```

**checked** on the small files of each chunk (repo ids, revision and folder layout). The large files
were already on the check machine from the same calls.
`qwen3_8b_posttrained_head.pt` is the same file as the official `CLM_v0.1-8B.pt`: same size and same
sha256 (**checked**).

### 1.4 Check the eval code against CLM's three heads

This is [RESULTS section 1](../innovations/RESULTS.md#1-checking-the-eval-code-against-the-authors-heads).
`--steps 0` means eval only. `--chunks shard_05/83` only tells the trainer where to read the vector
size from.

```bash
for h in pretrained midtrained posttrained; do
  python -m innovations.train --root data --chunks shard_05/83 --init data/heads/qwen3_8b_${h}_head.pt \
    --steps 0 --hn-limit 20000 --out runs/auth_$h
done
```

**checked**: the last line of each run starts with `RESULT` and holds `tricky_top1` and `gen_top1`.
The rerun gave 51.3%, 68.8% and 68.2% tricky and 83.6%, 83.9% and 83.9% general, the numbers in
RESULTS section 1. Each head took 17 to 32 s. The three `result.json` files of a second rerun, on
24 November, are saved in `results/innovations/eval_check/`. The tricky test scores 19,136 rows:
the rows with hard negatives among the first 20,000.

### 1.5 Quantize the heads and the action cache

This is the third table of [RESULTS section 2](../innovations/RESULTS.md#2-quantization-idea-3).

```bash
python -m innovations.quantize_heads --root data --head data/heads/qwen3_8b_midtrained_head.pt --out runs/quant_heads_mid.json
```

**checked**: 2 min 30 s. All six accuracies matched `results/innovations/quant_heads_mid.json` to
four digits, and so did the sizes. The `head_ms_cpu` timings depend on the machine and its load.
One quirk: the script reads a file as a CLM-layout head only when its name contains `qwen3_8b`.

### 1.6 Head layout sweep

This is [RESULTS section 3](../innovations/RESULTS.md#3-head-layout-embedding-dimensions-and-a-shared-space-ideas-2-and-4):
embedding dimensions, Matryoshka and the shared head. It needs the twelve training chunks from 1.3
(`shard_05`, chunks 0 to 11), so 31 GB of vectors in all.

```bash
ROOT=data OUT=runs bash innovations/sweep_cpu.sh
```

Every run: the same 1.2M pairs, batch 1,024, 3,000 steps, seed 1234. **recorded**: trainer minutes
on a CPU, from `results/innovations/cpu_sweep/*.json`:

| Run | Layout | Minutes |
|---|---|---|
| `base_w1536_p512` | baseline: two heads, 1536 wide, 512 out | 22.4 |
| `shared_p512` | one shared head | 22.5 |
| `shared_adapter_p512` | shared trunk, 64-rank adapter per side | 19.4 |
| `mrl_p1024` | Matryoshka, 1024 out | 28.1 |
| `p128` | 128 out | 19.0 |
| `w768_p512` | 768 wide | 11.3 |
| `depth2_p512` | 2 layers | 14.8 |

That is 2 h 17 min of training. The sweep was not rerun in full for this guide. A 20-step run of
the same script (`STEPS=20 ROOT=data OUT=runs_smoke bash innovations/sweep_cpu.sh`) was **checked**
as a smoke test. All nine layouts ran and wrote a `result.json` in 8 min 5 s. The seven layouts
with a committed result file had the same settings and parameter counts as that file. Scores after
20 steps mean nothing.

The script also lists `p256` and `p1024`. No result files exist for them: the Matryoshka run reads
out those sizes. The script skips any run whose `result.json` already exists, so you can stop and
restart it.

### 1.7 Build the data sets

All three builders are deterministic. **checked**: they reproduce the row counts in the run records.

```bash
# Listwise choice data: 232,541 training rows and 13,264 test rows (the "233k/13k" of the run record).
python -m innovations.fetch_choice_src --out choice_src          # about 320 MB of public sources
python -m innovations.build_choice_data --src choice_src --out choice

# Held-out choice test: 13,194 rows from eight sources, 34 rows dropped for overlap with training.
python -m innovations.build_heldout_choice --out heldout --train choice/choice_train.jsonl

# Q&A text slice for the backbone and quantization runs: 95,996 training pairs and 11,998 test pairs.
pip install -e ".[text]"                                         # the builder needs the Qwen tokenizer
python -m innovations.build_qa_text --out qa_text --train-docs 12000 --test-docs 1500 --max-context-tokens 512
```

Times on the check machine: 13 s for the choice builder, 16 s for the held-out builder with its
29 MB of downloads, 1 min 33 s for the Q&A slice with its 2.1 GB of downloads. The rebuilt held-out
set has the same per-source counts as `results/innovations/heldout/counts.json`, and the rebuilt
test sets contain all 60 rows of the demo sample.

What the docs say about these data sets is counted by one script, from the built files:

```bash
python scripts/data_checks.py --choice choice --heldout heldout \
    --qa-test qa_text/qa_text_test.jsonl --out my_data_checks.json
```

**checked** on 24 November: this wrote `results/innovations/data_checks.json`. Per source it holds
the row count, the number of options, the chance level (the mean of 1 / number of options) and, for
Glaive, how many states carry earlier dialogue turns and an end-of-text marker (510 of 1,129 test
rows). With `--qa-test` it downloads three metadata files of about 0.5 MB and counts how many pairs
of the Q&A test slice are in the published chunks that post-training replayed: 424 of 11,998 pairs,
from 53 of 1,500 documents. A few seconds on a CPU.

## 2. GPU jobs on Modal

Everything below ran on [Modal](https://modal.com), against one Modal volume named `clm-data`,
mounted at `/data`.

```bash
pip install -e ".[modal]"
modal setup                    # once: log in
```

Things to know before you start:

- **GPU type.** `innovations/modal_embed.py` and `innovations/modal_train.py` use an L40S unless you
  set `CLM_GPU`. `pipeline/run_trex.py` uses an L40S and `pipeline/run_deepswe.py` an A10G.
- **Muse Glimmer 30B needs an H100** (`CLM_GPU=H100`) and the unpinned stack (`CLM_NEW_STACK=1`).
- **Inkling 2 Mini needs a B200 and a patched vLLM image.** It does not run on an H100: its custom
  kernel has no build for that GPU. It also needs vLLM's own Inkling image and a loader patch, which
  `modal_embed.py` applies. The five attempts are in
  [`results/innovations/inkling_attempts.md`](../results/innovations/inkling_attempts.md).
- **Pinned versions.** Qwen3-8B jobs use `vllm==0.11.0` with `transformers<5`. T-Rex and DeepSWE
  clone CLM at commit `d5f9ef0`. The encoder is `Qwen/Qwen3-8B` at revision `b968826`.
- **Detached jobs.** With `--detach` a job keeps running when your terminal closes. Read it with
  `modal app logs <app>`. The apps are `clm-innovations-embed`, `clm-innovations-train`, `clm-trex`
  and `clm-deepswe`.
- **Results.** Every training or eval run writes `/data/runs/<name>/result.json`. Copy it out with
  `modal volume get clm-data /runs/<name>/result.json results/innovations/<name>.json`.
- **GPU runs are not bit-reproducible.** Expect numbers close to the result files, not equal. We have
  not measured how much they move between reruns.

### 2.1 Put the data and the starting heads on the volume

```bash
modal volume put clm-data qa_text /qa_text
modal volume put clm-data choice /choice
modal run innovations/modal_embed.py --job heads                    # the authors' three heads into /data/heads
modal run innovations/modal_embed.py --job published --chunks 0,1   # replay vectors with hard negatives, about 10 GB per chunk
```

CPU only. The last two are **recorded** in the docstring of `modal_embed.py`.

### 2.2 Backbone latency

First table of [RESULTS section 2](../innovations/RESULTS.md#2-quantization-idea-3).
**recorded** (`results/innovations/backbone_latency_l40s.json`):

```bash
modal run innovations/modal_embed.py --job latency --backbone qwen3-8b,qwen3-8b-fp8,qwen3-8b-awq
```

One L40S per backbone, vLLM 0.11, batch 1, the mean of 30 runs after 3 warm-ups, at 64, 256 and
1,024 tokens. It times the backbone only.

### 2.3 Embed the Q&A slice with each backbone

Each job writes `/data/emb/<backbone>/`: the training pairs, plus the test pairs with their hard
negatives as the tricky test and again as the general test.

| Backbone | Command | GPU | Mark |
|---|---|---|---|
| Qwen3-8B bf16 | `modal run --detach innovations/modal_embed.py --job qa --backbone qwen3-8b` | L40S | rebuilt |
| Qwen3-8B FP8 | `modal run --detach innovations/modal_embed.py --job qa --backbone qwen3-8b-fp8` | L40S | rebuilt |
| Qwen3-8B AWQ | `modal run innovations/modal_embed.py --job qa --backbone qwen3-8b-awq --splits test` | L40S | recorded |
| Muse Glimmer 30B | `CLM_NEW_STACK=1 CLM_GPU=H100 modal run --detach innovations/modal_embed.py --job qa --backbone muse-glimmer-30b` | H100 | rebuilt |
| Inkling 2 Mini | `CLM_IMAGE=vllm/vllm-openai:inkling-2-mini CLM_GPU=B200 modal run --detach innovations/modal_embed.py --job qa --backbone inkling-2-mini` | B200 | recorded |

- `CLM_NEW_STACK=1` installs the newest vllm and transformers when the image is built. On
  22 November 2026 that was vllm 0.34.0 and transformers 5.22.2 (`inkling_attempts.md`). A later
  rerun gets newer versions. `--engine hf` is a plain transformers fallback if vLLM's pooling runner
  rejects a model.
- Inkling 2 Mini, **recorded**: the engine took about 3.5 minutes to start and the 96k training pairs
  about 12.5 minutes. The working attempt used about 20 B200-minutes. The failed attempts before it
  used about 25 H100-minutes.
- The AWQ job embeds the test pairs only. That is all the accuracy check needs.

### 2.4 Accuracy on a quantized backbone

Second table of [RESULTS section 2](../innovations/RESULTS.md#2-quantization-idea-3): the authors'
mid-trained head, unchanged, scored on each backbone's test vectors. The first two are **rebuilt**,
the third is **recorded**.

```bash
modal run innovations/modal_train.py --args "--root /data/emb/qwen3-8b --init /data/heads/qwen3_8b_midtrained_head.pt --steps 0 --out /data/runs/auth_bf16"
modal run innovations/modal_train.py --args "--root /data/emb/qwen3-8b-fp8 --init /data/heads/qwen3_8b_midtrained_head.pt --steps 0 --out /data/runs/auth_fp8"
modal run innovations/modal_train.py --args "--root /data/emb/qwen3-8b-awq --chunks shard_05/83 --init /data/heads/qwen3_8b_midtrained_head.pt --steps 0 --out /data/runs/auth_awq"
```

Results: `auth_bf16.json`, `auth_fp8.json`, `auth_awq.json`. The tricky test here has 7,848 rows.

### 2.5 Backbone screen

[RESULTS section 4](../innovations/RESULTS.md#4-new-backbones-idea-1): the same head trained from
scratch on each backbone's vectors of the same slice. **rebuilt**:

```bash
for b in qwen3-8b muse-glimmer-30b inkling-2-mini; do
  modal run --detach innovations/modal_train.py --args "--root /data/emb/$b --out /data/runs/bb_$b --steps 3000 --batch 4096 --eval-every 1000"
done
```

Trainer minutes on an L40S, from `bb_*.json`: 1.95 for Qwen3-8B, 2.86 for Muse Glimmer 30B, 1.22
for Inkling 2 Mini.

Without `--chunks` the trainer uses every training chunk under `--root`, except the two test chunks
and the choice shards. In the recorded runs that was one chunk, `shard_00/0`, the whole Q&A slice
(`"chunks": 1` in `bb_*.json`). If you have already run the agent-data job of 2.8, it has added
`shard_20` chunks under `/data/emb/qwen3-8b`. Then add `--chunks shard_00/0` to train on the Q&A
slice alone.

### 2.6 Choice embeddings and post-training

[RESULTS section 5](../innovations/RESULTS.md#5-post-training-idea-5). First embed the choice rows
with Qwen3-8B. The job writes the training rows to `qa/shard_10` and the test rows to `qa/shard_11`
under `/data/emb/qwen3-8b`, each with its sibling options as hard negatives. It resumes if it is cut off.

```bash
modal run --detach innovations/modal_embed.py --job choice --backbone qwen3-8b
```

The post-training run behind **S1-Router v0.1** is `post_choice_hn_frompost`. **rebuilt**:

```bash
modal run --detach innovations/modal_train.py --args "--root /data/emb/qwen3-8b --chunks shard_10/0,shard_10/1,shard_10/2 \
  --hard-negs --init /data/heads/qwen3_8b_posttrained_head.pt --replay-root /data --replay-chunks shard_00/0,shard_00/1 \
  --replay-negs --replay-frac 0.4 --choice-eval /data/emb/qwen3-8b --lr 0.0003 --steps 400 --batch 4096 --eval-every 100 \
  --out /data/runs/post_choice_hn_frompost"
```

In words: 400 steps, batch 4,096, learning rate 0.0003, from the authors' post-trained head. 60%
of each batch is choice rows with their sibling options as hard negatives. 40% is replay of the
published Q&A vectors with their hard negatives. `result.json` confirms the steps, the batch, the
learning rate, the three choice chunks and the hard negatives. The replay flags come from the run
plan and the run table.

The other runs in the table of `slice_run_2026-11-22.md` change only these flags. All **rebuilt**:

| Run | Change from the command above | Trainer minutes |
|---|---|---|
| `auth_post` (the baseline to beat) | eval only: `--root /data/emb/qwen3-8b --init /data/heads/qwen3_8b_posttrained_head.pt --steps 0 --choice-eval /data/emb/qwen3-8b` | eval only |
| `post_choice` | `--init /data/heads/qwen3_8b_midtrained_head.pt`, replay from our own vectors without negatives: `--replay-root /data/emb/qwen3-8b --replay-chunks shard_00/0`, no `--replay-negs` | 2.27 |
| `post_choice_hn` | `--init /data/heads/qwen3_8b_midtrained_head.pt` | 2.43 |
| `post_choice_hn_frompost` | none | 3.53 |
| `post_from_post_gentle` | `--lr 0.0001 --steps 200 --replay-frac 0.5` | 1.88 |
| `post_choice_bb_qwen`, `post_choice_muse-glimmer-30b`, `post_choice_inkling-2-mini` | choice rows embedded with that backbone (`--job choice --backbone <b>`, same environment variables as in 2.3), then `--root`, `--replay-root` and `--choice-eval` all `/data/emb/<b>`, `--init /data/runs/bb_<b>/final.pt`, `--replay-chunks shard_00/0`, no `--replay-negs` | 2.99, 4.44, 1.33 |

`shard_00/0` is the one training chunk of the Q&A slice (`"chunks": 1` in `bb_qwen3-8b.json`). The
docstring of `innovations/modal_train.py` shows the `post_choice` command with three replay chunks,
`shard_00/0` to `shard_00/2`. On the slice recorded here only the first one exists.

### 2.7 T-Rex

```bash
modal run pipeline/run_trex.py --tag authors --ckpt authors                                   # CLM's own head
modal run pipeline/run_trex.py --tag <run> --ckpt /data/runs/<run>/final_authors_layout.pt    # any head on the volume
```

One L40S. The job starts the Qwen3-8B encoder and `clm-serve --ckpt <head>` in one container, then
plays the official courses in real time with the labeled prompt: 5 seeds of 60 s with the safety
shield on, then the same 5 seeds with it off. Summaries land in `/data/trex/<tag>/shield.json` and
`noshield.json`. The form of the command is **recorded** in `slice_run_2026-11-22.md`. The tag of
each run is **rebuilt**.

The 15-seed runs of day 3 are **recorded**:

```bash
modal run pipeline/run_trex.py --tag wise_0.15_15seeds --ckpt /data/runs/wise_0.15/final_authors_layout.pt --seeds 15 --shield-only
```

Three things to know when you compare your run with ours:

- **Seeds overlap.** `--seeds N` plays seeds 0 to N-1. The 5-seed runs are seeds 0 to 4 and the
  15-seed runs are seeds 0 to 14 (`results/innovations/trex_15seeds/`). So seeds 0 to 4 were played
  twice. The "19 of 20 games" for `wise_0.15` counts both runs: 15 different courses, 20 games.
- **Real-time play is not deterministic.** `wise_0.15` survived seed 0 in the 5-seed run and died on
  it in the 15-seed run. One game either way is inside the noise.
- **Always report both shield modes.** With the shield off the picture changes a lot. See RESULTS section 6.
- **What the shield count is.** `shield_interventions` in `trex.json` is the sum of three per-seed
  counters of CLM's harness: answers that the shield's check changed when they arrived, top picks
  that the planner had labelled unsafe and were replaced, and emergency moves. The per-seed files in
  `results/innovations/trex_15seeds/` and `trex_per_seed/` hold them one by one.

The wall-clock and the cost of a single T-Rex run were not recorded.

### 2.8 Day 2: blends, distillation, anchor, agent data

[RESULTS section 6](../innovations/RESULTS.md#6-keeping-t-rex-the-launch-head) and `day2_2026-11-22.md`.

**Blends.** A blend is made on a CPU in seconds, then scored like any other head. The
`interpolate` line is **recorded** (`day2_2026-11-22.md`, `day3_2026-11-23.md`). The rest is **rebuilt**.

```bash
modal volume get clm-data /heads/qwen3_8b_posttrained_head.pt qwen3_8b_posttrained_head.pt
modal volume get clm-data /runs/post_choice_hn_frompost/final_authors_layout.pt post_choice_hn_frompost.pt
python -m innovations.interpolate --a qwen3_8b_posttrained_head.pt --b post_choice_hn_frompost.pt --alpha 0.15 --out wise_0.15.pt
modal volume put clm-data wise_0.15.pt /runs/wise_0.15/final_authors_layout.pt
modal run innovations/modal_train.py --args "--root /data/emb/qwen3-8b --init /data/runs/wise_0.15/final_authors_layout.pt --steps 0 --choice-eval /data/emb/qwen3-8b --out /data/runs/wise_0.15"
```

The same with `--alpha` 0.175, 0.2, 0.25 and 0.5. The eval-only run saves the head again into its
output folder. The weights do not change (**checked** on a CPU: largest difference 0.0).

**Distillation and anchor.** The S1-Router command of 2.6 with these additions. **rebuilt**:

| Run | Added flags | Trainer minutes |
|---|---|---|
| `post_distill` | `--distill-from /data/heads/qwen3_8b_posttrained_head.pt --distill-weight 1.0` | 3.23 |
| `post_distill3_anchor10` | `--distill-from /data/heads/qwen3_8b_posttrained_head.pt --distill-weight 3.0 --anchor-weight 10` | 3.42 |

**Agent data.** A slice of the baseline's ADP reconstruction: the first 4,000 pairs of each of 34
text configs, which gives 118,465 training pairs (**recorded** in `day2_2026-11-22.md`). **rebuilt**:

```bash
modal run --detach pipeline/build_agent_data.py                       # ADP recordings to (state, action) pairs, CPU
CLM_AGENT_LIMIT=4000 modal run --detach pipeline/embed_agent.py       # L40S, up to 8 containers
modal run innovations/modal_embed.py --job agent --src /data/agent/vec_4000   # into trainer chunks qa/shard_20
```

| Run | Change from the S1-Router command | Trainer minutes |
|---|---|---|
| `post_agent_mix` | `--chunks shard_10/0,shard_10/1,shard_10/2,shard_20/0,shard_20/1`, plus `--distill-from /data/heads/qwen3_8b_posttrained_head.pt --distill-weight 1.0` | 2.44 |
| `post_agent_mix_nodistill` | the same chunks, no distillation | 3.07 |
| `wise_agent_0.5` | no training: `interpolate --alpha 0.5` between the authors' post head and `post_agent_mix` | none |

The replay fraction of the two agent-mix runs was 0.4. It was passed on the launch command line (`--replay-frac 0.4`) and is not stored in the runs' `result.json`, so the command line is its only record.

### 2.9 DeepSWE

**recorded** (`day3_2026-11-23.md`, `results/innovations/deepswe.json`):

```bash
modal run pipeline/run_deepswe.py --tag authors_s1 --init authors --seed 1
modal run pipeline/run_deepswe.py --tag wise_0.15_s1 --init /data/runs/wise_0.15/final_authors_layout.pt --seed 1
```

One A10G. The job fine-tunes the head with CLM's own `finetune.py` on the passing DeepSWE steps,
then counts how many of the 38 held-out tasks its best-of-4 pick solves. So the score is for the
head after that fine-tune, not for the head as shipped. Seeds used: 1234 (the
default), 1, 2, 3 and 4. The tag names are ours to choose. Only the first one above is **rebuilt**.

## 3. Rebuild both released heads from CLM's public head

| | S1-Router v0.1 | S1-Tweaks v0.1 |
|---|---|---|
| Run name | `post_choice_hn_frompost` | `wise_0.15` |
| Made by | post-training CLM's head | blending 85% CLM's head with 15% S1-Router |
| Needs | one L40S for a few minutes, plus the embedding jobs | a CPU for a few seconds |
| Released file | `s1_router_v0.1.pt`, 75,557,108 bytes | `s1_tweaks_v0.1.pt`, 75,557,108 bytes |

```text
sha256  d9f22c9fe0a52d4bd32fe1a6c965b3900fc854c44f1c6267f23bf6e897911b18  s1_router_v0.1.pt
sha256  c0a32217d1288d8452046bda9bcb045b53b052b943ef7de42652691e057fd9d9  s1_tweaks_v0.1.pt
```

CLM's public head is `CLM_v0.1-8B.pt` in
[Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B). The commands below
call it `qwen3_8b_posttrained_head.pt`, the name under which `--job heads` stores the same file.

```bash
# 1. Data, on a CPU (section 1.7)
python -m innovations.fetch_choice_src --out choice_src
python -m innovations.build_choice_data --src choice_src --out choice

# 2. Volume, heads, replay vectors and choice embeddings (sections 2.1 and 2.6). One L40S.
modal volume put clm-data choice /choice
modal run innovations/modal_embed.py --job heads
modal run innovations/modal_embed.py --job published --chunks 0,1
modal run --detach innovations/modal_embed.py --job choice --backbone qwen3-8b

# 3. S1-Router v0.1: post-train CLM's head. One L40S, 3.5 trainer minutes in our run.
modal run --detach innovations/modal_train.py --args "--root /data/emb/qwen3-8b --chunks shard_10/0,shard_10/1,shard_10/2 \
  --hard-negs --init /data/heads/qwen3_8b_posttrained_head.pt --replay-root /data --replay-chunks shard_00/0,shard_00/1 \
  --replay-negs --replay-frac 0.4 --choice-eval /data/emb/qwen3-8b --lr 0.0003 --steps 400 --batch 4096 --eval-every 100 \
  --out /data/runs/post_choice_hn_frompost"
modal volume get clm-data /runs/post_choice_hn_frompost/final_authors_layout.pt my_s1_router.pt

# 4. S1-Tweaks v0.1: blend on a CPU. No training.
modal volume get clm-data /heads/qwen3_8b_posttrained_head.pt qwen3_8b_posttrained_head.pt
python -m innovations.interpolate --a qwen3_8b_posttrained_head.pt --b my_s1_router.pt --alpha 0.15 --out my_s1_tweaks.pt
python -m innovations.scorer my_s1_router.pt my_s1_tweaks.pt     # prints the layout of each file
```

Step 3 also prints the tricky and general Q&A scores if the Q&A slice is on the volume
(`--job qa --backbone qwen3-8b`, section 2.3). Without it, only the choice scores are printed.

`interpolate` reads both files with `torch.load(weights_only=True)`. Its `--a` and `--b` also take
the names `clm`, `s1-tweaks` and `s1-router`, which download the public files. So
`python -m innovations.interpolate --a clm --b s1-router --alpha 0.15 --out my_blend.pt`
makes the blend with no Modal volume.

What to expect:

- **Your files will not have the published sha256.** Post-training on a GPU does not repeat bit for
  bit, and the checkpoint container differs between torch versions. Compare weights and scores, not
  hashes.
- **The blend itself can be checked exactly.** Download the released
  S1-Router, blend it with CLM's head at 0.15, and compare with the released S1-Tweaks:

  ```bash
  python - <<'EOF'
  import torch
  from huggingface_hub import hf_hub_download
  from innovations.interpolate import mix
  from innovations.scorer import download_s1_router, download_s1_tweaks

  load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
  clm = load(hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt"))
  mine, released = mix(clm, load(download_s1_router()), 0.15), load(download_s1_tweaks())
  diff = max(float((mine[s][k] - released[s][k].float()).abs().max()) for s in ("state_head", "action_head") for k in released[s])
  print("largest weight difference:", diff)
  EOF
  ```

  Run on 24 November 2026 on the files that became the release, with the same sha256 as the pinned ones, it
  printed 0.0 ([`blend_check.json`](../results/innovations/blend_check.json)).
- **Scores.** Score your heads as in section 4 and compare with `results/innovations/heldout.json`,
  `post_choice_hn_frompost.json` and `wise_0.15.json`.

## 4. Rerun the held-out test, T-Rex and DeepSWE on any head

A head must be on the volume in CLM's checkpoint layout. For a released file:

```bash
modal volume put clm-data s1_tweaks_v0.1.pt /runs/s1_tweaks/final_authors_layout.pt
```

**Held-out choice test** ([RESULTS section 7](../innovations/RESULTS.md#7-held-out-choice-test-built-by-us-from-public-sets)).
**recorded** in `day3_2026-11-23.md`. The build is free. The embedding and the scoring use an L40S.

```bash
python -m innovations.build_heldout_choice --out heldout --train choice/choice_train.jsonl
modal volume put clm-data heldout /heldout
modal run innovations/modal_embed.py --job choice --src /data/heldout --root /data/emb/qwen3-8b-heldout
modal run innovations/modal_train.py --args "--root /data/emb/qwen3-8b --init /data/runs/s1_tweaks/final_authors_layout.pt --steps 0 --choice-eval /data/emb/qwen3-8b-heldout --out /data/runs/heldout_s1_tweaks"
```

The `RESULT` line holds one `choice_heldout_<source>` score per source. Ours are in
`results/innovations/heldout/` and, as percentages, in `heldout.json`. The paired test of two heads
on the same rows (bootstrap interval and McNemar p-value per source) is one more CPU job on Modal,
**recorded** in the docstring of the script. It wrote `results/innovations/heldout_paired.json`:

```bash
modal run innovations/heldout_paired.py
```

 The held-out vectors sit in
their own folder, so no training run can read them by accident. `--root` needs at least one training
chunk to read the vector size from. If you skipped the Q&A slice, add `--chunks shard_10/0`.

**T-Rex** (section 2.7). One L40S per run:

```bash
modal run pipeline/run_trex.py --tag s1_tweaks --ckpt /data/runs/s1_tweaks/final_authors_layout.pt                          # 5 seeds, shield on, then off
modal run pipeline/run_trex.py --tag s1_tweaks_15seeds --ckpt /data/runs/s1_tweaks/final_authors_layout.pt --seeds 15 --shield-only
modal run pipeline/run_trex.py --tag authors --ckpt authors                                                              # CLM's head, for the comparison
```

Compare with `results/innovations/trex.json` and `trex_15seeds/`. To play on your own GPU instead,
serve the head with the official CLM engine (`MODEL_CARD.md`, "Serve a head with the official CLM
engine") and run CLM's `examples/t_rex/run.py`. We only ran it through Modal.

**DeepSWE** (section 2.9). One A10G per seed:

```bash
for seed in 1234 1 2 3 4; do
  modal run pipeline/run_deepswe.py --tag s1_tweaks_s$seed --init /data/runs/s1_tweaks/final_authors_layout.pt --seed $seed
done
```

**The demo sample.** **recorded**: `modal run innovations/export_demo_sample.py` writes the 60
decisions of `examples/data/` from the choice and held-out vectors on the volume.

## 5. Figures

```bash
pip install matplotlib
python scripts/make_figures.py
```

It reads `results/innovations/` and rewrites every figure in `assets/`. Run it after a result file changes.

## 6. The baseline at full scale

The Vizuara baseline (pre-training, mid-training and its own post-training on the full published
vectors) is described in [`docs/BASELINE.md`](BASELINE.md). It needs about 1.3 TB of Modal volume
storage. Nothing in this guide needs it. The full-scale runs of the shared head and of Matryoshka
would: their commands are in the docstring of `innovations/modal_train.py`, and nobody has run them yet.

## 7. What this guide cannot give you yet

- **Cost and wall-clock per GPU job.** Only day totals were recorded.
- **The exact day 1 and day 2 command lines.** They are rebuilt, as marked. Day 3 is recorded.
- **A second pair of hands.** No GPU result here has been rerun by someone else. The CPU results
  marked "checked" were rerun once, on one machine.
- **Run-to-run spread.** Each training run was done once, with seed 1234. T-Rex is the one test with
  repeated games, and it moved by one game between two runs.

If you fill one of these gaps, open a "Result replication" issue or a pull request against this file.
