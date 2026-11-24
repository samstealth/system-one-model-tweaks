# v0.1: S1-Tweaks and S1-Router heads for CLM-8B

Two drop-in heads for [CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), the open "System One" decision model from Stanford and NVIDIA Research. Both files use CLM's own checkpoint layout (`state_head`, `action_head`, `logit_scale`, `cfg`), so the official CLM engine serves either one with `--ckpt`. This is an independent project. It is not an official CLM release.

## Files

| File | What it is |
|---|---|
| `s1_tweaks_v0.1.pt` | **S1-Tweaks v0.1**, the default. Every weight is 85% CLM-8B's head and 15% S1-Router's, with no training after the blend. |
| `s1_router_v0.1.pt` | **S1-Router v0.1**, the routing specialist. CLM-8B's head plus 400 post-training steps on multiple-choice data. |

Both files are 75,557,108 bytes.

```text
sha256  s1_tweaks_v0.1.pt  c0a32217d1288d8452046bda9bcb045b53b052b943ef7de42652691e057fd9d9
sha256  s1_router_v0.1.pt  d9f22c9fe0a52d4bd32fe1a6c965b3900fc854c44f1c6267f23bf6e897911b18
```

You still need the frozen Qwen3-8B backbone to turn new text into vectors. Only the heads changed.

## Try it

```bash
git clone https://github.com/samstealth/system-one-model-tweaks
cd system-one-model-tweaks
python -m venv .venv && source .venv/bin/activate
# Linux, optional: the CPU build of torch skips the large CUDA wheels
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
# no GPU needed: scores 60 real decisions on your CPU with all three heads
python examples/demo_cpu.py
```

The demo downloads both files from this release and checks their sha256 before it loads them.

## Measured results

Top-1 accuracy in %. Same backbone, same vectors and same scoring code for every head. The held-out tests are our own multiple-choice builds from public sets, so they are not the benchmarks' own metrics.

| Held-out test (rows) | Chance | CLM-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|---|
| BFCL v3, our 10-option build (1,252) | 10.0 | 65.4 | 68.7 | 76.8 |
| MASSIVE, intent routing (2,951) | 10.0 | 26.3 | 31.7 | 60.7 |
| HWU64, intent routing (1,072) | 10.0 | 29.5 | 34.4 | 57.3 |
| Bitext, support routing (2,000) | 10.0 | 16.1 | 17.9 | 39.8 |
| SciQ, science questions (1,000) | 25.0 | 56.5 | 60.3 | 69.8 |
| QASC, science questions (923) | 12.5 | 36.4 | 39.0 | 48.6 |
| MMLU test, mixed knowledge (1,997) | 25.0 | 28.9 | 29.2 | 31.6 |
| MedMCQA, medical questions (1,999) | 25.0 | 29.0 | 29.1 | 30.0 |

| CLM-8B's own tests | CLM-8B | S1-Tweaks v0.1 | S1-Router v0.1 |
|---|---|---|---|
| T-Rex games survived, safety shield on | 20 of 20 | 19 of 20 | 0 of 5 |
| T-Rex deaths in 5 courses, safety shield off | 22 | 45 | 45 |
| General Q&A on our rebuilt slice | 81.8 | 81.7 | 79.9 |

On a paired test over the same rows, S1-Tweaks v0.1 gains on six of the eight held-out tests. Pooled over those six (9,198 rows) the gain is 3.84 points, with a 95% bootstrap interval of 3.39 to 4.29. MMLU and MedMCQA do not move.

## Read this before you rely on them

- **T-Rex is a near-tie, not a tie, and only with the benchmark's safety shield on.** With the shield off CLM-8B is far better, and no head survives a full course.
- **S1-Router v0.1 lost CLM's real-time game skill.** Use it for routing and tool choice only.
- **The held-out gains of S1-Tweaks v0.1 are modest.** The option wording of the tests is ours, CLM-8B never saw it, and no other wording was tested.
- **This is slice-scale work**: 400 post-training steps on about 233k choice rows, not a full retrain.

Full tables, limits and how each head was made: [`MODEL_CARD.md`](https://github.com/samstealth/system-one-model-tweaks/blob/main/MODEL_CARD.md) and [`innovations/RESULTS.md`](https://github.com/samstealth/system-one-model-tweaks/blob/main/innovations/RESULTS.md).

## Licence and credit

Apache 2.0. Both heads are modified copies of the [CLM-v0.1-8B](https://github.com/Contrastive-LM/CLM) weights (Apache 2.0), and S1-Tweaks v0.1 is 85% those weights. The pipeline builds on the [Vizuara System One workshop](https://github.com/VizuaraAI/system-one-from-scratch). See [`NOTICE`](https://github.com/samstealth/system-one-model-tweaks/blob/main/NOTICE).
