# Contributing

This repo is a set of measured experiments on a System One decision model. The contributions that help
most are:

1. A rerun of one of our numbers, matching or not.
2. A new experiment with its result file, including one that fails.
3. A fix to code or docs.

You do not need a GPU for the first steps. The tests, the no-GPU demo and three of the experiments run on a laptop CPU.

## Set up

```bash
git clone https://github.com/samstealth/system-one-model-tweaks
cd system-one-model-tweaks
python -m venv .venv && source .venv/bin/activate

# Optional on Linux: the CPU build of torch is far smaller than the default wheel.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Python 3.10 or newer works. CI uses Python 3.11. Two more extras exist: `text` for
`examples/route_text.py` (needs a GPU) and `modal` for the GPU runs.

## Run the tests

```bash
python -m pytest -q tests
python examples/demo_cpu.py      # optional: 60 real decisions scored on your CPU
```

The tests use synthetic vectors. They need no GPU, no data and no network, and they finish in under a
minute. CI runs the same command on every push and pull request (`.github/workflows/ci.yml`, workflow `tests`).

If you add a code path, add a test for it on synthetic vectors. `tests/test_innovations.py` shows the
pattern: `make_root` writes a tiny vector store in a temp folder and the test trains on it for a few steps.

## The rule: every number points to a result file

Readers of this repo are skeptical, and they are right to be. So:

- **Every number in `README.md`, `MODEL_CARD.md`, `docs/` and `innovations/RESULTS.md` must be readable
  from a file under `results/`: a result JSON, or a dated run note for things like costs and commands.**
  New work goes in `results/innovations/`.
- The trainer writes a `result.json` for every run. Copy it to `results/innovations/<run name>.json` and
  commit it with the docs change. T-Rex summaries go in `trex.json`, DeepSWE counts in `deepswe.json`,
  held-out choice scores in `heldout.json`, paired intervals in `heldout_paired.json`, and row counts,
  option counts and chance levels of the data sets in `data_checks.json` (`python scripts/data_checks.py`).
- State the limits next to the claim: how many seeds, how many test rows, and whether the test rows come
  from sources the head was trained on.
- Say "near-tie" when it is one. 19 of 20 games against 20 of 20 is not a match.
- Not measured yet? Write `<!-- PENDING: what is missing -->`. Never fill a gap with an estimate. If
  you must give a guess, call it a guess in the same sentence.
- Never edit a measured number in `results/`. If a write-up there turns out to be wrong, add a dated
  correction note under it, as in `results/innovations/day2_2026-11-22.md`.
- Charts are built from the result files. Run `python scripts/make_figures.py` after a result file
  changes. Do not edit the SVGs by hand.
- Add the exact command, the hardware, the wall-clock time and the cost to `docs/REPRODUCE.md`. Several
  day-1 and day-2 commands had to be rebuilt from run tables afterwards. Save yours when you run it.
- A result that did not work is still a result. Put it in the table. Several rows in
  `results/innovations/day2_2026-11-22.md` are failures, and they are the reason the blend is trusted.

## Style

- Plain words and short sentences. No hype words.
- Core code keeps to numpy, torch, pyarrow, huggingface_hub and the standard library. Heavier
  dependencies (modal, vllm, transformers, matplotlib) are imported only by the scripts that need them.
- Do not commit weights or vectors. `*.pt` and `*.npy` are ignored on purpose. The one exception is the
  small demo sample in `examples/data/`.
- Keep pull requests small: one experiment or one fix.

## Open questions you can pick up

Each item says what is known, where it is written, and where to start. Open an "Experiment idea" issue
to say you are working on one, so two people do not spend money on the same run. Commands and costs for
everything already run are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

### The big ones

- **Keep T-Rex with the safety shield off.** With the shield off, CLM-8B dies 22 times over 5 seeds and
  scores 221. Every blend and every post-trained head dies 45 times (`results/innovations/trex.json`,
  `results/innovations/day2_2026-11-22.md`). The only head that kept 22 deaths was the one anchored so
  hard that it gained nothing. Is there a blend or a method that gains on the choice tests and keeps
  this? Not tried yet: blend weights below 0.15, blending only the state head or only the action head,
  a different weight per layer, low-rank updates. `innovations/interpolate.py` is about 70 lines. Needs one
  L40S for a few minutes per head.
- **Agent data that restores CLM's T-Rex skill.** Our agent mix (118,465 training pairs from 34 ADP
  configs) reached 2/5 T-Rex courses with distillation and 0/5 without. CLM's own post-training keeps
  5/5. The CLM README says its mix was about 1M agent trajectories from ADP plus terminal traces from
  Endless-Terminals and LiteCoder-Terminal-SFT. Ours was a small ADP slice with no terminal traces.
  Find the data, or the recipe, that brings the skill back from the mid-trained head. Start in
  `pipeline/build_agent_data.py` and the agent rows of `results/innovations/day2_2026-11-22.md`.
- **The shared head at full scale.** One shared head for states and actions learned fastest in the CPU
  sweep with half the parameters: 64.8% general against 56.5% after 3,000 steps
  (`results/innovations/cpu_sweep/`, `innovations/RESULTS.md` section 3). It may only be faster early
  on. The check is `innovations/modal_train.py --tower shared_adapter` on the published vectors. Our
  guess is about $10 of GPU time, which is a guess and not a measurement. It needs the published vectors staged
  on a Modal volume first (`docs/BASELINE.md`). A shared head cannot be exported to CLM's checkpoint
  layout, so the official engine cannot serve it as is.
- **Matryoshka at full scale.** In the same sweep a Matryoshka head at 64 dims was within 0.3 points
  of its score at 1024 (`results/innovations/cpu_sweep/mrl_p1024.json`). If that holds after full training, the action
  cache can be 8 times smaller. The full-scale command is in the docstring of `innovations/modal_train.py`.
- **A full-scale retrain on a newer backbone.** Muse Glimmer 30B led the backbone screen (36.9% tricky
  against 34.2% for Qwen3-8B), but that was a 96k-pair slice where every head overfits
  (`results/innovations/bb_*.json`). A real answer needs the full Q&A text embedded with the new
  backbone and a full pre-training and mid-training run. Its latency per decision is not measured
  either: only the Qwen variants were timed.
- **Close AWQ's 4.5-point gap.** The AWQ 4-bit backbone has 36 to 37% lower latency than bf16 but costs 4.5
  points of tricky Q&A with CLM's unchanged head, against 0.7 for FP8 (`results/innovations/auth_awq.json`,
  `auth_fp8.json`, `backbone_latency_l40s.json`). Two things to try: an AWQ calibrated on the pooling
  task, with states and options as calibration text and the last-token vector as the target. Or a few
  hundred head training steps on AWQ vectors.
- **More held-out decision benchmarks.** The held-out test has 13,194 rows that we built from eight
  public sets, seven of them sources we never trained on (`results/innovations/heldout.json`). It has
  single tool calls only, short routing messages, English only and one option wording, which is ours.
  A test with a different option wording would show how much of the gain is the template. Wanted: other tool-calling sets, multi-step agent decisions, routing in other
  languages, and any agent benchmark besides T-Rex and DeepSWE that can be run from public parts.
  `innovations/build_heldout_choice.py` shows how to add a source and how to drop rows that overlap the
  training data.

### Runs on a CPU

- **Can int8 activations be made to work?** Dynamic int8 quantization of weights and activations drops
  tricky Q&A from 68.8% to 49.1%, while int8 weights alone cost nothing
  (`results/innovations/quant_heads_mid.json`). The suspected cause is large outlier dimensions in
  Qwen's hidden states. Try static quantization with calibration data, or clipping the outliers first.
  Start in `innovations/quantize_heads.py`.
- **Are the probabilities calibrated?** Only top-1 accuracy was measured. Measure calibration for
  CLM-8B, S1-Tweaks v0.1 and S1-Router v0.1 on the demo sample or the choice test vectors with
  `innovations/scorer.py`.
- **Does the head layout ranking hold with more steps?** The layout sweep stopped at 3,000 steps on
  1.2M pairs, far from convergence. Rerun `innovations/sweep_cpu.sh` with a larger `STEPS` and see
  whether the shared head keeps its lead.

### Needs one GPU for a short time

- **Finish testing S1-Router v0.1.** It has no DeepSWE run and no 15-seed T-Rex run
  (`MODEL_CARD.md`).
- **Quantized backbones with the released heads.** FP8 and AWQ were scored only with CLM's mid-trained
  head. Nobody has scored S1-Tweaks v0.1 or S1-Router v0.1 on FP8 or AWQ vectors.
- **More seeds for the blends between 0.15 and 0.25.** Survival is not monotonic in the blend weight:
  0.175 survives 3/5 courses, 0.2 survives 4/5 and 0.25 survives 4/5, with 5 seeds each
  (`results/innovations/day3_2026-11-23.md`). Only CLM-8B and the 0.15 blend have a second run, over
  seeds 0 to 14. One game is inside the noise.
- **Inkling 2 Mini on an H100.** It currently needs a B200 and a patched vLLM image
  (`results/innovations/inkling_attempts.md`). A vLLM or transformers release that loads it would also
  make the loader patch in `innovations/modal_embed.py` unnecessary.
- **More backbones.** Only three were compared. `qwen3-4b` is wired into `innovations/modal_embed.py`
  and was never run.

## Pull requests

Fill in the checklist in the pull request template. If your change adds a result, a reviewer should be
able to rerun it from `docs/REPRODUCE.md` alone.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security
problems go through the private route described in [`.github/SECURITY.md`](.github/SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, the licence
of this repository. See `LICENSE` and `NOTICE`.
