# Inkling 2 Mini embedding attempts (22 Nov 2026)

Goal: `modal_embed.py --job qa --backbone inkling-2-mini`, to compare a head trained on Inkling vectors
with the same-budget Qwen3-8B head (`bb_qwen3-8b`).

**Outcome: it works on a B200 (attempt 5)**, using vLLM's Inkling image plus an lm_head-skip patch. It does
not run on H100 with that image, because Inkling's custom `conv_mix` CUDA kernel has no sm_90 build.

Working command:

    CLM_IMAGE=vllm/vllm-openai:inkling-2-mini CLM_GPU=B200       modal run --detach innovations/modal_embed.py --job qa --backbone inkling-2-mini

Engine start took about 3.5 min (211 s, mostly Triton/compile). The 96k train pairs took about 12.5 min,
then the 12k test pairs and their up-to-10 hard negatives followed.

| # | Stack | Result |
|---|---|---|
| 1 | `CLM_NEW_STACK=1`: vllm 0.34.0, transformers 5.22.2 (latest releases) | `ValueError`: model type `inkling2_loop` "but Transformers does not recognize this architecture"; vLLM's `ModelConfig` validation fails before load. |
| 2 | `CLM_IMAGE=vllm/vllm-openai:inkling-2-mini` (the image the vLLM recipe names until vLLM >= 0.35 ships) | Image build failed: the image has `python3` only, so Modal's pip step got `python: not found`. Fixed with a symlink. |
| 3 | Same image, fixed: vllm `0.1.dev25855+g7cb5dae22`, transformers 5.23.0, Python 3.14 | vLLM recognizes the model as `Inkling2LoopForEmbedding` (`runner="pooling"`), then weight loading fails: `ValueError: Unexpected checkpoint tensor: lm_head.weight`. The embedding wrapper does not skip the LM head. |

Not attempted: `--engine hf`. In the same image, `AutoConfig.from_pretrained("thinkingmachines/Inkling-2-Mini")`
also fails with "Transformers does not recognize this architecture", so the HF path cannot load it either.

| 4 | Same image, H100, with `_skip_lm_head_in_pooling_loader()` (drops `lm_head*` / `*unembed.weight` before `load_weights`; engine in-process) | Weights load. Then the first forward fails: `RuntimeError: launch, /workspace/csrc/inkling2/conv_mix.cu:978, no kernel image is available for execution on the device`. The kernel is not built for H100 (sm_90). The checkpoint's experts are also NVFP4, which is native on Blackwell. |
| 5 | Same as 4, on **B200** (sm_100) | **Works.** `Supported pooling task: embed`; train 95,996 and test 11,998 embedded into `/data/emb/inkling-2-mini`. |

Note on the patch: vLLM builds `Inkling2LoopForEmbedding` on the fly from the registered
`Inkling2LoopForConditionalGeneration`, so no `*Embed*` architecture is in `ModelRegistry`. The check that
raises lives in `Inkling2ForConditionalGeneration.load_weights` (`vllm/models/inkling2/nvidia/model.py`).
The helper therefore patches every class in each Inkling architecture's MRO that defines `load_weights`.
It filters on raw checkpoint names (before vLLM's renames), so it catches `model.llm.unembed.weight`.

## What would make it simpler
- A vLLM release (>= 0.35) whose Inkling pooling loader skips `lm_head.weight`, or a local patch that drops
  `lm_head.*` in `Inkling2LoopForEmbedding.load_weights`.
- Or a transformers release with `inkling2_loop`, then `--engine hf`
  (load with `AutoModelForMultimodalLM`, take the last hidden state).
- Or the generation runner with hidden-state output, instead of the pooling runner.

## Code kept
`innovations/modal_embed.py` gains an optional `CLM_IMAGE` override: run any backbone in a prebuilt registry
image, with a python symlink and unpinned pyarrow so images on Python 3.14 build. Attempt 3 confirmed it builds
and reaches model load.

Cost of the failed attempts: about 25 H100-minutes (attempts 1, 3 and 4, including vLLM's automatic engine restarts).
Attempt 5: about 20 B200-minutes.
