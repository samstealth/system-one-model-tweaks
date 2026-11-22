# Inkling 2 Mini embedding attempts (22 Nov 2026)

Goal: `modal_embed.py --job qa --backbone inkling-2-mini` on one H100, to compare a head trained on
Inkling vectors with the same-budget Qwen3-8B head (`bb_qwen3-8b`). It did not run on any stack available
on 22 Nov 2026. No Inkling vectors were produced.

| # | Stack | Result |
|---|---|---|
| 1 | `CLM_NEW_STACK=1`: vllm 0.34.0, transformers 5.22.2 (latest releases) | `ValueError`: model type `inkling2_loop` "but Transformers does not recognize this architecture"; vLLM's `ModelConfig` validation fails before load. |
| 2 | `CLM_IMAGE=vllm/vllm-openai:inkling-2-mini` (the image the vLLM recipe names until vLLM >= 0.35 ships) | Image build failed: the image has `python3` only, so Modal's pip step got `python: not found`. Fixed with a symlink. |
| 3 | Same image, fixed: vllm `0.1.dev25855+g7cb5dae22`, transformers 5.23.0, Python 3.14 | vLLM recognizes the model as `Inkling2LoopForEmbedding` (`runner="pooling"`), then weight loading fails: `ValueError: Unexpected checkpoint tensor: lm_head.weight`. The embedding wrapper does not skip the LM head. |

Not attempted: `--engine hf`. In the same image, `AutoConfig.from_pretrained("thinkingmachines/Inkling-2-Mini")`
also fails with "Transformers does not recognize this architecture", so the HF path cannot load it either.

## What would unblock it
- A vLLM release (>= 0.35) whose Inkling pooling loader skips `lm_head.weight`, or a local patch that drops
  `lm_head.*` in `Inkling2LoopForEmbedding.load_weights`.
- Or a transformers release with `inkling2_loop`, then `--engine hf`
  (load with `AutoModelForMultimodalLM`, take the last hidden state).
- Or the generation runner with hidden-state output, instead of the pooling runner.

## Code kept
`innovations/modal_embed.py` gains an optional `CLM_IMAGE` override: run any backbone in a prebuilt registry
image, with a python symlink and unpinned pyarrow so images on Python 3.14 build. Attempt 3 confirmed it builds
and reaches model load.

Cost: about 15 H100-minutes across attempts 1 and 3, including vLLM's automatic engine restarts.
