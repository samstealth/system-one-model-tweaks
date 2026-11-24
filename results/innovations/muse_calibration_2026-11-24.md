# Muse Glimmer 30B: embedding cost calibration (24 Nov 2026)

Question: what would it cost to embed CLM's full Q&A corpus with Muse Glimmer 30B instead of using the
published Qwen3-8B vectors? Measured on a real sample, not extrapolated from tokens.

- Sample: 95,857 Q&A pairs from 11,984 documents, picked by a stable hash across all 6 shards of the public
  token dataset, at CLM's full state length (up to 2,047 tokens), plus 10,000 hard negatives.
- Setup: one H100, vLLM 0.34 pooling runner (last token, normalized), the same settings as `modal_embed.py`.
- Lengths with Muse's tokenizer: states mean 735, median 704, p95 1,301, max 2,047 tokens; answers mean 9.6;
  negatives mean 6.7.
- Speed, embedding only (model load 126 s and tokenization 104 s excluded): **78.2 pairs/s**
  (states 63.3k tokens/s), negatives 1,053/s.
- Cost: the bill went from $68.09 to $68.56 for sampling (CPU) and to $70.97 for the H100 step, so
  **$2.88 for the calibration**. That is $5.92 per H100-container hour, all in.

| Scope | H100 hours | Cost |
|---|---|---|
| Per million pairs | 3.55 | $21 |
| 5M-pair subset | 17.8 | $105 |
| 10M-pair subset | 35.5 | $210 |
| All 62.45M pairs + 26.7M hard negatives | 229 | **$1,356** |

Arithmetic: 1,000,000 / 78.2 pairs/s = 12,788 s = 3.55 h, and 3.55 h x $5.92/h = $21.0 per million pairs.
The negatives take 26.7M / 1,053 per s = 7.0 h, or $42.

Storage at fp16, 6,656 numbers per vector: 1.66 TB for pairs and 0.36 TB for negatives, **2.0 TB** in total
(Qwen3-8B's 4,096-d vectors: 1.24 TB, which matches the published set).

Not included: training the heads (about $10-25), container starts, CPU tokenization at scale (it can run
alongside the GPU), and any re-runs. Script: `innovations/muse_calibration.py`; numbers:
`muse_calibration.json`.
