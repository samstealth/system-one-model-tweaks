"""Measure what it costs to embed CLM's Q&A corpus with Muse Glimmer 30B, before paying for it.

1. sample (CPU): ~100k Q&A pairs drawn across all six shards of the public token dataset
   (jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens), documents picked by a stable hash
   (crc32(doc_id) % MOD == 0), text decoded with the Qwen3 tokenizer at CLM's full length
   (2,047 context tokens), plus a 10k sample of hard negatives. Saved to /data/muse_calib/.
2. embed (one H100): Muse Glimmer 30B in vLLM's pooling runner, the same settings as
   modal_embed.py. Model load and embedding are timed separately; states keep their last 2,047
   tokens, answers and negatives their first 2,047, as in the corpus recipe.

    modal run innovations/muse_calibration.py --step sample
    modal run innovations/muse_calibration.py --step embed
"""
import json
import os
import time
import zlib

import modal

REPO = "jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens"
PARTS = [f"part_{i:06d}" for i in range(6)]
MOD = 650          # ~7.8M docs / 6 parts / 650 = ~2,000 docs (~16k pairs) per part
CAP = 2047
OUT = "/data/muse_calib"
MUSE = "meta-models/Muse-Glimmer-30B"

vol = modal.Volume.from_name("clm-data", version=2)
app = modal.App("clm-muse-calibration")
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("pyarrow==18.1.0", "huggingface_hub", "hf_transfer", "transformers<5", "numpy")
             .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("vllm", "transformers", "accelerate", "numpy", "huggingface_hub", "hf_transfer")
             .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=cpu_image, volumes={"/data": vol}, cpu=8, memory=32768, timeout=3600)
def sample():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)
    os.makedirs(OUT, exist_ok=True)
    pairs, negs, stats = [], [], {}
    for part in PARTS:
        f = {k: hf_hub_download(REPO, f"{k}/{part}.parquet", repo_type="dataset") for k in ("qa_pairs", "docs", "neg_pairs")}
        qa = pq.read_table(f["qa_pairs"], columns=["doc_id", "pair_idx", "question_ids", "answer_ids"]).to_pydict()
        keep = [i for i, d in enumerate(qa["doc_id"]) if zlib.crc32(d.encode()) % MOD == 0]
        docs = {qa["doc_id"][i] for i in keep}
        ctx = {}
        dp = pq.ParquetFile(f["docs"])
        for rg in range(dp.num_row_groups):
            t = dp.read_row_group(rg, columns=["doc_id", "context_ids"]).to_pydict()
            for d, ids in zip(t["doc_id"], t["context_ids"]):
                if d in docs:
                    ctx[d] = dec(ids[:CAP])
        for i in keep:
            d = qa["doc_id"][i]
            if d in ctx:
                pairs.append({"doc_id": d, "pair_idx": qa["pair_idx"][i], "state": ctx[d] + "\n\n" + dec(qa["question_ids"][i]),
                              "answer": dec(qa["answer_ids"][i])})
        if len(negs) < 10000:
            nt = pq.ParquetFile(f["neg_pairs"]).read_row_group(0, columns=["doc_id", "neg_ids"]).to_pydict()
            negs += [dec(ids) for d, ids in zip(nt["doc_id"], nt["neg_ids"]) if zlib.crc32(d.encode()) % 7 == 0][: 10000 - len(negs)]
        stats[part] = {"pairs": sum(1 for p in pairs if True), "docs_picked": len(docs), "docs_found": len(ctx)}
        print(part, stats[part], flush=True)
    with open(f"{OUT}/pairs.jsonl", "w") as w:
        for p in pairs:
            w.write(json.dumps(p) + "\n")
    json.dump(negs, open(f"{OUT}/negatives.json", "w"))
    vol.commit()
    return {"pairs": len(pairs), "negatives": len(negs), "per_part_cumulative": stats}


@app.function(image=gpu_image, volumes={"/data": vol}, gpu="H100", cpu=8, memory=64000, timeout=90 * 60)
def embed():
    import numpy as np
    vol.reload()
    t0 = time.perf_counter()
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    import vllm, transformers
    tok = AutoTokenizer.from_pretrained(MUSE)
    llm = LLM(model=MUSE, runner="pooling", max_model_len=CAP + 1, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    t_load = time.perf_counter() - t0
    pairs = [json.loads(l) for l in open(f"{OUT}/pairs.jsonl")]
    negs = json.load(open(f"{OUT}/negatives.json"))
    ids = lambda texts, keep: [(lambda x: x[-CAP:] if keep == "tail" else x[:CAP])(tok(t, add_special_tokens=False)["input_ids"]) or [0] for t in texts]
    t = time.perf_counter(); s_ids = ids([p["state"] for p in pairs], "tail"); a_ids = ids([p["answer"] for p in pairs], "head"); n_ids = ids(negs, "head")
    t_tok = time.perf_counter() - t

    def run(seqs):
        t = time.perf_counter()
        out = llm.embed([TokensPrompt(prompt_token_ids=x) for x in seqs], use_tqdm=False)
        dt = time.perf_counter() - t
        assert len(out) == len(seqs) and len(out[0].outputs.embedding) > 0
        return dt, len(out[0].outputs.embedding)

    t_s, dim = run(s_ids)
    t_a, _ = run(a_ids)
    t_n, _ = run(n_ids)
    L = lambda seqs: (lambda a: {"mean": round(float(a.mean()), 1), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                                 "max": int(a.max()), "total": int(a.sum())})(np.array([len(x) for x in seqs]))
    return {"model": MUSE, "gpu": "H100", "vllm": vllm.__version__, "transformers": transformers.__version__, "dim": dim,
            "pairs": len(pairs), "negatives": len(negs),
            "seconds": {"load_model": round(t_load, 1), "tokenize": round(t_tok, 1), "embed_states": round(t_s, 1),
                        "embed_answers": round(t_a, 1), "embed_negatives": round(t_n, 1)},
            "tokens": {"states": L(s_ids), "answers": L(a_ids), "negatives": L(n_ids)}}


@app.local_entrypoint()
def main(step: str = "sample"):
    r = sample.remote() if step == "sample" else embed.remote()
    print("RESULT " + json.dumps(r))
