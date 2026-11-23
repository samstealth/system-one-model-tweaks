"""Export a small CPU-only demo sample: real decisions with their frozen Qwen3-8B vectors.

Visitors can score a head on these without a GPU or the 8B encoder. 12 rows per source from the
held-out test splits (never trained on): glaive_fc, banking77, clinc_oos (in-distribution tests) and
heldout_bfcl, heldout_massive (sources no head here was trained on).

Writes examples/data/sample_decisions.npz with
  q       (N, 4096) float16   state vectors
  opts    (N, K, 4096) float16 option vectors, gold at its original index, zero-padded past n_opts
  n_opts  (N,) uint8
  gold    (N,) int64
and examples/data/sample_decisions.jsonl with source, state, options, gold per row in the same order.

    modal run innovations/export_demo_sample.py
"""
import json
import os

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.1.3", "pyarrow==18.1.0")
vol = modal.Volume.from_name("clm-data", version=2)
app = modal.App("clm-demo-sample")
SETS = [("/data/emb/qwen3-8b", "/data/choice/choice_test.jsonl", ["glaive_fc", "banking77", "clinc_oos"]),
        ("/data/emb/qwen3-8b-heldout", "/data/heldout/choice_test.jsonl", ["heldout_bfcl", "heldout_massive"])]
PER, K = 12, 10


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, memory=32768)
def export(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    qs, os_, nn, gg, rows = [], [], [], [], []
    for root, text, sources in SETS:
        base = f"{root}/qa/shard_11/chunk_0000"
        q = np.load(base + "_q.npy", mmap_mode="r")
        a = np.load(base + "_a.npy", mmap_mode="r")
        n = np.load(f"{root}/neg/shard_11/chunk_0000_n.npy", mmap_mode="r")
        c = np.load(f"{root}/neg/shard_11/chunk_0000_ncnt.npy")
        lines = [json.loads(l) for l in open(text, encoding="utf-8") if l.strip()]
        assert len(lines) == len(q), (text, len(lines), len(q))
        for s in sources:
            idx = [i for i, r in enumerate(lines) if r["source"] == s]
            for i in sorted(rng.choice(idx, PER, replace=False)):
                r, g, k = lines[i], lines[i]["gold"], int(c[i])
                negs = np.asarray(n[i, :k])
                o = np.zeros((K, q.shape[1]), np.float16)
                full = np.concatenate([negs[:g], np.asarray(a[i])[None], negs[g:]])
                o[: len(full)] = full
                assert len(full) == len(r["options"]), (s, i, len(full), len(r["options"]))
                qs.append(np.asarray(q[i])); os_.append(o); nn.append(len(full)); gg.append(g)
                rows.append({"source": s, "state": r["state"], "options": r["options"], "gold": g})
    out = "/data/demo"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/sample_decisions.npz", q=np.stack(qs).astype(np.float16), opts=np.stack(os_),
                        n_opts=np.array(nn, np.uint8), gold=np.array(gg, np.int64))
    with open(f"{out}/sample_decisions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    vol.commit()
    return {"rows": len(rows), "bytes": os.path.getsize(f"{out}/sample_decisions.npz")}


@app.local_entrypoint()
def main():
    print(export.remote())
