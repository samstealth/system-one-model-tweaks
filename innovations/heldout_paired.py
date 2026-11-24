"""Paired significance test on the held-out choice set (sources no head was trained on).

For every row of /data/emb/qwen3-8b-heldout (built by build_heldout_choice.py and embedded by
modal_embed.py --job choice), each head is scored with the same rule as evals.tricky and the
--choice-eval numbers: a row is right when cosine(state, gold) is strictly greater than every other
option's cosine. Per source, and pooled, for each (new head, baseline) pair this writes:

  n, correct counts for both heads, both_right / only_baseline_right / only_new_right / neither,
  the exact two-sided McNemar p-value (binomial test on the discordant pairs), and a paired bootstrap
  95% interval for the accuracy difference in points (10,000 resamples of rows, numpy seed 0;
  pooled entries resample rows from the pooled set).

It also records the number of options per source (min / median / max) so chance levels have a file.

    modal run innovations/heldout_paired.py      # writes results/innovations/heldout_paired.json locally
"""
import json
import math
import os

import modal

ROOT = "/data/emb/qwen3-8b-heldout"
HEADS = {"authors_post": "/data/heads/qwen3_8b_posttrained_head.pt",
         "wise_0.15": "/data/runs/wise_0.15/final_authors_layout.pt",
         "post_choice_hn_frompost": "/data/runs/post_choice_hn_frompost/final_authors_layout.pt"}
PAIRS = [("wise_0.15", "authors_post"), ("post_choice_hn_frompost", "authors_post")]
POOLS = {"pooled_6": ["bfcl", "massive", "hwu64", "bitext", "sciq", "qasc"],
         "pooled_8": ["bfcl", "massive", "hwu64", "bitext", "sciq", "qasc", "mmlu", "medmcqa"]}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "innovations", "heldout_paired.json")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0")
         .add_local_python_source("innovations"))
vol = modal.Volume.from_name("clm-data", version=2)
app = modal.App("clm-heldout-paired")


def mcnemar_exact(b, c):
    """Exact two-sided McNemar: binomial(b + c, 0.5) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def compare(new, base, rng, reps=10000):
    import numpy as np
    new, base = np.asarray(new, bool), np.asarray(base, bool)
    n = len(new)
    d = new.astype(np.float64) - base.astype(np.float64)
    idx = rng.integers(0, n, size=(reps, n))
    boots = d[idx].mean(1) * 100
    b = int((base & ~new).sum())
    c = int((new & ~base).sum())
    return {"n": n, "baseline_correct": int(base.sum()), "new_correct": int(new.sum()),
            "baseline_acc": round(100 * base.mean(), 2), "new_acc": round(100 * new.mean(), 2),
            "diff_points": round(100 * d.mean(), 2),
            "both_right": int((new & base).sum()), "only_baseline_right": b, "only_new_right": c,
            "neither": int((~new & ~base).sum()), "mcnemar_p_exact": mcnemar_exact(b, c),
            "bootstrap95_points": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4, memory=32768)
def run():
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from innovations.heads import TwoTower
    base = f"{ROOT}/qa/shard_11/chunk_0000"
    q = torch.from_numpy(np.load(base + "_q.npy").astype(np.float32))
    a = torch.from_numpy(np.load(base + "_a.npy").astype(np.float32))
    n = np.load(f"{ROOT}/neg/shard_11/chunk_0000_n.npy", mmap_mode="r")
    cnt = torch.from_numpy(np.load(f"{ROOT}/neg/shard_11/chunk_0000_ncnt.npy").astype(np.int64))
    src = [s.replace("heldout_", "") for s in pq.read_table(base + "_meta.parquet", columns=["source"]).column("source").to_pylist()]
    src = np.array(src)
    K = n.shape[1]
    right = {}
    with torch.no_grad():
        for h, path in HEADS.items():
            m = TwoTower.from_authors(path).eval()
            hits = []
            for i in range(0, len(q), 1024):
                j = min(i + 1024, len(q))
                zq, za = m.encode_state(q[i:j]), m.encode_action(a[i:j])
                zn = m.encode_action(torch.from_numpy(np.asarray(n[i:j], np.float32)).view(-1, n.shape[2])).view(j - i, K, -1)
                c = cnt[i:j]
                neg = torch.einsum("bh,bkh->bk", zq, zn).masked_fill(torch.arange(K)[None] >= c[:, None], -9.0)
                hits.append(((zq * za).sum(-1) > neg.max(1).values).numpy())
            right[h] = np.concatenate(hits)
    valid = cnt.numpy() > 0  # rows with at least one negative, as in evals.tricky
    opts = cnt.numpy() + 1
    rng = np.random.default_rng(0)
    out = {"note": __doc__.split("\n\n")[1].replace("\n", " "), "heads": HEADS, "rows_scored": int(valid.sum()),
           "options_per_source": {}, "sources": {}, "pooled": {}}
    groups = {s: [s] for s in sorted(set(src))}
    for s in groups:
        o = opts[(src == s) & valid]
        out["options_per_source"][s] = {"min": int(o.min()), "median": float(np.median(o)), "max": int(o.max()),
                                        "chance_pct": round(100 * float(np.mean(1.0 / o)), 2)}
    for name, members in list(groups.items()) + list(POOLS.items()):
        mask = np.isin(src, members) & valid
        entry = {f"{new}_vs_{bl}": compare(right[new][mask], right[bl][mask], rng) for new, bl in PAIRS}
        (out["pooled"] if name in POOLS else out["sources"])[name] = entry
    return out


@app.local_entrypoint()
def main():
    out = run.remote()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for name, e in list(out["sources"].items()) + list(out["pooled"].items()):
        for k, v in e.items():
            print(f"{name:10s} {k:40s} n={v['n']:6d} {v['baseline_acc']:6.2f} -> {v['new_acc']:6.2f} "
                  f"diff {v['diff_points']:+6.2f} CI {v['bootstrap95_points']} p={v['mcnemar_p_exact']:.2e}")
    print("wrote", OUT)
