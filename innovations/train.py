"""Train one TwoTower config on local vector chunks and score it on the held-out tests.

Runs on CPU for small comparisons and on a GPU for full runs, with the same code.

    python -m innovations.train --root /data --proj 256 --tower shared --steps 2000 --batch 1024 --out runs/p256_shared
    python -m innovations.train --root /data --matryoshka 64,128,256,512,1024 --proj 1024 --out runs/mrl

Defaults copy the baseline's pre-training recipe, except batch size and learning rate,
which are scaled down (sqrt rule, as in pipeline/train_heads.py) for small machines.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from innovations import data, evals
from innovations.heads import HeadConfig, TwoTower, model_loss


def build_eval(root, gen_limit, hn_limit):
    g = [data.load_chunk(root, c, limit=gen_limit) for c in data.HOLD_GEN if c in data.available(root)]
    h = [data.load_chunk(root, c, negs=True, limit=hn_limit) for c in data.HOLD_HN
         if c in data.available(root) and c in data.available(root, "neg", "_ncnt.npy")]
    return g, h


def score(model, g, h, dev, dims=(None,)):
    out = {}
    for d in dims:
        tag = f"@{d}" if d else ""
        for x in g:
            for k, v in evals.general(model, x["q"], x["a"], x["doc"], d, dev).items():
                out[k + tag] = v
        for x in h:
            for k, v in evals.tricky(model, x["q"], x["a"], x["n"], x["c"], d, dev).items():
                out[k + tag] = v
    return out


def chunks_arg(s):
    return [(c.split("/")[0], int(c.split("/")[1])) for c in s.split(",") if c]


def mixed(main_it, replay_it, frac, batch):
    """Concatenate a (1-frac) slice of the main batch with a frac slice of replay. Rows that carry
    hard negatives go first, which is the layout info_nce expects (negatives for the first k rows)."""
    n_r = int(batch * frac)
    for b, r in zip(main_it, replay_it):
        parts = [{k: v[: batch - n_r] for k, v in b.items()}, {k: v[:n_r] for k, v in r.items()}]
        parts.sort(key=lambda x: "n" not in x)
        out = {k: np.concatenate([x[k] for x in parts]) for k in ("q", "a", "doc")}
        with_n = [x for x in parts if "n" in x]
        if with_n:
            out["n"] = np.concatenate([x["n"] for x in with_n])
            out["c"] = np.concatenate([x["c"] for x in with_n])
        yield out


def choice_scores(model, root, dev):
    """Listwise top-1 per source on the held-out decision sets (banking77, clinc, ARC, tools, ...).
    Chunks are memory-mapped and read lazily at each eval, so nothing large stays resident."""
    if not root:
        return {}
    import pyarrow.parquet as pq
    out = {}
    for c in [c for c in data.available(root) if c[0] == data.CHOICE_TEST]:
        src = np.array(pq.read_table(data.stem(root, c) + "_meta.parquet", columns=["source"]).column("source").to_pylist())
        d = data.load_chunk(root, c, negs=True)
        for s in sorted(set(src)):
            i = np.where(src == s)[0]
            r = evals.tricky(model, d["q"][i], d["a"][i], d["n"][i], d["c"][i], dev=dev)
            out[f"choice_{s}"] = r["tricky_top1"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--proj", type=int, default=512)
    p.add_argument("--tower", default="separate", choices=["separate", "shared", "shared_adapter"])
    p.add_argument("--adapter-rank", type=int, default=64)
    p.add_argument("--matryoshka", default="", help="comma list of prefix dims, e.g. 64,128,256,512")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--hard-negs", action="store_true", help="mid-training style: add each pair's hard negatives")
    p.add_argument("--chunks", default="", help="comma list like shard_05/0,shard_05/1 (default: all non-held-out)")
    p.add_argument("--init", default="", help="TwoTower .pt or the authors' head .pt (separate tower only)")
    p.add_argument("--gen-limit", type=int, default=20480)
    p.add_argument("--hn-limit", type=int, default=8192)
    p.add_argument("--replay-root", default="", help="second vector root mixed into every batch (post-training replay)")
    p.add_argument("--replay-chunks", default="", help="chunks of --replay-root to use (default: all non-held-out)")
    p.add_argument("--replay-frac", type=float, default=0.0, help="share of each batch drawn from --replay-root")
    p.add_argument("--replay-negs", action="store_true", help="replay rows keep their hard negatives (needs neg/ chunks)")
    p.add_argument("--choice-eval", default="", help="vector root with listwise test chunks (qa/shard_11) to score per source")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    mrl = tuple(int(x) for x in a.matryoshka.split(",") if x)
    chunks = chunks_arg(a.chunks) or data.training_chunks(a.root)
    if not chunks:
        p.error(f"no training chunks under {a.root}/qa (choice shards need --chunks shard_10/N)")
    if any(c[0] == data.CHOICE_TEST for c in chunks):
        p.error("shard_11 is the choice test split; do not train on it")
    first = data.load_chunk(a.root, chunks[0])
    in_dim = first["q"].shape[1]
    cfg = HeadConfig(in_dim=in_dim, width=a.width, depth=a.depth, proj=a.proj, tower=a.tower, adapter_rank=a.adapter_rank, matryoshka=mrl)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model = TwoTower.load(a.init) if "cfg" in ck and "model" in ck else TwoTower.from_authors(a.init)
    else:
        model = TwoTower(cfg)
    model.to(dev)
    lr = a.lr or 2e-3 * math.sqrt(1024 / a.width) * math.sqrt(a.batch / 1024)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    g, h = build_eval(a.root, a.gen_limit, a.hn_limit)
    dims = mrl or (None,)
    meta = {"cfg": model.cfg.to_dict(), "params": model.n_params(), "lr": lr, "batch": a.batch, "chunks": len(chunks),
            "device": dev, "hard_negs": a.hard_negs}
    print(json.dumps(meta), flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    T = lambda x: torch.from_numpy(x).to(dev).float()
    t0 = time.time()
    it = data.batches(chunks, a.root, a.batch, a.seed, negs=a.hard_negs)
    if a.replay_frac > 0:
        rch = chunks_arg(a.replay_chunks) or data.training_chunks(a.replay_root)
        if not a.replay_root or not rch:
            p.error("--replay-frac > 0 needs --replay-root (and optionally --replay-chunks) with Q&A chunks")
        it = mixed(it, data.batches(rch, a.replay_root, a.batch, a.seed + 1, negs=a.replay_negs), a.replay_frac, a.batch)
    for step in range(1, a.steps + 1):
        b = next(it)
        n = T(b["n"]) if "n" in b else None
        c = torch.from_numpy(b["c"].astype(np.int64)).to(dev) if "n" in b else None
        loss = model_loss(model, T(b["q"]), T(b["a"]), torch.from_numpy(b["doc"]).to(dev), n, c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} scale {model.scale().item():.1f} {(time.time()-t0)/60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = {"step": step, "loss": round(loss.item(), 4), "minutes": round((time.time() - t0) / 60, 2),
                 **score(model, g, h, dev, dims), **choice_scores(model, a.choice_eval, dev)}
            print(json.dumps(m), flush=True)
            log.write(json.dumps(m) + "\n"); log.flush()
    model.save(os.path.join(a.out, "final.pt"), meta=meta)
    if model.cfg.tower == "separate":
        model.export_authors(os.path.join(a.out, "final_authors_layout.pt"))
    m["head_latency_ms_cpu_1000_actions"] = evals.head_latency(model.cpu(), in_dim=model.cfg.in_dim)
    result = {**meta, **m}
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
