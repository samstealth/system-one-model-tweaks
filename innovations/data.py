"""Read the published Qwen3-8B vector chunks from a local folder (or a Modal volume mounted at /data).

Layout, the same one pipeline/stage_data.py writes:
  <root>/qa/<shard>/chunk_NNNN_{q,a}.npy        float16 [100000, 4096]
  <root>/qa/<shard>/chunk_NNNN_meta.parquet     doc_id, pair_idx
  <root>/neg/<shard>/chunk_NNNN_{n,ncnt}.npy    float16 [100000, 10, 4096], uint8 [100000]

Arrays are memory-mapped, so a 16 GB laptop can work through chunks far larger than its RAM.
"""
import glob
import os
import zlib

import numpy as np

HOLD_GEN = [("shard_05", 83), ("shard_05", 84)]
HOLD_HN = [("shard_00", 28), ("shard_00", 29)]


def stem(root, c, kind="qa"):
    return os.path.join(root, kind, c[0], f"chunk_{c[1]:04d}")


def available(root, kind="qa", suffix="_q.npy"):
    out = []
    for p in sorted(glob.glob(os.path.join(root, kind, "shard_*", f"chunk_*{suffix}"))):
        out.append((p.split(os.sep)[-2], int(os.path.basename(p).split("_")[1])))
    return out


def doc_ids(root, c):
    import pyarrow.parquet as pq
    ids = pq.read_table(stem(root, c) + "_meta.parquet", columns=["doc_id"]).column("doc_id").to_pylist()
    return np.fromiter((zlib.crc32(x.encode()) for x in ids), dtype=np.int64, count=len(ids))


def load_chunk(root, c, negs=False, limit=None):
    d = {"q": np.load(stem(root, c) + "_q.npy", mmap_mode="r"),
         "a": np.load(stem(root, c) + "_a.npy", mmap_mode="r"),
         "doc": doc_ids(root, c)}
    if negs:
        d["n"] = np.load(stem(root, c, "neg") + "_n.npy", mmap_mode="r")
        d["c"] = np.load(stem(root, c, "neg") + "_ncnt.npy")
    if limit:
        d = {k: v[:limit] for k, v in d.items()}
    return d


def training_chunks(root):
    held = set(HOLD_GEN) | set(HOLD_HN)
    return [c for c in available(root) if c not in held]


def batches(chunks, root, batch, seed=0, negs=False, limit=None):
    """Endless shuffled batches over the given chunks, one chunk in memory at a time."""
    rng = np.random.default_rng(seed)
    while True:
        for i in rng.permutation(len(chunks)):
            d = load_chunk(root, chunks[i], negs, limit)
            n = len(d["q"])
            idx = rng.permutation(n)
            for j in range(0, n - batch + 1, batch):
                b = np.sort(idx[j:j + batch])
                yield {k: np.asarray(v[b]) for k, v in d.items()}
