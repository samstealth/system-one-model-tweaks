"""Write a SYNTHETIC sample in the format examples/demo_cpu.py reads. For tests and plumbing checks only.

The vectors are random numbers, not Qwen3-8B output, so accuracy on this file means nothing.
Every row is marked "synthetic": true, every source name starts with "synthetic_", and the script
refuses to write into examples/data, where the real sample lives. Do not commit its output.

    python examples/make_synthetic_sample.py                 # writes to a fresh temp folder and prints it
    python examples/make_synthetic_sample.py --out /tmp/s1   # or to a folder you name
    python examples/demo_cpu.py --data /tmp/s1

Format (the same contract as the real files):
  sample_decisions.npz    q (N, D) float16, opts (N, K, D) float16 with the gold option at its
                          original index and zero rows as padding, n_opts (N,) uint8, gold (N,) int64
  sample_decisions.jsonl  one row per decision, same order: {"source", "state", "options": [...], "gold"}
"""
import argparse
import json
import os
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(HERE, "data")
NAME = "sample_decisions"
SOURCES = ("synthetic_tools", "synthetic_routing")


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def write_sample(out_dir=None, n=48, k=6, dim=4096, seed=0, noise=0.5):
    """Write the two files into out_dir (a new temp folder if None). Returns (npz_path, jsonl_path).

    Each state vector is its gold option's vector plus noise, so a head whose two sides agree can find
    the gold option. That makes the format testable; it says nothing about any real head.
    """
    out_dir = out_dir or tempfile.mkdtemp(prefix="s1_synthetic_")
    if os.path.realpath(out_dir) == os.path.realpath(REAL_DATA_DIR):
        raise SystemExit("Refusing to write synthetic data into examples/data. That folder is for the real sample. "
                         "Pick another --out.")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_opts = rng.integers(2, k + 1, n).astype(np.uint8)
    gold = np.array([rng.integers(0, m) for m in n_opts], dtype=np.int64)
    opts = np.zeros((n, k, dim), np.float32)
    q = np.zeros((n, dim), np.float32)
    rows = []
    for i in range(n):
        m = int(n_opts[i])
        opts[i, :m] = _unit(rng.normal(size=(m, dim)))
        q[i] = _unit(opts[i, gold[i]] + noise * _unit(rng.normal(size=dim)))
        rows.append({"source": SOURCES[i % len(SOURCES)],
                     "state": f"Synthetic state {i}: random numbers, not a real decision.\n\nWhich option is correct?",
                     "options": [f"option_{j}: synthetic option {j} of decision {i}" for j in range(m)],
                     "gold": int(gold[i]), "synthetic": True})
    npz, jsonl = os.path.join(out_dir, NAME + ".npz"), os.path.join(out_dir, NAME + ".jsonl")
    np.savez(npz, q=q.astype(np.float16), opts=opts.astype(np.float16), n_opts=n_opts, gold=gold)
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return npz, jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description="Write a synthetic sample for examples/demo_cpu.py (random vectors, tests only).")
    p.add_argument("--out", default=None, help="folder to write into (default: a new temp folder)")
    p.add_argument("--n", type=int, default=48, help="number of decisions")
    p.add_argument("--k", type=int, default=6, help="most options per decision")
    p.add_argument("--dim", type=int, default=4096, help="vector size (4096 matches Qwen3-8B heads)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    npz, jsonl = write_sample(a.out, a.n, a.k, a.dim, a.seed)
    print("wrote", npz)
    print("wrote", jsonl)
    print("SYNTHETIC data: random vectors. Try it with: python examples/demo_cpu.py --data", os.path.dirname(npz))
    return npz, jsonl


if __name__ == "__main__":
    main()
