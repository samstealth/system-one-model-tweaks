"""Fast CPU checks of every code path on synthetic vectors. Run: python -m pytest -q tests"""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from innovations import train
from innovations.heads import HeadConfig, TwoTower, model_loss

D = 64


def make_root(tmp, shards, n=512, negs=(), source=None):
    rng = np.random.default_rng(0)
    for shard, idx in shards:
        d = tmp / "qa" / shard; d.mkdir(parents=True, exist_ok=True)
        base = rng.normal(size=(n, D)).astype(np.float32)
        q = base + 0.1 * rng.normal(size=(n, D)); a = base + 0.1 * rng.normal(size=(n, D))
        nz = lambda x: (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float16)
        np.save(d / f"chunk_{idx:04d}_q.npy", nz(q)); np.save(d / f"chunk_{idx:04d}_a.npy", nz(a))
        meta = {"doc_id": [f"{shard}-{idx}-{i // 2}" for i in range(n)]}
        if source:
            meta["source"] = [source[i % len(source)] for i in range(n)]
        pq.write_table(pa.table(meta), d / f"chunk_{idx:04d}_meta.parquet")
        if (shard, idx) in negs:
            nd = tmp / "neg" / shard; nd.mkdir(parents=True, exist_ok=True)
            np.save(nd / f"chunk_{idx:04d}_n.npy", nz(rng.normal(size=(n, 10, D))))
            np.save(nd / f"chunk_{idx:04d}_ncnt.npy", rng.integers(0, 11, n).astype(np.uint8))
    return str(tmp)


def test_towers_and_matryoshka():
    for tower in ("separate", "shared", "shared_adapter"):
        m = TwoTower(HeadConfig(in_dim=D, width=32, proj=16, tower=tower, matryoshka=(4, 8, 16)))
        q, a = torch.randn(8, D), torch.randn(8, D)
        loss = model_loss(m, q, a, torch.arange(8), torch.randn(4, 10, D), torch.tensor([10, 3, 0, 5]))
        loss.backward()
        assert torch.isfinite(loss)
        assert m.encode_state(q, 4).shape == (8, 4)
    sep = TwoTower(HeadConfig(in_dim=D, width=32, proj=16)).n_params()
    shared = TwoTower(HeadConfig(in_dim=D, width=32, proj=16, tower="shared")).n_params()
    assert shared * 2 == sep


def test_authors_layout_roundtrip(tmp_path):
    m = TwoTower(HeadConfig())
    ck = {"state_head": m.state_head.state_dict(), "action_head": m.action_head.state_dict(), "logit_scale": torch.tensor(3.0)}
    torch.save(ck, tmp_path / "h.pt")
    m2 = TwoTower.from_authors(str(tmp_path / "h.pt"))
    x = torch.randn(2, 4096)
    assert torch.allclose(m.encode_state(x), m2.encode_state(x))


def test_train_end_to_end(tmp_path):
    root = make_root(tmp_path / "qa", [("shard_05", 0), ("shard_05", 83), ("shard_00", 28)], negs={("shard_00", 28)})
    r = train.main(["--root", root, "--out", str(tmp_path / "run"), "--width", "32", "--proj", "16", "--steps", "30",
                    "--batch", "64", "--eval-every", "30", "--gen-limit", "256", "--hn-limit", "256", "--lr", "0.01",
                    "--matryoshka", "8,16", "--tower", "shared_adapter"])
    assert r["gen_top1@16"] > 0.2 and "tricky_top1@8" in r
    m = TwoTower.load(str(tmp_path / "run" / "final.pt"))
    assert m.cfg.in_dim == D


def test_choice_with_replay(tmp_path):
    croot = make_root(tmp_path / "ch", [("shard_10", 0), ("shard_11", 0)], negs={("shard_10", 0), ("shard_11", 0)},
                      source=["banking77", "arc"])
    qroot = make_root(tmp_path / "qa", [("shard_05", 0), ("shard_05", 83), ("shard_00", 28)], negs={("shard_00", 28)})
    r = train.main(["--root", croot, "--chunks", "shard_10/0", "--hard-negs", "--replay-root", qroot, "--replay-frac", "0.4",
                    "--choice-eval", croot, "--out", str(tmp_path / "run"), "--width", "32", "--proj", "16", "--steps", "20",
                    "--batch", "64", "--eval-every", "20", "--lr", "0.01"])
    assert "choice_banking77" in r and "choice_arc" in r


def test_choice_shards_never_default_training(tmp_path):
    from innovations import data
    root = make_root(tmp_path / "r", [("shard_00", 0), ("shard_10", 0), ("shard_11", 0), ("shard_00", 28)])
    assert data.training_chunks(root) == [("shard_00", 0)]


def test_labeled_split_is_question_disjoint(tmp_path):
    import random
    from innovations import build_choice_data as b
    mc = [b.mcq(f"Question {i}?", ["x", "y", "z"], i % 3, "arc", "train") for i in range(2000)]
    lab = b.labeled(mc, random.Random(0), 2000)
    test_q = {b.question_of(r) for r in lab if r["split"] == "test"}
    kept = [r for r in mc if not b.labeled_test(b.question_of(r))]
    assert test_q and not test_q & {b.question_of(r) for r in kept}
    assert all(": " in o for r in lab for o in r["options"])
