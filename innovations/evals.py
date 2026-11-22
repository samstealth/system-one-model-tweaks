"""The two held-out tests from pipeline/final_eval.py, rewritten for any TwoTower and any output dim.

  general  in-batch retrieval over 1024-pair blocks, same-document pairs masked (gen_top1, gen_loss)
  tricky   one gold answer against its own up-to-10 hard negatives (tricky_top1, the 69% number)
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


def _T(x, dev):
    return torch.from_numpy(np.asarray(x)).to(dev).float()


@torch.no_grad()
def general(model, q, a, doc, dim=None, dev="cpu", block=1024, max_blocks=None):
    model.eval()
    block = min(block, len(q))
    s = model.scale()
    perm = np.random.default_rng(0).permutation(len(q))
    L, hit, tot = [], 0, 0
    for n_b, i in enumerate(range(0, len(perm) - block + 1, block)):
        if max_blocks and n_b >= max_blocks:
            break
        b = np.sort(perm[i:i + block])
        zq, za = model.encode_state(_T(q[b], dev), dim), model.encode_action(_T(a[b], dev), dim)
        d = torch.from_numpy(doc[b]).to(dev)
        same = d[None] == d[:, None]
        same.fill_diagonal_(False)
        lg = (s * zq @ za.t()).masked_fill(same, float("-inf"))
        y = torch.arange(len(b), device=dev)
        L.append(((F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2).item())
        hit += (lg.argmax(1) == y).sum().item()
        tot += len(b)
    model.train()
    return {"gen_loss": round(float(np.mean(L)), 4), "gen_top1": round(hit / tot, 4)}


@torch.no_grad()
def tricky(model, q, a, n, cnt, dim=None, dev="cpu", step=2048, limit=None):
    model.eval()
    hit = tot = 0
    end = min(len(q), limit) if limit else len(q)
    for i in range(0, end, step):
        j = min(i + step, end)
        c = torch.from_numpy(np.asarray(cnt[i:j]).astype(np.int64)).to(dev)
        zq, za = model.encode_state(_T(q[i:j], dev), dim), model.encode_action(_T(a[i:j], dev), dim)
        K, D = n.shape[1], n.shape[2]
        zn = model.encode_action(_T(n[i:j], dev).view(-1, D), dim).view(j - i, K, -1)
        neg = torch.einsum("bh,bkh->bk", zq, zn).masked_fill(torch.arange(K, device=dev)[None] >= c[:, None], -9.0)
        v = c > 0
        hit += ((zq * za).sum(-1)[v] > neg[v].max(1).values).sum().item()
        tot += v.sum().item()
    model.train()
    return {"tricky_top1": round(hit / max(tot, 1), 4), "tricky_n": tot}


@torch.no_grad()
def head_latency(model, dev="cpu", n_actions=1000, reps=50, in_dim=4096):
    """Time one decision with the head alone: one state vector against n cached action embeddings.
    Action embeddings are computed once and cached, which is how a router with a fixed tool list runs."""
    model.eval()
    x = torch.randn(1, in_dim, device=dev)
    acts = model.encode_action(torch.randn(n_actions, in_dim, device=dev))
    for _ in range(5):
        (model.encode_state(x) @ acts.t()).softmax(-1)
    t = time.perf_counter()
    for _ in range(reps):
        (model.encode_state(x) @ acts.t()).softmax(-1)
    model.train()
    return round((time.perf_counter() - t) / reps * 1000, 3)
