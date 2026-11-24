"""Innovation 3, the part that runs without a GPU: quantize the trained heads and the cached
action embeddings, then measure accuracy and head latency.

A System One router scores one state against a fixed list of actions (tools, ticket
categories). Action embeddings are computed once and cached, so per decision the cost is
  backbone(state) + state head + one matrix-vector product against the cache.
This script covers the last two. The backbone, which dominates latency, needs a GPU:
see innovations/modal_quant_backbone.py.

Variants:
  fp32           baseline
  bf16           heads and cache in bfloat16
  int8-dyn       torch dynamic int8 quantization of the Linear layers (CPU kernels)
  int8-weights   per-output-channel symmetric int8 weights, dequantized on the fly (works on any device)
  cache-int8     fp32 heads, action cache stored as per-vector int8
  cache-binary   fp32 heads, action cache stored as sign bits, scored by Hamming-style dot product

    python -m innovations.quantize_heads --root /data --head /data/heads/qwen3_8b_midtrained_head.pt --out results/quant_heads.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from innovations import data
from innovations.heads import TwoTower


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach()
        s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        self.register_buffer("q", torch.round(w / s).to(torch.int8))
        self.register_buffer("s", s)
        self.register_buffer("b", lin.bias.detach().clone())

    def forward(self, x):
        return F.linear(x, self.q.float() * self.s, self.b)


def int8_weights(model):
    m = copy.deepcopy(model)
    for name, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, Int8Linear(child))
    return m


def quant_vec_int8(z):
    s = z.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(z / s).to(torch.int8), s


def head_bytes(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)  # counts packed int8 weights too
    return buf.tell()


@torch.no_grad()
def tricky_custom(enc_s, enc_a, score_fn, q, a, n, cnt, step=2048, limit=None):
    hit = tot = 0
    end = min(len(q), limit) if limit else len(q)
    for i in range(0, end, step):
        j = min(i + step, end)
        c = torch.from_numpy(np.asarray(cnt[i:j]).astype(np.int64))
        zs = enc_s(torch.from_numpy(np.asarray(q[i:j])).float())
        za = enc_a(torch.from_numpy(np.asarray(a[i:j])).float())
        K, D = n.shape[1], n.shape[2]
        zn = enc_a(torch.from_numpy(np.asarray(n[i:j])).float().view(-1, D)).view(j - i, K, -1)
        pos = score_fn(zs[:, None], za[:, None])[:, 0]
        neg = score_fn(zs[:, None], zn).masked_fill(torch.arange(K)[None] >= c[:, None], -1e9)
        v = c > 0
        hit += (pos[v] > neg[v].max(1).values).sum().item()
        tot += v.sum().item()
    return round(hit / max(tot, 1), 4)


def latency(enc_s, cache, score_fn, reps=200, in_dim=4096):
    x = torch.randn(1, in_dim)
    for _ in range(10):
        score_fn(enc_s(x)[:, None], cache[None]).softmax(-1)
    t = time.perf_counter()
    for _ in range(reps):
        score_fn(enc_s(x)[:, None], cache[None]).softmax(-1)
    return round((time.perf_counter() - t) / reps * 1000, 3)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data")
    p.add_argument("--head", required=True)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--n-actions", type=int, default=10000)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    torch.set_grad_enabled(False)
    base = TwoTower.from_authors(a.head) if "qwen3_8b" in a.head else TwoTower.load(a.head)
    base.eval()
    hn = data.load_chunk(a.root, data.HOLD_HN[0], negs=True, limit=a.limit)
    args = (hn["q"], hn["a"], hn["n"], hn["c"])
    dot = lambda zs, za: (zs * za).sum(-1)
    acts = torch.from_numpy(np.asarray(data.load_chunk(a.root, data.HOLD_GEN[0])["a"][:a.n_actions])).float()
    res = {}

    def run(name, model, enc_cast=None, cache_q=None):
        enc_s = lambda x: model.encode_state(x if enc_cast is None else x.to(enc_cast)).float()
        enc_a = lambda x: model.encode_action(x if enc_cast is None else x.to(enc_cast)).float()
        if cache_q == "int8":
            def enc_a(x, _m=model):
                qv, s = quant_vec_int8(_m.encode_action(x))
                return qv.float() * s
        if cache_q == "binary":
            def enc_a(x, _m=model):
                return torch.sign(_m.encode_action(x))
        cache = enc_a(acts)
        if cache_q == "int8":
            cache_bytes = acts.shape[0] * (cache.shape[1] + 4)
        elif cache_q == "binary":
            cache_bytes = acts.shape[0] * cache.shape[1] // 8
        else:
            cache_bytes = cache.numel() * (2 if enc_cast == torch.bfloat16 else 4)
        res[name] = {"tricky_top1": tricky_custom(enc_s, enc_a, dot, *args),
                     "head_ms_cpu": latency(enc_s, cache, dot),
                     "head_MB": round(head_bytes(model) / 1e6, 1),
                     f"cache_MB_{a.n_actions}_actions": round(cache_bytes / 1e6, 2)}
        print(name, json.dumps(res[name]), flush=True)

    run("fp32", base)
    run("bf16", copy.deepcopy(base).to(torch.bfloat16), enc_cast=torch.bfloat16)
    try:
        dq = torch.ao.quantization.quantize_dynamic(copy.deepcopy(base), {nn.Linear}, dtype=torch.qint8)
        run("int8-dyn", dq)
    except Exception as e:  # not every torch build ships the CPU int8 kernels
        res["int8-dyn"] = {"error": repr(e)[:200]}
    run("int8-weights", int8_weights(base))
    run("cache-int8", base, cache_q="int8")
    run("cache-binary", base, cache_q="binary")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
