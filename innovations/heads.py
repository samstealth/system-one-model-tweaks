"""Configurable System One heads and the contrastive loss used by every experiment.

The baseline (pipeline/train_heads.py) hard-codes two separate heads,
4096 -> 1536 -> 1536 (LayerNorm) -> 512 with GELU. Here every part of that is a knob:

  width, depth, proj      the "embedding dimensions" idea (innovation 2)
  tower="separate"        baseline: one head for states, one for actions
  tower="shared"          one head for both sides, one embedding space (innovation 4)
  tower="shared_adapter"  shared trunk plus a small per-side low-rank adapter and bias
  matryoshka=(64,128,..)  train one head whose first k output dims are a usable
                          k-dim embedding, so one run answers "which proj dim?"

A head built with the default config loads the authors' checkpoints unchanged
(same parameter names: inp, hidden, norms, out).
"""
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    in_dim: int = 4096
    width: int = 1536
    depth: int = 3          # number of Linear layers, as stored in the authors' cfg
    proj: int = 512
    tower: str = "separate"  # separate | shared | shared_adapter
    adapter_rank: int = 64
    matryoshka: tuple = field(default_factory=tuple)

    def to_dict(self):
        d = asdict(self)
        d["matryoshka"] = list(self.matryoshka)
        return d


class Head(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.in_dim, cfg.width)
        n_hidden = max(cfg.depth - 2, 0)
        self.hidden = nn.ModuleList([nn.Linear(cfg.width, cfg.width) for _ in range(n_hidden)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width) for _ in range(n_hidden)])
        self.out = nn.Linear(cfg.width, cfg.proj)
        self.act = nn.GELU()

    def trunk(self, x):
        x = self.act(self.inp(x))
        for lin, norm in zip(self.hidden, self.norms):
            x = self.act(norm(lin(x)))
        return x

    def forward(self, x):
        return self.out(self.trunk(x))


class SideAdapter(nn.Module):
    """Low-rank residual on the trunk output plus a bias, one per side (state or action)."""

    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(self.down(h))


class TwoTower(nn.Module):
    """Maps (state vectors, action vectors) to normalized embeddings under one of three tower layouts."""

    def __init__(self, cfg: HeadConfig, init_scale=1 / 0.07):
        super().__init__()
        self.cfg = cfg
        if cfg.tower == "separate":
            self.state_head, self.action_head = Head(cfg), Head(cfg)
        elif cfg.tower in ("shared", "shared_adapter"):
            self.head = Head(cfg)
            if cfg.tower == "shared_adapter":
                self.adapt = nn.ModuleDict({"s": SideAdapter(cfg.width, cfg.adapter_rank),
                                            "a": SideAdapter(cfg.width, cfg.adapter_rank)})
        else:
            raise ValueError(cfg.tower)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())

    def _side(self, x, side):
        if self.cfg.tower == "separate":
            return (self.state_head if side == "s" else self.action_head)(x)
        h = self.head.trunk(x)
        if self.cfg.tower == "shared_adapter":
            h = self.adapt[side](h)
        return self.head.out(h)

    def encode_state(self, x, dim=None):
        z = self._side(x, "s")
        return F.normalize(z[..., :dim] if dim else z, dim=-1)

    def encode_action(self, x, dim=None):
        z = self._side(x, "a")
        return F.normalize(z[..., :dim] if dim else z, dim=-1)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p is not self.logit_scale)

    @classmethod
    def from_authors(cls, path):
        """Load jackyk02/qwen3_8b_head checkpoints (or the baseline trainer's best.pt)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(HeadConfig())
        m.state_head.load_state_dict(ck["state_head"])
        m.action_head.load_state_dict(ck["action_head"])
        m.logit_scale.data.copy_(torch.as_tensor(ck["logit_scale"]).float())
        return m

    def export_authors(self, path):
        """Write the separate-tower model in the layout pipeline/run_trex.py and the CLM engine load."""
        assert self.cfg.tower == "separate", "the CLM engine only serves two separate heads"
        c = self.cfg
        torch.save({"state_head": self.state_head.state_dict(), "action_head": self.action_head.state_dict(),
                    "logit_scale": self.logit_scale.detach().cpu(), "hidden_size": c.in_dim, "projection_dim": c.proj,
                    "cfg": {"hidden_size": c.in_dim, "projection_dim": c.proj, "width": c.width, "depth": c.depth,
                            "activation": "gelu", "layernorm": True, "residual": False}}, path)

    def save(self, path, **extra):
        torch.save({"model": self.state_dict(), "cfg": self.cfg.to_dict(), **extra}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = dict(ck["cfg"]); c["matryoshka"] = tuple(c.get("matryoshka", ()))
        m = cls(HeadConfig(**c))
        m.load_state_dict(ck["model"])
        return m


def info_nce(zq, za, scale, doc=None, zn=None, ncnt=None, neg_weight=1.0):
    """Symmetric InfoNCE exactly as the baseline trainer does it.

    zq, za: [B, d] normalized. doc: [B] int64 ids; other pairs from the same document are masked.
    zn: [k, K, d] hard negatives for the first k rows (state->action direction only), ncnt: [k] counts.
    """
    lg = scale * zq @ za.t()
    if doc is not None:
        same = doc[None] == doc[:, None]
        same.fill_diagonal_(False)
        lg = lg.masked_fill(same, float("-inf"))
    y = torch.arange(len(zq), device=zq.device)
    back = F.cross_entropy(lg.t(), y)
    if zn is not None:
        k, K = zn.shape[:2]
        ln = scale * torch.einsum("bd,bkd->bk", zq[:k], zn)
        ln = ln.masked_fill(torch.arange(K, device=zq.device)[None] >= ncnt[:, None], float("-inf"))
        if neg_weight != 1.0:
            ln = ln + torch.log(torch.tensor(float(neg_weight), device=zq.device))
        if k < len(zq):
            ln = torch.cat([ln, torch.full((len(zq) - k, K), float("-inf"), device=zq.device)])
        forw = F.cross_entropy(torch.cat([lg, ln], 1), y)
    else:
        forw = F.cross_entropy(lg, y)
    return (forw + back) / 2


def model_loss(model, q, a, doc=None, n=None, ncnt=None):
    """Loss for a batch of raw backbone vectors. With matryoshka dims, the loss is averaged over prefixes."""
    dims = model.cfg.matryoshka or (None,)
    s = model.scale()
    zs_full, za_full = model._side(q, "s"), model._side(a, "a")
    zn_full = None
    if n is not None:
        k, K, D = n.shape
        zn_full = model._side(n.reshape(-1, D), "a").view(k, K, -1)
    total = 0.0
    for d in dims:
        cut = (lambda z: z[..., :d]) if d else (lambda z: z)
        zq, za = F.normalize(cut(zs_full), dim=-1), F.normalize(cut(za_full), dim=-1)
        zn = F.normalize(cut(zn_full), dim=-1) if zn_full is not None else None
        total = total + info_nce(zq, za, s, doc, zn, ncnt)
    return total / len(dims)
