"""Weight-space interpolation between two heads with the same layout (WiSE-FT / task arithmetic).

    out = (1 - alpha) * A + alpha * B

With A = the authors' post-trained head (keeps T-Rex) and B = our listwise post-trained head started
from A (gains routing, tools and tricky Q&A), a small alpha can keep most of both.

    python -m innovations.interpolate --a /data/heads/qwen3_8b_posttrained_head.pt \\
        --b /data/runs/post_choice_hn_frompost/final_authors_layout.pt --alpha 0.3 --out /data/runs/wise_0.3/final_authors_layout.pt
"""
import argparse
import os

import torch


def mix(a, b, alpha):
    out = dict(a)
    for side in ("state_head", "action_head"):
        out[side] = {k: (1 - alpha) * a[side][k].float() + alpha * b[side][k].float() for k in a[side]}
    out["logit_scale"] = (1 - alpha) * torch.as_tensor(a["logit_scale"]).float() + alpha * torch.as_tensor(b["logit_scale"]).float()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    x = p.parse_args(argv)
    A = torch.load(x.a, map_location="cpu", weights_only=False)
    B = torch.load(x.b, map_location="cpu", weights_only=False)
    os.makedirs(os.path.dirname(x.out) or ".", exist_ok=True)
    torch.save(mix(A, B, x.alpha), x.out)
    print("wrote", x.out)


if __name__ == "__main__":
    main()
