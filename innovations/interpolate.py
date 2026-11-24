"""Weight-space interpolation between two heads with the same layout (WiSE-FT / task arithmetic).

    out = (1 - alpha) * A + alpha * B

With A = the authors' post-trained head (it has the T-Rex skill) and B = our listwise post-trained
head started from A (better at routing and tool choice, no T-Rex skill), alpha trades one for the
other. S1-Tweaks v0.1 is alpha 0.15.

    python -m innovations.interpolate --a clm --b s1-router --alpha 0.15 --out my_blend.pt
    python -m innovations.interpolate --a /data/heads/qwen3_8b_posttrained_head.pt \\
        --b /data/runs/post_choice_hn_frompost/final_authors_layout.pt --alpha 0.15 --out /data/runs/wise_0.15/final_authors_layout.pt

--a and --b take a local .pt file in CLM's checkpoint layout, or one of the names "clm",
"s1-tweaks" and "s1-router", which are downloaded the way innovations/scorer.py does it.
Files are read with torch.load(weights_only=True).
"""
import argparse
import os

import torch

SIDES = ("state_head", "action_head")


def mix(a, b, alpha):
    out = dict(a)
    for side in SIDES:
        out[side] = {k: (1 - alpha) * a[side][k].float() + alpha * b[side][k].float() for k in a[side]}
    out["logit_scale"] = (1 - alpha) * torch.as_tensor(a["logit_scale"]).float() + alpha * torch.as_tensor(b["logit_scale"]).float()
    return out


def _load(src):
    """A local file as it is. The names clm, s1-tweaks and s1-router are downloaded first."""
    path = src
    if not os.path.isfile(src):
        from . import scorer  # imported here because scorer imports mix from this module
        key = src.strip().lower()
        if key in scorer.CLM_ALIASES:
            path = scorer.download_clm_head()
        elif key in scorer.S1_TWEAKS_ALIASES:
            path = scorer.download_s1_tweaks()
        elif key in scorer.S1_ROUTER_ALIASES:
            path = scorer.download_s1_router()
        else:
            raise FileNotFoundError(f'no head file at {src!r}. Pass a local .pt file or one of the names '
                                    '"clm", "s1-tweaks", "s1-router".')
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # not a checkpoint, or one that holds more than tensors and plain types
        raise ValueError(f"{src} could not be read with weights_only=True ({type(e).__name__}: {str(e)[:160]})") from e
    if not (isinstance(ck, dict) and all(k in ck for k in SIDES + ("logit_scale",))):
        raise ValueError(f"{src} is not in CLM's checkpoint layout (state_head, action_head, logit_scale).")
    return ck


def main(argv=None):
    p = argparse.ArgumentParser(description="Blend two heads in weight space: out = (1 - alpha) * A + alpha * B.")
    p.add_argument("--a", required=True, help='head A: a local .pt file, or "clm", "s1-tweaks", "s1-router"')
    p.add_argument("--b", required=True, help="head B, same forms as --a")
    p.add_argument("--alpha", type=float, required=True, help="share of head B, from 0 to 1")
    p.add_argument("--out", required=True)
    x = p.parse_args(argv)
    try:
        A, B = _load(x.a), _load(x.b)
    except (OSError, ValueError) as e:  # a missing file or a head that is not published yet: one line, no traceback
        raise SystemExit(f"interpolate: {e}")
    os.makedirs(os.path.dirname(x.out) or ".", exist_ok=True)
    torch.save(mix(A, B, x.alpha), x.out)
    print("wrote", x.out)


if __name__ == "__main__":
    main()
