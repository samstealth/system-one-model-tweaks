"""Score a short list of options against a state with a System One head. A CPU is enough.

A head does not read text. It reads backbone vectors: the last-token hidden state of Qwen3-8B,
L2 normalized, 4096 numbers, one for the state and one for each option. The answer is

    softmax over the options of   scale * cosine(state_head(state), action_head(option))

which is what the official CLM engine computes (clm/heads.py, clm/engine.py).

    from innovations.scorer import load_head

    clm = load_head("clm")                  # CLM-v0.1-8B from Hugging Face, cached after the first call
    s1 = load_head("s1-tweaks")             # S1-Tweaks v0.1 from this repo's release page
    router = load_head("s1-router")         # S1-Router v0.1: stronger at tools and routing, no T-Rex skill
    mine = load_head("runs/my_head.pt")     # a local file, an https URL, or hf://org/repo/file.pt

    p = s1.probs(state_vec, option_vecs)    # (4096,), (K, 4096)            -> (K,), sums to 1
    P = s1.probs(states, options, n_opts)   # (N, 4096), (N, K, 4096), (N,) -> (N, K), padding gets 0
    P = s1.probs(states, tool_vecs)         # (N, 4096), (K, 4096): one fixed option list for every state

    python -m innovations.scorer clm        # print the layout, parameter count and scale of a head

Checkpoints are read with torch.load(weights_only=True), which accepts tensors and plain Python
types only. Use torch 2.6 or newer: older versions had a flaw in that mode.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from .heads import HeadConfig, TwoTower
from .interpolate import mix

CLM_NAME = "CLM-8B"
CLM_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_FILE = "CLM_v0.1-8B.pt"
# The Hub commit our results were measured against (the file is byte-identical to
# jackyk02/qwen3_8b_head/qwen3_8b_posttrained_head.pt). Pinned so the numbers stay reproducible.
CLM_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"

RELEASE_URL = "https://github.com/samstealth/system-one-model-tweaks/releases/download/v0.1/"

# S1-Tweaks v0.1: 85% CLM's head + 15% S1-Router, blended in weight space (run wise_0.15). With the T-Rex safety
# shield on it survived 19 of 20 games against 20 of 20 for CLM's head. With the shield off the T-Rex skill is gone.
S1_TWEAKS_NAME = "S1-Tweaks v0.1"
S1_TWEAKS_FILE = "s1_tweaks_v0.1.pt"
S1_TWEAKS_URL = RELEASE_URL + S1_TWEAKS_FILE
S1_TWEAKS_SHA256 = "c0a32217d1288d8452046bda9bcb045b53b052b943ef7de42652691e057fd9d9"  # every fresh download is checked

# S1-Router v0.1: CLM's head post-trained on listwise choice data (run post_choice_hn_frompost). Much stronger
# at tool choice and routing, but it loses CLM's T-Rex skill (0/5 courses).
S1_ROUTER_NAME = "S1-Router v0.1"
S1_ROUTER_FILE = "s1_router_v0.1.pt"
S1_ROUTER_URL = RELEASE_URL + S1_ROUTER_FILE
S1_ROUTER_SHA256 = "d9f22c9fe0a52d4bd32fe1a6c965b3900fc854c44f1c6267f23bf6e897911b18"

CLM_ALIASES = ("clm", "clm-8b", "clm-v0.1-8b")
S1_TWEAKS_ALIASES = ("s1-tweaks", "s1-tweaks-v0.1", "s1_tweaks", "s1_tweaks_v0.1")
S1_ROUTER_ALIASES = ("s1-router", "s1-router-v0.1", "s1_router", "s1_router_v0.1")


class HeadNotAvailable(FileNotFoundError):
    """A head could not be fetched. The message says why and what to do instead."""


def cache_dir():
    """Where downloaded heads are kept. Override with the S1_TWEAKS_CACHE environment variable."""
    return os.environ.get("S1_TWEAKS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "system-one-tweaks")


def _download(url, dest, force=False, timeout=60):
    """Fetch url to dest through a .part file, so an interrupted download never leaves a broken head."""
    if os.path.isfile(dest) and not force:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {url}", file=sys.stderr, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "system-one-tweaks"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def download_clm_head(revision=CLM_REVISION):
    """CLM's official head (Contrastive-LM/CLM-v0.1-8B, Apache 2.0) through huggingface_hub. Returns the local path."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(CLM_REPO, CLM_FILE, revision=revision)
    except Exception as e:  # no network, a proxy, a renamed file: say what to do instead
        raise HeadNotAvailable(
            f"Could not get CLM's head ({CLM_REPO}, file {CLM_FILE}) from Hugging Face: {type(e).__name__}: {e}\n"
            f"Download it by hand from https://huggingface.co/{CLM_REPO} and pass the file path instead."
        ) from e


def _download_release(name, fname, url, sha256, env, how_to_build, force=False):
    """One of this repo's released heads, from the release page into cache_dir(). Returns the local path."""
    local = os.environ.get(env)
    if local:
        if not os.path.isfile(local):
            raise HeadNotAvailable(f"{env} is set to {local}, which is not a file.")
        return local
    options = (f"  - use a local copy: give its path in place of the name (in the demo: --head /path/to/{fname}),\n"
               f"    or set {env} to that path\n"
               f"  - build it yourself with {how_to_build} (see docs/REPRODUCE.md)")
    dest = os.path.join(cache_dir(), fname)
    fresh = force or not os.path.isfile(dest)
    try:
        _download(url, dest, force)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HeadNotAvailable(f"{name} is not published yet: {url} answered HTTP 404, "
                                   f"so the v0.1 release is not live.\nUntil it is, you can:\n{options}") from None
        raise HeadNotAvailable(f"Could not download {name} from {url}: HTTP {e.code}.\nYou can:\n{options}") from None
    except OSError as e:  # URLError, timeouts, no network
        raise HeadNotAvailable(f"Could not download {name} from {url}: {type(e).__name__}: {e}\n"
                               f"You can:\n{options}") from None
    if fresh and sha256:
        with open(dest, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != sha256:
            os.remove(dest)
            raise HeadNotAvailable(f"The file at {url} does not match the published checksum (got sha256 {got}). It was not kept.")
    return dest


def download_s1_tweaks(url=S1_TWEAKS_URL, force=False):
    """S1-Tweaks v0.1 from the release page. Returns the local path.

    Set S1_TWEAKS_HEAD=/path/to/s1_tweaks_v0.1.pt to use a local copy and skip the download.
    """
    return _download_release(S1_TWEAKS_NAME, S1_TWEAKS_FILE, url, S1_TWEAKS_SHA256 if url == S1_TWEAKS_URL else None,
                             "S1_TWEAKS_HEAD", "python -m innovations.interpolate", force)


def download_s1_router(url=S1_ROUTER_URL, force=False):
    """S1-Router v0.1 from the release page. Returns the local path.

    Set S1_ROUTER_HEAD=/path/to/s1_router_v0.1.pt to use a local copy and skip the download.
    """
    return _download_release(S1_ROUTER_NAME, S1_ROUTER_FILE, url, S1_ROUTER_SHA256 if url == S1_ROUTER_URL else None,
                             "S1_ROUTER_HEAD", "the post-training command", force)


def _read(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"{path} could not be read as a head checkpoint ({type(e).__name__}: {str(e)[:200]}). "
                         "The loader accepts tensors and plain Python types only (torch.load with weights_only=True).") from e
    if not isinstance(ck, dict):
        raise ValueError(f"{path} holds a {type(ck).__name__}, not a checkpoint dict.")
    return ck


def _model_from_checkpoint(ck, where="checkpoint"):
    """Build a TwoTower from CLM's layout (state_head / action_head / logit_scale / cfg) or from this
    repo's trainer layout (model / cfg, written by TwoTower.save)."""
    if "state_head" in ck and "action_head" in ck:
        sd, cfg = ck["state_head"], dict(ck.get("cfg") or {})
        n_hidden = len({k.split(".")[1] for k in sd if k.startswith("hidden.")})
        bad = []
        if cfg.get("activation", "gelu") != "gelu":
            bad.append(f"activation={cfg['activation']}")
        if cfg.get("residual", False):
            bad.append("residual=True")
        if n_hidden and not any(k.startswith("norms.") for k in sd):
            bad.append("layernorm=False")
        if bad:
            raise ValueError(f"{where}: unsupported head layout ({', '.join(bad)}). This loader builds GELU heads with "
                             "LayerNorm and no residual, the layout of CLM-v0.1-8B.")
        m = TwoTower(HeadConfig(in_dim=sd["inp.weight"].shape[1], width=sd["inp.weight"].shape[0],
                                depth=n_hidden + 2, proj=sd["out.weight"].shape[0]))
        m.state_head.load_state_dict({k: v.float() for k, v in sd.items()})
        m.action_head.load_state_dict({k: v.float() for k, v in ck["action_head"].items()})
        m.logit_scale.data.copy_(torch.as_tensor(ck["logit_scale"]).float())
        return m
    if "model" in ck and "cfg" in ck:
        c = dict(ck["cfg"])
        c["matryoshka"] = tuple(c.get("matryoshka", ()))
        m = TwoTower(HeadConfig(**c))
        m.load_state_dict(ck["model"])
        return m
    raise ValueError(f"{where}: expected the keys state_head, action_head and logit_scale (CLM's layout) or "
                     f"model and cfg (this repo's trainer); found {sorted(ck)[:8]}.")


class Scorer:
    """A loaded head. Everything runs under no_grad and returns numpy arrays."""

    def __init__(self, model, name="head", source=None, device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.name, self.source, self.device = name, source, device

    # ---------------------------------------------------------------- facts about the head
    @property
    def in_dim(self):
        return self.model.cfg.in_dim

    @property
    def proj_dim(self):
        return self.model.cfg.proj

    @property
    def scale(self):
        """exp(logit_scale), capped at 100 as in the CLM engine."""
        return float(self.model.scale())

    @property
    def logit_scale(self):
        return float(self.model.logit_scale)

    @property
    def n_params(self):
        """Trainable weights of the head(s), not counting the scale."""
        return self.model.n_params()

    def describe(self):
        c = self.model.cfg
        dims = " -> ".join(str(d) for d in [c.in_dim] + [c.width] * (c.depth - 1) + [c.proj])
        return (f"{self.name}: {self.n_params:,} parameters, tower={c.tower}, {dims}, "
                f"scale {self.scale:.1f} (logit_scale {self.logit_scale:.4f})")

    def __repr__(self):
        return f"<Scorer {self.describe()}>"

    # ---------------------------------------------------------------- scoring
    def _vec(self, x, normalize=True):
        if not torch.is_tensor(x):
            x = np.asarray(x)
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"{self.name} expects {self.in_dim}-d backbone vectors, got {x.shape[-1]}-d. "
                             "A head only works with the backbone it was trained against.")
        # astype copies, so float16 files and read-only memory maps are fine
        t = (x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32))).to(self.device, torch.float32)
        return F.normalize(t, dim=-1) if normalize else t

    @torch.no_grad()
    def embed_state(self, state_vecs, normalize=True):
        """Backbone state vectors (..., in_dim) -> unit embeddings (..., proj_dim)."""
        return self.model.encode_state(self._vec(state_vecs, normalize)).cpu().numpy()

    @torch.no_grad()
    def embed_options(self, option_vecs, normalize=True):
        """Backbone option vectors (..., in_dim) -> unit embeddings (..., proj_dim). Cache these for a fixed
        option list: logits are then  scorer.scale * state_embedding @ option_embeddings.T ."""
        return self.model.encode_action(self._vec(option_vecs, normalize)).cpu().numpy()

    @torch.no_grad()
    def logits(self, state_vec, option_vecs, n_opts=None, normalize=True):
        """scale * cosine for every option. Shapes as in probs(). Padded options get -inf."""
        s_nd = state_vec.ndim if hasattr(state_vec, "ndim") else np.ndim(state_vec)
        o_nd = option_vecs.ndim if hasattr(option_vecs, "ndim") else np.ndim(option_vecs)
        if (s_nd, o_nd) not in ((1, 2), (2, 2), (2, 3)):
            raise ValueError("pass state (D,) with options (K, D), or states (N, D) with options (N, K, D) or (K, D); "
                             f"got {s_nd}-d states and {o_nd}-d options")
        scale = self.model.scale()
        if o_nd == 2:  # one option list, one or many states
            zs = self.model.encode_state(self._vec(state_vec, normalize).reshape(-1, self.in_dim))
            lg = scale * zs @ self.model.encode_action(self._vec(option_vecs, normalize)).t()
        else:
            N, K = len(option_vecs), option_vecs.shape[1]
            if len(state_vec) != N:
                raise ValueError(f"{len(state_vec)} states but {N} option lists")
            step = max(1, (1 << 26) // (K * self.in_dim))  # about 256 MB of float32 per chunk
            out = []
            for i in range(0, N, step):
                zs = self.model.encode_state(self._vec(state_vec[i:i + step], normalize))
                zo = self.model.encode_action(self._vec(option_vecs[i:i + step], normalize))
                out.append(scale * torch.einsum("nd,nkd->nk", zs, zo))
            lg = torch.cat(out) if out else torch.zeros(0, K)
        if n_opts is not None:
            n = torch.as_tensor(np.asarray(n_opts).astype(np.int64), device=lg.device).reshape(-1)
            if len(n) != len(lg) or (len(n) and (n.min() < 1 or n.max() > lg.shape[1])):
                raise ValueError(f"n_opts needs one count per state, each between 1 and {lg.shape[1]}")
            lg = lg.masked_fill(torch.arange(lg.shape[1], device=lg.device)[None] >= n[:, None], float("-inf"))
        lg = lg.cpu().numpy()
        return lg[0] if s_nd == 1 else lg

    def probs(self, state_vec, option_vecs, n_opts=None, temperature=1.0, normalize=True):
        """Probability of each option: softmax(scale * cosine / temperature).

        state (D,)   + options (K, D)     -> (K,)
        states (N, D) + options (N, K, D) -> (N, K)   n_opts (N,) marks how many of the K are real; the rest get 0
        states (N, D) + options (K, D)    -> (N, K)   the same option list for every state

        Inputs are raw backbone vectors. They are L2 normalized here, as the CLM engine does, unless normalize=False.
        """
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        lg = torch.from_numpy(self.logits(state_vec, option_vecs, n_opts, normalize))
        return torch.softmax(lg / temperature, dim=-1).numpy()

    def pick(self, state_vec, option_vecs, n_opts=None, normalize=True):
        """Index of the most likely option: an int for one state, an array for a batch."""
        i = self.logits(state_vec, option_vecs, n_opts, normalize).argmax(-1)
        return int(i) if np.ndim(i) == 0 else i

    # ---------------------------------------------------------------- saving
    def checkpoint(self):
        """The head as a dict in CLM's checkpoint layout, which clm-serve --ckpt can serve."""
        if self.model.cfg.tower != "separate":
            raise ValueError("CLM's layout holds two separate heads; this head uses tower=" + self.model.cfg.tower)
        c = self.model.cfg
        cpu = lambda sd: {k: v.detach().cpu().clone() for k, v in sd.items()}
        return {"state_head": cpu(self.model.state_head.state_dict()), "action_head": cpu(self.model.action_head.state_dict()),
                "logit_scale": self.model.logit_scale.detach().cpu().clone(), "hidden_size": c.in_dim, "projection_dim": c.proj,
                "cfg": {"hidden_size": c.in_dim, "projection_dim": c.proj, "width": c.width, "depth": c.depth,
                        "activation": "gelu", "layernorm": True, "residual": False}}

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.checkpoint(), path)
        return path


def load_head(src, device="cpu", name=None):
    """Load a head and return a Scorer.

    src is one of:
      "clm"                       CLM's official head, downloaded from Hugging Face
      "s1-tweaks"                 S1-Tweaks v0.1, downloaded from this repo's release page
      "s1-router"                 S1-Router v0.1, downloaded from this repo's release page
      a local path                a .pt file in CLM's checkpoint layout (or this repo's trainer layout)
      an http(s) URL              downloaded once into cache_dir()
      "hf://org/repo/file.pt"     a file in a Hugging Face model repo
    """
    if isinstance(src, Scorer):
        return src
    if isinstance(src, TwoTower):
        return Scorer(src, name or "head", None, device)
    src = os.fspath(src)
    key = src.strip().lower()
    if key in CLM_ALIASES:
        path, label = download_clm_head(), CLM_NAME
    elif key in S1_TWEAKS_ALIASES:
        path, label = download_s1_tweaks(), S1_TWEAKS_NAME
    elif key in S1_ROUTER_ALIASES:
        path, label = download_s1_router(), S1_ROUTER_NAME
    elif os.path.isfile(src):
        path, label = src, os.path.splitext(os.path.basename(src))[0]
    elif key.startswith(("http://", "https://")):
        base = os.path.basename(src.split("?")[0]) or "head.pt"
        label = os.path.splitext(base)[0]
        try:
            path = _download(src, os.path.join(cache_dir(), hashlib.sha1(src.encode()).hexdigest()[:12] + "-" + base))
        except OSError as e:
            raise HeadNotAvailable(f"Could not download a head from {src}: {type(e).__name__}: {e}") from None
    elif key.startswith("hf://"):
        parts = src[len("hf://"):].split("/")
        if len(parts) < 3:
            raise ValueError(f"{src}: write Hugging Face files as hf://org/repo/file.pt")
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
        except Exception as e:
            raise HeadNotAvailable(f"Could not get {src} from Hugging Face: {type(e).__name__}: {e}") from e
        label = os.path.splitext(parts[-1])[0]
    else:
        raise FileNotFoundError(f"No head at {src!r}. Pass \"clm\", \"s1-tweaks\", \"s1-router\", a local .pt file, an https URL "
                                "or hf://org/repo/file.pt.")
    return Scorer(_model_from_checkpoint(_read(path), path), name or label, path, device)


def blend(a, b, alpha, name=None, device="cpu"):
    """Weight-space blend of two heads with the same layout:  (1 - alpha) * a + alpha * b .

    a and b are Scorers or anything load_head accepts. S1-Tweaks v0.1 is
    blend("clm", "s1-router", 0.15). No training is involved.
    """
    A, B = load_head(a), load_head(b)
    ca, cb = A.checkpoint(), B.checkpoint()
    for side in ("state_head", "action_head"):
        if {k: v.shape for k, v in ca[side].items()} != {k: v.shape for k, v in cb[side].items()}:
            raise ValueError(f"cannot blend {A.name} and {B.name}: their {side} layouts differ")
    out = mix(ca, cb, float(alpha))
    return Scorer(_model_from_checkpoint(out, "blend"), name or f"blend({A.name}, {B.name}, {alpha:g})", None, device)


def main(argv=None):
    p = argparse.ArgumentParser(description="Print the layout, parameter count and scale of one or more heads.")
    p.add_argument("heads", nargs="*", default=["clm"],
                   help='"clm", "s1-tweaks", "s1-router", a local .pt file, an https URL or hf://org/repo/file.pt (default: clm)')
    a = p.parse_args(argv)
    code = 0
    for h in a.heads:
        try:
            s = load_head(h)
        except (OSError, ValueError) as e:
            print(f"{h}: {e}", file=sys.stderr)
            code = 1
            continue
        print(s.describe())
        print(f"  file: {s.source}")
    return code


if __name__ == "__main__":
    sys.exit(main())
