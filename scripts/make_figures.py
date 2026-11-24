#!/usr/bin/env python3
"""Regenerate every figure in assets/ from the result files.

    pip install matplotlib
    python scripts/make_figures.py                      # every figure, light and dark
    python scripts/make_figures.py --only hero,heldout
    python scripts/make_figures.py --numbers            # also print the numbers each chart shows, as JSON on stdout
    python scripts/make_figures.py --results DIR --out DIR   # other folders, for testing

Outputs (in assets/):
    hero, heldout, blend-tradeoff, quantization, head-layouts, backbones, architecture
        each as <name>-light.svg and <name>-dark.svg, transparent background, text as paths
    social-preview.png  (1280 x 640)

Where the numbers come from
    Everything is read at run time from results/innovations/*.json. The few facts that exist only
    in the markdown write-ups live in MD_ONLY below, each with its source file.

Result files and what the charts take from them
    auth_post.json, wise_<w>.json, post_choice_hn_frompost.json   in-distribution evals per head
    heldout.json             {"rows", "dropped", "heads": {head: {source: top-1 %}}}
    heldout_paired.json      paired bootstrap interval of each held-out gain
    data_checks.json         options per row and chance level of each choice test (scripts/data_checks.py)
    trex.json                T-Rex summaries. A run stored as <head>_<N>seeds is a later run of
                             the same head; the hero adds the games of all runs of a head
    deepswe.json             {"seeds", "heads": {head: {"solved": [...], "mean": x}}}
    backbone_latency_l40s.json, auth_bf16.json, auth_fp8.json, auth_awq.json, quant_heads_mid.json
    bb_*.json, cpu_sweep/*.json

Colours: grey is CLM-8B, blue is S1-Tweaks v0.1 (and our blends), teal is S1-Router v0.1, in
every chart. Red marks a loss or a broken variant.

Fonts: Inter if installed, else matplotlib's DejaVu Sans. Text is written as paths, so the SVGs
look the same everywhere.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.patches import Circle, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------
# Facts that exist ONLY in the markdown write-ups. No JSON file holds them, so they are kept
# here, in one place, each with its source. Everything else is read from the JSON files.
# ---------------------------------------------------------------------------------------------
MD_ONLY = {
    # innovations/RESULTS.md, last line: "Total Modal spend for all three rounds: about $69"
    "modal_spend_usd": "69",
    # innovations/RESULTS.md section 4: "the same 96k-pair slice"
    "backbone_slice_pairs": "96k",
    # innovations/RESULTS.md section 3: "12 published chunks (1.2M pairs), batch 1,024, 3,000 steps, seed 1234"
    # (so every layout was trained once: one seed)
    "cpu_sweep_pairs": "1.2M",
    # innovations/RESULTS.md section 2: quant_heads_mid.json was measured on the authors' mid-trained head
    "quant_head_name": "CLM's mid-trained head",
    # innovations/RESULTS.md section 2, latency table header: "batch 1, one L40S, vLLM 0.11"
    "latency_setup": "batch 1, one L40S",
    # innovations/RESULTS.md section 7: MMLU is a held-out split of a source we trained on (its
    # auxiliary_train set), the other sources are new to our heads
    "heldout_split_only": ("mmlu", "MMLU"),
    # docs/REPRODUCE.md section 2.7: "--seeds N plays seeds 0 to N-1", so a later run replays the
    # courses of an earlier, shorter one (results/innovations/trex_15seeds/*.json list seeds 0 to 14)
    "trex_seeds_start_at": 0,
    # docs/REPRODUCE.md section 2.9: each head is fine-tuned on DeepSWE steps with CLM's finetune.py, then scored
    "deepswe_setup": "each head is first fine-tuned on DeepSWE steps with CLM's own script, then scored",
    # results/innovations/day3_2026-11-23.md section 3: "DeepSWE over 5 seeds (38 held-out tasks)";
    # used only if the note inside deepswe.json stops saying so
    "deepswe_tasks": 38,
    # innovations/RESULTS.md title: "Results (22 and 23 November 2026)"
    "measured_on": "22 and 23 Nov 2026",
    # results/innovations/slice_run_2026-11-22.md: the CPU sweep and the backbone screen ran on day 1
    "measured_day1": "22 Nov 2026",
}

# Names (not numbers). Run names are the file names in results/innovations/.
PROJECT = "System One Model Tweaks"
OURS_NAME = "S1-Tweaks v0.1"
ROUTER_NAME = "S1-Router v0.1"
CLM_NAME = "CLM-8B"
REPO = "github.com/samstealth/system-one-model-tweaks"
RELEASE_ALPHA = 0.15                    # S1-Tweaks v0.1 is the run wise_0.15
CLM_EVAL_RUN = "auth_post"              # CLM's post-trained head, scored with our evals
CLM_TREX_RUN = "authors_post"           # the same head in trex.json, deepswe.json and heldout.json
FULL_RUN = "post_choice_hn_frompost"    # our fully post-trained head (blend weight 1.0) = S1-Router v0.1
OURS_RUN = f"wise_{RELEASE_ALPHA:g}"

FLAT_BELOW = 1.0   # a change smaller than this many points is printed in grey, not green or red

CHOICE_LABELS = {
    "choice_glaive_fc": "Tool choice (Glaive)",
    "choice_clinc_oos": "Intent routing (CLINC)",
    "choice_banking77": "Ticket routing (Banking77)",
    "choice_labeled": "Label reading",
    "choice_arc": "ARC",
    "choice_csqa": "CommonsenseQA",
}
# Held-out sources: key in heldout.json -> (name, row label in the hero, group in the held-out chart).
# The kinds are the ones in the table of innovations/RESULTS.md section 7.
HELDOUT_SOURCES = [
    ("bfcl", "BFCL v3", "Tool choice (BFCL)", "Tool choice"),
    ("massive", "MASSIVE", "Intent routing (MASSIVE)", "Routing"),
    ("hwu64", "HWU64", "Intent routing (HWU64)", "Routing"),
    ("bitext", "Bitext", "Support routing (Bitext)", "Routing"),
    ("sciq", "SciQ", "Science questions (SciQ)", "Science questions"),
    ("qasc", "QASC", "Science questions (QASC)", "Science questions"),
    ("mmlu", "MMLU test", "Mixed knowledge (MMLU)", "General and medical knowledge"),
    ("medmcqa", "MedMCQA", "Medical questions (MedMCQA)", "General and medical knowledge"),
]
HELDOUT_KIND = {  # shown under the source name in the held-out chart
    "bfcl": "tool choice", "massive": "intent routing", "hwu64": "intent routing", "bitext": "support routing",
    "sciq": "science", "qasc": "science", "mmlu": "mixed subjects", "medmcqa": "medical",
}
BLEND_SERIES = [  # the held-out tests drawn on the blend chart: key, name, kind
    ("bfcl", "BFCL", "tool choice"),
    ("massive", "MASSIVE", "intent routing"),
]
BACKBONE_LABELS = {
    "qwen3-8b": "Qwen3-8B",
    "muse-glimmer-30b": "Muse Glimmer 30B",
    "inkling-2-mini": "Inkling 2 Mini",
}
LATENCY_LABELS = {  # key in backbone_latency_l40s.json -> (label, accuracy file or None)
    "qwen3-8b": ("bf16", "auth_bf16.json"),
    "qwen3-8b-fp8": ("FP8", "auth_fp8.json"),
    "qwen3-8b-awq": ("AWQ 4-bit", "auth_awq.json"),
}
QUANT_HEAD_ROWS = [  # key in quant_heads_mid.json -> (label, group)
    ("fp32", "fp32, as released", "Head weights"),
    ("bf16", "bf16", "Head weights"),
    ("int8-weights", "int8 weights, per channel", "Head weights"),
    ("int8-dyn", "int8 weights and activations", "Head weights"),
    ("cache-int8", "int8 action cache", "Action cache"),
    ("cache-binary", "1-bit action cache", "Action cache"),
]
LAYOUT_LABELS = {  # cpu_sweep file stem -> label
    "base_w1536_p512": "CLM layout",
    "shared_p512": "Shared head",
    "shared_adapter_p512": "Shared + adapters",
    "w768_p512": "768 wide",
    "depth2_p512": "2 layers",
    "p128": "128 out",
    "mrl_p1024": "Matryoshka",
}

# ---------------------------------------------------------------------------------------------
# Theme tokens. Series colours follow one rule in every chart:
#   clm     grey   CLM-8B, or the unquantized reference
#   ours    blue   S1-Tweaks v0.1 and our other blends and variants
#   router  teal   S1-Router v0.1
#   soft    a second step of the same blue, for a middle variant between grey and blue (FP8)
#   bad     red    a loss or a broken variant
# "ours_dim" is only the connector between two marks, never a series. "fade" is the opacity of
# the marks of the hero's in-distribution rows.
# Every pair that shares a chart was run through the dataviz palette check (OKLab distance,
# normal vision >= 15, simulated protan/deutan >= 8) on GitHub's page colours #ffffff and
# #0d1117. Teal is 2.8:1 on white, so every teal bar carries its value as text. Text tokens are
# 4.5:1 or better.
# ---------------------------------------------------------------------------------------------
THEMES = {
    "light": dict(
        name="light", surface="#ffffff", ink="#1f2328", ink2="#59636e", muted="#6e7781",
        grid="#e6eaef", axis="#c8d1da",
        ours="#2a78d6", router="#1baf7a", soft="#86b6ef", ours_dim="#86b6ef", clm="#7d828b", bad="#d03b3b",
        good_text="#006300", bad_text="#c5302f", fade=0.5,
        box="#f6f8fa", box_edge="#d1d9e0", ours_box="#e6f0fc", ours_box_edge="#2a78d6",
    ),
    "dark": dict(
        name="dark", surface="#0d1117", ink="#f0f6fc", ink2="#c9d1d9", muted="#8b949e",
        grid="#21262d", axis="#3d444d",
        ours="#3987e5", router="#199e70", soft="#9ec5f4", ours_dim="#1c5cab", clm="#676f78", bad="#d03b3b",
        good_text="#0ca30c", bad_text="#e66767", fade=0.7,
        box="#161b22", box_edge="#30363d", ours_box="#10233f", ours_box_edge="#3987e5",
    ),
}

W = 800          # every chart is drawn 800 units wide; 1 unit = 1 px when the README shows it 800 px wide
MINUS = "−"      # U+2212, a real minus sign
TIMES = "×"
ARROW = "→"
DOT = "·"


class Skip(Exception):
    """A figure cannot be drawn because a result file is missing."""


def warn(msg: str) -> None:
    print(f"  [warn] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        warn(f"could not read {path.name}: {e}")
        return None


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def pct(v: float) -> str:
    """Accuracy fraction -> text, for the head evals (auth_*, wise_*, post_*).

    Plain float formatting, exactly as the tables in results/innovations/ print it,
    so 0.6665 shows as 66.6 there and here.
    """
    return f"{v * 100:.1f}"


def up1(v: float) -> str:
    """One decimal, halves rounded up on the decimal value: 15.45 -> 15.5.

    innovations/RESULTS.md rounds its latency, backbone and CPU-sweep tables this way, and
    the charts built from those files use the same rule so figure and table always agree.
    """
    return str(Decimal(repr(float(v))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct_up(v: float) -> str:
    """Accuracy fraction -> text with halves rounded up: 0.4825 -> 48.3 (see up1)."""
    return str((Decimal(repr(float(v))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def points_between(a: float, b: float) -> float:
    """b - a in accuracy points for two stored fractions, exact on the decimal values: 0.6636, 0.6563 -> -0.7."""
    d = (Decimal(repr(float(b))) - Decimal(repr(float(a)))) * 100
    return float(d.quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN))


def signed(d: float, digits: int = 1) -> str:
    d = round(d, digits)
    if d == 0:
        return f"{0:.{digits}f}"
    return f"{'+' if d > 0 else MINUS}{abs(d):.{digits}f}"


def alpha_txt(a: float) -> str:
    return f"{a:g}"


def plural(n, word: str) -> str:
    return word if abs(n) == 1 else word + "s"


_SEEDS_TAG = re.compile(r"^(.*)_(\d+)seeds$")


def run_for_alpha(a: float) -> str:
    """Name of the head at a blend weight, as trex.json and heldout.json spell it."""
    if a == 0:
        return CLM_TREX_RUN
    if a == 1:
        return FULL_RUN
    return f"wise_{a:g}"


def _alpha_of(name: str):
    if name == CLM_TREX_RUN:
        return 0.0
    if name == FULL_RUN:
        return 1.0
    if name.startswith("wise_") and not name.startswith("wise_agent_"):
        try:
            return float(name[len("wise_"):])
        except ValueError:
            return None
    return None


class Data:
    def __init__(self, res: Path):
        self.res = res
        self.clm = load_json(res / f"{CLM_EVAL_RUN}.json")
        self.full = load_json(res / f"{FULL_RUN}.json")
        self.wise = {}
        for p in sorted(res.glob("wise_*.json")):
            if p.name.startswith("wise_agent_"):
                continue
            try:
                a = float(p.stem[len("wise_"):])
            except ValueError:
                warn(f"{p.name}: cannot read a blend weight from the file name, skipped")
                continue
            d = load_json(p)
            if isinstance(d, dict):
                self.wise[a] = d
        self.ours = self.wise.get(RELEASE_ALPHA)
        self.trex, self.trex_more = self._load_trex()
        self.deepswe = self._load_deepswe()
        self.heldout = self._load_heldout()
        self.paired = load_json(res / "heldout_paired.json")
        self.checks = load_json(res / "data_checks.json")
        self.latency = load_json(res / "backbone_latency_l40s.json")
        self.quant_heads = load_json(res / "quant_heads_mid.json")
        self.backbones = {}
        for p in sorted(res.glob("bb_*.json")):
            d = load_json(p)
            if isinstance(d, dict):
                self.backbones[p.stem[len("bb_"):]] = d
        self.sweep = {}
        for p in sorted((res / "cpu_sweep").glob("*.json")):
            d = load_json(p)
            if isinstance(d, dict):
                self.sweep[p.stem] = d

    # -- T-Rex ---------------------------------------------------------------------------
    def _load_trex(self):
        """Read trex*.json. Returns (first, more).

        first[head][mode]   the run stored under the head's plain name (5 seeds each so far)
        more[head][mode]    later runs of the same head, stored as <head>_<N>seeds, as a list
        mode is "shield" or "noshield".
        """
        first: dict = {}
        more: dict = {}
        for p in sorted(self.res.glob("trex*.json")):
            d = load_json(p)
            if not isinstance(d, dict):
                continue
            block = d.get("runs", d)
            if not isinstance(block, dict):
                continue
            for name, run in block.items():
                if not isinstance(run, dict):
                    continue
                tag = _SEEDS_TAG.match(name)
                head = tag.group(1) if tag else name
                for mode in ("shield", "noshield"):
                    m = run.get(mode)
                    if not (isinstance(m, dict) and is_num(m.get("survived")) and is_num(m.get("seeds")) and m["seeds"]):
                        continue
                    if tag:
                        more.setdefault(head, {}).setdefault(mode, []).append(m)
                    else:
                        first.setdefault(head, {})[mode] = m
        return first, more

    def trex_first(self, head: str, mode: str = "shield"):
        return self.trex.get(head, {}).get(mode)

    def trex_extra(self, head: str, mode: str = "shield") -> list:
        return self.trex_more.get(head, {}).get(mode, [])

    def trex_pooled(self, head: str, mode: str = "shield"):
        """Every game a head played in this mode, added over its runs: survived, seeds, deaths, runs."""
        runs = ([self.trex_first(head, mode)] if self.trex_first(head, mode) else []) + self.trex_extra(head, mode)
        if not runs:
            return None
        return dict(survived=sum(r["survived"] for r in runs), seeds=sum(r["seeds"] for r in runs),
                    deaths=sum(r.get("deaths", 0) for r in runs), runs=[r["seeds"] for r in runs])

    # -- DeepSWE ---------------------------------------------------------------------------
    def _load_deepswe(self) -> dict:
        """head -> dict(solved=[one count per seed], mean=..., total=tasks)."""
        d = load_json(self.res / "deepswe.json")
        out: dict = {}
        if not isinstance(d, dict):
            return out
        m = re.search(r"(\d+)\s+held-out tasks", str(d.get("note", "")))
        total = int(m.group(1)) if m else MD_ONLY["deepswe_tasks"]
        for name, val in (d.get("heads") or {}).items():
            solved = [s for s in (val.get("solved") or []) if is_num(s)] if isinstance(val, dict) else []
            if solved:
                mean = val["mean"] if is_num(val.get("mean")) else sum(solved) / len(solved)
                out[name] = dict(solved=solved, mean=float(mean), total=total)
        return out

    def deepswe_for(self, *names):
        for n in names:
            if n in self.deepswe:
                return self.deepswe[n]
        return None

    # -- held-out choice tests ---------------------------------------------------------------
    def _load_heldout(self):
        """heldout.json -> dict(rows={source: n}, dropped={source: n}, heads={head: {source: top-1 %}})."""
        d = load_json(self.res / "heldout.json")
        if not isinstance(d, dict):
            return None
        heads = {h: {s: float(v) for s, v in vals.items() if is_num(v)}
                 for h, vals in (d.get("heads") or {}).items() if isinstance(vals, dict)}
        rows = {s: int(n) for s, n in (d.get("rows") or {}).items() if is_num(n)}
        if not heads or not rows:
            warn("heldout.json: needs \"rows\" and \"heads\"; held-out results skipped")
            return None
        return dict(rows=rows, dropped={s: int(n) for s, n in (d.get("dropped") or {}).items() if is_num(n)},
                    heads=heads)

    def heldout_sources(self, *heads) -> list:
        """[(key, name, hero label, group)] for every source all the given heads were scored on."""
        if not self.heldout:
            return []
        have = [self.heldout["heads"].get(h) for h in heads]
        if any(v is None for v in have):
            return []
        ok = [s for s in self.heldout["rows"] if all(s in v for v in have)]
        known = [t for t in HELDOUT_SOURCES if t[0] in ok]
        seen = {t[0] for t in known}
        return known + [(s, s.upper(), s.upper(), "Other") for s in ok if s not in seen]

    def chance(self, key: str):
        """Chance level of a choice test in %: the mean of 1 / number of options. None if not recorded.

        key is a held-out source ("bfcl") or an in-distribution metric ("choice_glaive_fc").
        """
        ck = self.checks if isinstance(self.checks, dict) else {}
        if key.startswith("choice_"):
            d = ((ck.get("choice") or {}).get("test") or {}).get(key[len("choice_"):])
        else:
            d = (ck.get("heldout") or {}).get(key)
            if d is None and isinstance(self.paired, dict):
                o = (self.paired.get("options_per_source") or {}).get(key) or {}
                return float(o["chance_pct"]) if is_num(o.get("chance_pct")) else None
        return float(d["chance"]) * 100 if isinstance(d, dict) and is_num(d.get("chance")) else None

    def paired_ci(self, key: str, new: str = OURS_RUN, base: str = CLM_TREX_RUN):
        """Paired bootstrap 95% interval, in points, of new minus base on one held-out source. None if not run."""
        if not isinstance(self.paired, dict):
            return None
        ci = (((self.paired.get("sources") or {}).get(key) or {}).get(f"{new}_vs_{base}") or {}).get("bootstrap95_points")
        return tuple(ci) if isinstance(ci, list) and len(ci) == 2 and all(is_num(v) for v in ci) else None

    def heldout_what(self, keys) -> str:
        """Says in words which held-out sources are new and which is only a held-out split."""
        key, name = MD_ONLY["heldout_split_only"]
        if key in keys:
            return f"{len(keys) - 1} sources we never trained on, plus {name}'s test split"
        return f"{len(keys)} sources we never trained on"

    def trex_latency_range(self):
        """(min, max) of the median decision latency over every shield-on T-Rex run, in ms."""
        runs = [m for h in self.trex.values() for k, m in h.items() if k == "shield"]
        runs += [m for h in self.trex_more.values() for m in h.get("shield", [])]
        vals = [m["latency_ms_p50_median"] for m in runs if is_num(m.get("latency_ms_p50_median"))]
        return (min(vals), max(vals)) if vals else None

    # -- blend points ----------------------------------------------------------------------
    def blend_points(self) -> list:
        """One dict per measured blend weight: alpha, head, trex (first shield run), more, held."""
        alphas = {0.0, 1.0} | set(self.wise)
        for name in list(self.trex) + list((self.heldout or {}).get("heads", {})):
            a = _alpha_of(name)
            if a is not None:
                alphas.add(a)
        pts = []
        for a in sorted(alphas):
            head = run_for_alpha(a)
            pts.append(dict(alpha=a, head=head, trex=self.trex_first(head), more=self.trex_extra(head),
                            held=(self.heldout or {}).get("heads", {}).get(head)))
        return [p for p in pts if p["trex"] or p["held"]]


def _pretty_metric(k: str) -> str:
    if k in CHOICE_LABELS:
        return CHOICE_LABELS[k]
    s = k[len("choice_"):] if k.startswith("choice_") else k
    s = s.replace("_top1", "").replace("_acc", "").replace("_", " ").strip()
    words = [w.upper() if (len(w) <= 4 and w.isalpha()) else w for w in s.split()]
    s = " ".join(words)
    return s[:1].upper() + s[1:] if s else k


# ---------------------------------------------------------------------------------------------
# Drawing: a thin canvas over one matplotlib axes, in pixel units with y pointing down
# ---------------------------------------------------------------------------------------------
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HAVE_INTER = "inter" in Path(findfont(FontProperties(family="Inter"))).name.lower()


@lru_cache(maxsize=None)
def font(weight: int = 400, mono: bool = False) -> FontProperties:
    if mono:
        return FontProperties(family="DejaVu Sans Mono", weight="bold" if weight >= 600 else "normal")
    if HAVE_INTER:
        return FontProperties(family="Inter", weight=weight)
    return FontProperties(family="DejaVu Sans", weight="bold" if weight >= 600 else "normal")


@lru_cache(maxsize=None)
def _x1(s: str, size: float, weight: int, mono: bool) -> float:
    return TextPath((0, 0), s, size=size, prop=font(weight, mono)).get_extents().x1


def text_w(s: str, size: float = 12, weight: int = 400, mono: bool = False) -> float:
    """Advance width of a string in px (keeps leading and trailing spaces)."""
    if not s:
        return 0.0
    return _x1(s + "|", size, weight, mono) - _x1("|", size, weight, mono)


def rrect(x, y, w, h, radii) -> MPath:
    """Rectangle with per-corner radii (top-left, top-right, bottom-right, bottom-left)."""
    tl, tr, br, bl = [max(0.0, min(r, abs(w) / 2, abs(h) / 2)) for r in radii]
    k = 0.5523
    M, L, C = MPath.MOVETO, MPath.LINETO, MPath.CURVE4
    v = [(x + tl, y)]
    c = [M]

    def corner(p1, p2, p3):
        v.extend([p1, p2, p3])
        c.extend([C, C, C])

    v.append((x + w - tr, y)); c.append(L)
    corner((x + w - tr + k * tr, y), (x + w, y + tr - k * tr), (x + w, y + tr))
    v.append((x + w, y + h - br)); c.append(L)
    corner((x + w, y + h - br + k * br), (x + w - br + k * br, y + h), (x + w - br, y + h))
    v.append((x + bl, y + h)); c.append(L)
    corner((x + bl - k * bl, y + h), (x, y + h - bl + k * bl), (x, y + h - bl))
    v.append((x, y + tl)); c.append(L)
    corner((x, y + tl - k * tl), (x + tl - k * tl, y), (x + tl, y))
    v.append((x + tl, y)); c.append(MPath.CLOSEPOLY)
    return MPath(v, c)


class Canvas:
    def __init__(self, w: float, h: float, theme: dict, name: str = ""):
        self.w, self.h, self.t, self.name = w, h, theme, name
        self.fig = plt.figure(figsize=(w / 72.0, h / 72.0), dpi=72)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")
        self.fig.patch.set_alpha(0)
        self.ax.patch.set_alpha(0)
        self._z = 10
        self.boxes = []          # text boxes, for the layout check and label placement

    def z(self) -> int:
        self._z += 1
        return self._z

    # -- text ------------------------------------------------------------------------------
    def box_of(self, x, y, s, size=12, weight=400, ha="left", va="baseline", mono=False):
        w = text_w(s, size, weight, mono)
        bx = x - w / 2 if ha == "center" else x - w if ha == "right" else x
        if va == "baseline":
            y0, y1 = y - size * 0.76, y + size * 0.22
        elif va == "center":
            y0, y1 = y - size * 0.40, y + size * 0.40
        else:  # top
            y0, y1 = y, y + size
        return (bx, y0, bx + w, y1)

    def text(self, x, y, s, size=12, weight=400, color=None, ha="left", va="baseline", mono=False,
             alpha=1.0, check=True):
        s = str(s)
        fp = font(weight, mono).copy()
        fp.set_size(size)
        mva = {"baseline": "baseline", "center": "center_baseline", "top": "top"}[va]
        self.ax.text(x, y, s, fontproperties=fp, color=color or self.t["ink"], ha=ha, va=mva,
                     alpha=alpha, zorder=self.z())
        b = self.box_of(x, y, s, size, weight, ha, va, mono)
        if check:
            self.boxes.append(b + (s,))
        return b[2] - b[0]

    def rich(self, x, y, parts, size=12, ha="left"):
        """One line made of (text, weight, colour) runs. Returns the total width."""
        total = sum(text_w(s, size, wt) for s, wt, _ in parts)
        cx = x - total if ha == "right" else x - total / 2 if ha == "center" else x
        for s, wt, col in parts:
            if s.strip():
                lead = text_w(s, size, wt) - text_w(s.lstrip(), size, wt)
                self.text(cx + lead, y, s.strip(), size=size, weight=wt, color=col)
            cx += text_w(s, size, wt)
        return total

    def fit(self, x, y, s, size, weight=600, color=None, max_w=None, min_size=11, **kw):
        """Text that shrinks until it fits max_w."""
        max_w = max_w or (self.w - 2 * 16)
        while size > min_size and text_w(s, size, weight) > max_w:
            size -= 0.5
        return self.text(x, y, s, size=size, weight=weight, color=color, **kw)

    def wrap(self, x, y, s, size=12, weight=400, color=None, max_w=None, lh=None):
        """Greedy word wrap. Returns the y of the next free baseline."""
        max_w = max_w or (self.w - 2 * 16)
        lh = lh or size * 1.42
        line = ""
        for word in s.split():
            trial = (line + " " + word).strip()
            if line and text_w(trial, size, weight) > max_w:
                self.text(x, y, line, size=size, weight=weight, color=color)
                y += lh
                line = word
            else:
                line = trial
        if line:
            self.text(x, y, line, size=size, weight=weight, color=color)
            y += lh
        return y

    # -- shapes ----------------------------------------------------------------------------
    def rect(self, x, y, w, h, color, r=0.0, radii=None, alpha=1.0, edge=None, lw=0.0):
        path = rrect(x, y, w, h, radii or (r, r, r, r))
        self.ax.add_patch(PathPatch(path, fc=color if color else "none", ec=edge or "none", lw=lw,
                                    alpha=alpha, zorder=self.z(), joinstyle="round"))

    def hbar(self, x0, yc, length, thick, color, alpha=1.0):
        """Horizontal bar: square at the baseline, 4 px rounded at the data end."""
        r = min(4.0, thick / 2, max(length, 0))
        self.rect(x0, yc - thick / 2, max(length, 0), thick, color, radii=(0, r, r, 0), alpha=alpha)

    def vbar(self, xc, y_base, height, width, color, alpha=1.0):
        r = min(4.0, width / 2, max(height, 0))
        self.rect(xc - width / 2, y_base - height, width, max(height, 0), color, radii=(r, r, 0, 0), alpha=alpha)

    def line(self, xs, ys, color, lw=2.0, alpha=1.0, cap="round"):
        self.ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle=cap,
                     solid_joinstyle="round", zorder=self.z())

    def hair(self, x0, y0, x1, y1, color=None, lw=1.0):
        self.line([x0, x1], [y0, y1], color or self.t["grid"], lw=lw, cap="butt")

    def vgrid(self, x, y0, y1, color, labels=(), z=None, pad=3.0):
        """A vertical grid line from y0 to y1 that breaks around every label box it would cross.

        Pass a z reserved with self.z() before the bars were drawn to keep the line under them.
        """
        cuts = sorted((b[1] - pad, b[3] + pad) for b in labels if b[0] - pad < x < b[2] + pad)
        y = y0
        for a, b in cuts + [(y1, y1)]:
            if min(a, y1) - y > 0.5:
                self.ax.plot([x, x], [y, min(a, y1)], color=color, lw=1.0, solid_capstyle="butt",
                             zorder=z if z is not None else self.z())
            y = max(y, b)

    def area(self, xs, ys, color, alpha=0.10):
        self.ax.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc=color, ec="none", alpha=alpha, zorder=self.z()))

    def dot(self, x, y, r, color, ring=2.0, alpha=1.0):
        if ring:
            self.ax.add_patch(Circle((x, y), r + ring, fc=self.t["surface"], ec="none", zorder=self.z()))
        self.ax.add_patch(Circle((x, y), r, fc=color, ec="none", alpha=alpha, zorder=self.z()))

    def ring(self, x, y, r, color, lw=2.0, fill=None, alpha=1.0):
        self.ax.add_patch(Circle((x, y), r, fc=fill or self.t["surface"], ec="none", zorder=self.z()))
        self.ax.add_patch(Circle((x, y), r, fc="none", ec=color, lw=lw, alpha=alpha, zorder=self.z()))

    def arrow(self, x0, y0, x1, y1, color=None, lw=1.5, head=6.0):
        """Straight arrow with a small filled head."""
        color = color or self.t["muted"]
        ang = math.atan2(y1 - y0, x1 - x0)
        bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
        self.line([x0, bx], [y0, by], color, lw=lw, cap="butt")
        px, py = -math.sin(ang) * head * 0.55, math.cos(ang) * head * 0.55
        self.ax.add_patch(Polygon([(x1, y1), (bx + px, by + py), (bx - px, by - py)], closed=True,
                                  fc=color, ec="none", zorder=self.z()))

    # -- shared furniture ------------------------------------------------------------------
    def header(self, title, subtitle=None, y=30):
        self.fit(16, y, title, size=18, weight=600)
        if subtitle:
            return self.wrap(16, y + 22, subtitle, size=12.5, color=self.t["ink2"], lh=17.5)
        return y + 22

    def footer(self, text=None, when=None):
        text = text or f"Source: results/innovations/, measured {when or MD_ONLY['measured_on']} {DOT} {REPO}"
        self.text(16, self.h - 12, text, size=10.5, color=self.t["muted"])

    # -- label placement -------------------------------------------------------------------
    def free(self, box, obstacles, pad=2.0) -> bool:
        x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        if x0 < 2 or y0 < 2 or x1 > self.w - 2 or y1 > self.h - 2:
            return False
        for o in obstacles:
            if x0 < o[2] and o[0] < x1 and y0 < o[3] and o[1] < y1:
                return False
        return True

    def place(self, x, y, s, obstacles, order="NE,SE,NW,SW,E,W,N,S", dist=7.0, size=11.5, weight=500,
              color=None, soft=()):
        """Put a short label next to (x, y) at the first compass spot that hits nothing.

        `soft` holds things a label should stay off if it can, such as grid lines.
        """
        spots = {
            "E": (x + dist + 2, y, "left", "center"), "W": (x - dist - 2, y, "right", "center"),
            "N": (x, y - dist - 3, "center", "baseline"), "S": (x, y + dist + size * 0.80, "center", "baseline"),
            "NE": (x + dist - 1, y - dist + 1, "left", "baseline"), "NW": (x - dist + 1, y - dist + 1, "right", "baseline"),
            "SE": (x + dist - 1, y + dist + size * 0.62, "left", "baseline"),
            "SW": (x - dist + 1, y + dist + size * 0.62, "right", "baseline"),
        }
        chosen = None
        for avoid_soft in ((True, False) if soft else (False,)):
            for key in order.split(","):
                tx, ty, ha, va = spots[key]
                b = self.box_of(tx, ty, s, size, weight, ha, va)
                if self.free(b, obstacles) and not (avoid_soft and not self.free(b, soft, pad=0.5)):
                    chosen = (tx, ty, ha, va, b)
                    break
            if chosen:
                break
        if chosen is None:
            tx, ty, ha, va = spots[order.split(",")[0]]
            chosen = (tx, ty, ha, va, self.box_of(tx, ty, s, size, weight, ha, va))
        tx, ty, ha, va, b = chosen
        self.text(tx, ty, s, size=size, weight=weight, color=color, ha=ha, va=va)
        obstacles.append(b)
        return b

    # -- output ----------------------------------------------------------------------------
    def check(self) -> None:
        """Warn about text that leaves the canvas or sits on other text."""
        bs = self.boxes
        for b in bs:
            if b[0] < 1 or b[2] > self.w - 1 or b[1] < 1 or b[3] > self.h - 1:
                warn(f"{self.name}: text leaves the canvas: {b[4]!r}")
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                if a[0] < b[2] - 1 and b[0] < a[2] - 1 and a[1] < b[3] - 1 and b[1] < a[3] - 1:
                    warn(f"{self.name}: text overlaps: {a[4]!r} and {b[4]!r}")

    def save_svg(self, path: Path, title: str, desc: str) -> None:
        self.check()
        self.fig.savefig(path, format="svg", transparent=True,
                         metadata={"Date": None, "Creator": "scripts/make_figures.py", "Title": title,
                                   "Description": desc})
        plt.close(self.fig)


def poly_boxes(xs, ys, step=3.0, half=2.5):
    """Small boxes along a polyline, used as obstacles for label placement."""
    out = []
    for i in range(len(xs) - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        n = max(1, int(max(abs(dx), abs(dy)) / step))
        for k in range(n + 1):
            px, py = xs[i] + dx * k / n, ys[i] + dy * k / n
            out.append((px - half, py - half, px + half, py + half))
    return out


def spread(ys, gap):
    """Push label positions apart so neighbours are at least `gap` px apart; keeps the order."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for _ in range(50):
        moved = False
        for a, b in zip(idx, idx[1:]):
            d = out[b] - out[a]
            if d < gap - 1e-6:
                push = (gap - d) / 2
                out[a] -= push
                out[b] += push
                moved = True
        if not moved:
            break
    return out


def wrap_lines(s: str, size: float = 12, weight: int = 400, max_w: float = W - 32) -> list:
    """Greedy word wrap, the same rule as Canvas.wrap, as a list of lines (to size a canvas first)."""
    lines, line = [], ""
    for word in s.split():
        trial = (line + " " + word).strip()
        if line and text_w(trial, size, weight) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


# ---------------------------------------------------------------------------------------------
# 1. hero: CLM-8B vs S1-Tweaks v0.1. Held-out tests first, then the agent benchmarks with the
#    shield-off caveat, then the in-distribution tests, drawn lighter.
# ---------------------------------------------------------------------------------------------
COLOUR_NAME = {1: "green", -1: "red", 0: "grey"}   # how --numbers reports the colour of a change


def _sign(d_points: float) -> int:
    """+1 or -1 for a change of FLAT_BELOW points or more, 0 for anything smaller."""
    return 0 if abs(d_points) < FLAT_BELOW else (1 if d_points > 0 else -1)


def _pct_row(label, a_txt, b_txt, chance=None):
    """A row from two printed percentages. The change is the difference of the numbers as printed."""
    d = round(float(b_txt) - float(a_txt), 1)
    return dict(label=label, clm=float(a_txt) / 100, ours=float(b_txt) / 100, clm_txt=a_txt, ours_txt=b_txt,
                delta=signed(d), sign=_sign(d), chance=chance)


def _count_row(label, a, b, total, unit, digits=0):
    """A row from two counts out of the same total (games survived, tasks solved)."""
    d = round(b - a, 1)
    txt = lambda v: f"{v:.{digits}f}/{total:g}"  # noqa: E731
    delta = f"0 {unit}s" if d == 0 else f"{'+' if d > 0 else MINUS}{abs(d):g} {plural(d, unit)}"
    return dict(label=label, clm=a / total, ours=b / total, clm_txt=txt(a), ours_txt=txt(b), delta=delta,
                sign=_sign(d / total * 100))


def hero_rows(D: Data):
    if not (D.clm and D.ours):
        raise Skip(f"needs {CLM_EVAL_RUN}.json and {OURS_RUN}.json")
    held, agent, indist = [], [], []
    for key, _, label, _ in D.heldout_sources(CLM_TREX_RUN, OURS_RUN):
        heads = D.heldout["heads"]
        held.append(_pct_row(label, f"{heads[CLM_TREX_RUN][key]:.1f}", f"{heads[OURS_RUN][key]:.1f}", D.chance(key)))
    tc, to = D.trex_pooled(CLM_TREX_RUN), D.trex_pooled(OURS_RUN)
    if tc and to and tc["seeds"] == to["seeds"]:
        agent.append(_count_row("T-Rex games survived, shield on", tc["survived"], to["survived"], tc["seeds"],
                                "game"))
    elif tc and to:
        warn("hero: the two heads played a different number of T-Rex games, row left out")
    dc, do = D.deepswe_for(CLM_TREX_RUN, CLM_EVAL_RUN), D.deepswe_for(OURS_RUN)
    if dc and do and dc["total"] == do["total"]:
        n = min(len(dc["solved"]), len(do["solved"]))
        label = f"DeepSWE after fine-tune, {n} seeds" if n > 1 else "DeepSWE after fine-tune"
        agent.append(_count_row(label, dc["mean"], do["mean"], dc["total"], "task", digits=1 if n > 1 else 0))
    for key in D.clm:
        if key.startswith("choice_") and is_num(D.clm[key]) and is_num(D.ours.get(key)):
            indist.append(_pct_row(CHOICE_LABELS.get(key, _pretty_metric(key)), pct(D.clm[key]), pct(D.ours[key]),
                                   D.chance(key)))
    indist.sort(key=lambda r: -(float(r["ours_txt"]) - float(r["clm_txt"])))
    for key, label in (("tricky_top1", "Tricky Q&A"), ("gen_top1", "General Q&A")):
        if is_num(D.clm.get(key)) and is_num(D.ours.get(key)):
            indist.append(_pct_row(label, pct(D.clm[key]), pct(D.ours[key])))
    return held, agent, indist


def hero_title(held, agent) -> str:
    """Built from the rows, so it cannot drift from them. It never calls T-Rex level."""
    parts = []
    if held:
        up, down = sum(r["sign"] > 0 for r in held), sum(r["sign"] < 0 for r in held)
        parts.append(f"up on {up} of {len(held)} held-out tests" + (f", down on {down}" if down else ""))
    trex = next((r for r in agent if r["label"].startswith("T-Rex")), None)
    if trex:
        parts.append(f"T-Rex {trex['ours_txt']} vs {trex['clm_txt']} with the shield on")
    if not parts:
        return f"{OURS_NAME} vs {CLM_NAME} on every test both were scored on"
    return f"{OURS_NAME} vs {CLM_NAME}: " + ", ".join(parts)


def fig_hero(T, D: Data):
    held, agent, indist = hero_rows(D)
    nc, no = D.trex_first(CLM_TREX_RUN, "noshield"), D.trex_first(OURS_RUN, "noshield")
    shield_off = bool(nc and no)
    groups = []
    if held:
        held_keys = [k for k, *_ in D.heldout_sources(CLM_TREX_RUN, OURS_RUN)]
        rows_total = sum(D.heldout["rows"][k] for k in held_keys)
        groups.append(("held", "Held-out tests",
                       f"{rows_total:,} rows, our builds: {D.heldout_what(held_keys)}", held))
    if agent or shield_off:
        groups.append(("agent", "Agent benchmarks", f"{CLM_NAME}'s own: T-Rex as shipped, DeepSWE after its fine-tune", agent))
    if indist:
        groups.append(("indist", "In-distribution tests",
                       "test rows share sources with our training data: an upper bound, drawn lighter", indist))

    # notes under the chart, known before the canvas so its height fits them
    has_chance = any(r.get("chance") is not None for r in held + indist)
    notes = ["Slice-scale runs, not a full retrain.",
             f"A change under {FLAT_BELOW:g} point is printed in grey."]
    if has_chance:
        notes.append("The tick on a row marks chance: a random pick among its options.")
    cis = [D.paired_ci(k) for k, *_ in D.heldout_sources(CLM_TREX_RUN, OURS_RUN)] if held else []
    if cis and all(cis):
        notes.append(f"Paired bootstrap on the held-out rows: {sum(lo > 0 for lo, _ in cis)} of {len(cis)} gains "
                     f"have a 95% interval above zero.")
    tc = D.trex_pooled(CLM_TREX_RUN)
    if tc and any(r["label"].startswith("T-Rex") for r in agent):
        s0 = MD_ONLY["trex_seeds_start_at"]
        notes.append(f"T-Rex with the shield on: {tc['seeds']:g} games per head on {max(tc['runs']):g} courses "
                     f"({', then '.join(f'seeds {s0:g} to {s0 + n - 1:g}' for n in tc['runs'])}).")
    dc, do = D.deepswe_for(CLM_TREX_RUN, CLM_EVAL_RUN), D.deepswe_for(OURS_RUN)
    if dc and do and len(dc["solved"]) > 1:
        rng = lambda d: f"{min(d['solved']):g} to {max(d['solved']):g}"  # noqa: E731
        same = rng(dc) == rng(do)
        notes.append(f"DeepSWE: {MD_ONLY['deepswe_setup']} on {dc['total']:g} tasks. {rng(dc)} solved per seed for "
                     + ("both heads." if same else f"{CLM_NAME} and {rng(do)} for {OURS_NAME}."))
    note_lines = wrap_lines(" ".join(notes), 11.5, 400, W - 32)

    ROW, GH, OFF_H = 26, 30, 54
    top = 122
    body = sum(GH + ROW * len(g[3]) for g in groups) + (OFF_H if shield_off else 0)
    axis_y = top + body + 2
    H = axis_y + 40 + 16 * len(note_lines) + 28
    c = Canvas(W, H, T, "hero")

    blend = (f"{round((1 - RELEASE_ALPHA) * 100):g}% {CLM_NAME}'s post-trained head + "
             f"{round(RELEASE_ALPHA * 100):g}% ours")
    c.header(hero_title(held, agent),
             f"Same frozen Qwen3-8B backbone. Only the small heads differ: {OURS_NAME} is a weight blend, "
             f"{blend}, with no extra training. Top-1 accuracy in % unless the row says otherwise.")

    x0, x1 = 236, 560
    X = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    col_clm, col_ours, col_d = 628, 706, 784

    # key for the two markers, and the column heads
    ly = top - 14
    c.ring(x0 + 7, ly - 4, 5.5, T["clm"], lw=2)
    w1 = c.text(x0 + 19, ly, CLM_NAME, size=11.5, weight=500, color=T["ink2"])
    c.dot(x0 + 19 + w1 + 22, ly - 4, 4.5, T["ours"], ring=0)
    w2 = c.text(x0 + 19 + w1 + 32, ly, OURS_NAME, size=11.5, weight=500, color=T["ink2"])
    if has_chance:
        kx = x0 + 19 + w1 + 32 + w2 + 22
        c.line([kx, kx], [ly - 10.5, ly + 2.5], T["ink2"], lw=1.5, cap="butt")
        c.text(kx + 8, ly, "chance", size=11.5, weight=500, color=T["ink2"])
    for cx, s in ((col_clm, CLM_NAME), (col_ours, "S1-Tweaks"), (col_d, "Change")):
        c.text(cx, ly, s, size=11.5, weight=600, color=T["ink2"], ha="right")

    def numbers(r, yb, faded=False):
        wt = 400 if faded else 600
        c.text(col_clm, yb, r["clm_txt"], size=12.5, color=T["ink2"], ha="right")
        c.text(col_ours, yb, r["ours_txt"], size=12.5, weight=wt, color=T["ink2"] if faded else T["ink"], ha="right")
        dcol = T["good_text"] if r["sign"] > 0 else T["bad_text"] if r["sign"] < 0 else T["ink2"]
        c.text(col_d, yb, r["delta"], size=12.5, weight=wt, color=dcol, ha="right")

    y = top
    shown = []
    for kind, gtitle, gnote, rows in groups:
        faded = kind == "indist"
        c.rect(12, y + 6, W - 24, 19, T["box"], r=4)
        c.rich(18, y + 20, [(gtitle, 600, T["ink"]), ("   " + gnote, 400, T["ink2"])], size=11)
        y += GH
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            if rows:
                c.hair(X(v), y - 2, X(v), y + ROW * len(rows) + 2, T["grid"])
        for r in rows:
            yc = y + ROW / 2
            c.text(16, yc + 4.5, r["label"], size=12.5, color=T["ink2"] if faded else T["ink"])
            a, b = X(r["clm"]), X(r["ours"])
            if r.get("chance") is not None:
                kx = X(r["chance"] / 100)
                c.line([kx, kx], [yc - 7, yc + 7], T["ink2"], lw=1.5, cap="butt", alpha=T["fade"] if faded else 1.0)
            if abs(b - a) > 1.5:
                link = T["ours_dim"] if b > a else T["bad"] if r["sign"] < 0 else T["axis"]
                c.line([a, b], [yc, yc], link, lw=3, cap="butt", alpha=T["fade"] if faded else 1.0)
            c.ring(a, yc, 6.5, T["clm"], lw=2, alpha=T["fade"] if faded else 1.0)
            c.dot(b, yc, 4.0, T["ours"], ring=1.5, alpha=T["fade"] if faded else 1.0)
            numbers(r, yc + 4.5, faded)
            shown.append(dict(group=gtitle, **{k: r[k] for k in ("label", "clm_txt", "ours_txt", "delta")},
                              colour=COLOUR_NAME[r["sign"]],
                              **({"chance": f"{r['chance']:.1f}"} if r.get("chance") is not None else {})))
            y += ROW
        if kind == "agent" and shield_off:
            # the caveat sits with the T-Rex claim: the same game without the safety shield
            c.text(16, y + 21, "T-Rex score, shield off", size=12.5)
            c.text(16, y + 36, f"mean best score over {nc['seeds']:g} games", size=10.5, color=T["muted"])
            scale = (x1 - x0) * 0.5 / max(nc["mean_best_score"], no["mean_best_score"])
            c.hair(x0, y + 5, x0, y + OFF_H - 5, T["axis"])
            for i, (name, run, col) in enumerate(((CLM_NAME, nc, T["clm"]), (OURS_NAME, no, T["ours"]))):
                yc = y + 17 + i * 20
                L = run["mean_best_score"] * scale
                c.hbar(x0, yc, L, 12, col)
                c.text(x0 + L + 8, yc + 4, f"{name}, {run['deaths']:g} deaths", size=11.5, color=T["ink2"])
            a_txt, b_txt = f"{nc['mean_best_score']:.0f}", f"{no['mean_best_score']:.0f}"
            d = int(b_txt) - int(a_txt)
            off = dict(clm_txt=a_txt, ours_txt=b_txt, delta=signed(d, 0), sign=(d > 0) - (d < 0))
            numbers(off, y + 31.5)
            shown.append(dict(group=gtitle, label="T-Rex score, shield off (mean best score)", clm_txt=a_txt,
                              ours_txt=b_txt, delta=off["delta"], colour=COLOUR_NAME[off["sign"]],
                              deaths=f"{nc['deaths']:g} vs {no['deaths']:g} over {nc['seeds']:g} games"))
            y += OFF_H
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.text(X(v), axis_y + 14, f"{v * 100:g}%", size=10.5, color=T["muted"], ha="center")
    yy = axis_y + 40
    for s in note_lines:
        c.text(16, yy, s, size=11.5, color=T["ink2"])
        yy += 16
    c.footer()
    return c, dict(title=hero_title(held, agent), rows=shown)


# ---------------------------------------------------------------------------------------------
# 2. heldout: three heads on the eight sources no head was trained on
# ---------------------------------------------------------------------------------------------
def fig_heldout(T, D: Data):
    heads = [(CLM_TREX_RUN, CLM_NAME, T["clm"]), (OURS_RUN, OURS_NAME, T["ours"]),
             (FULL_RUN, ROUTER_NAME, T["router"])]
    sources = D.heldout_sources(*(h for h, _, _ in heads))
    if not sources:
        raise Skip(f"needs heldout.json with {CLM_TREX_RUN}, {OURS_RUN} and {FULL_RUN}")
    acc = {h: D.heldout["heads"][h] for h, _, _ in heads}
    rows_n = D.heldout["rows"]
    groups = []
    for key, name, _, group in sources:
        if not groups or groups[-1][0] != group:
            groups.append((group, []))
        groups[-1][1].append((key, name))
    gain = lambda h: [round(acc[h][k] - acc[CLM_TREX_RUN][k], 1) for k, *_ in sources]  # noqa: E731
    g_ours, g_router = gain(OURS_RUN), gain(FULL_RUN)
    rng = lambda g: f"{min(g):.1f} to {max(g):.1f}"  # noqa: E731

    tc, to, tr = D.trex_pooled(CLM_TREX_RUN), D.trex_pooled(OURS_RUN), D.trex_pooled(FULL_RUN)
    dropped = sum(D.heldout["dropped"].get(k, 0) for k, *_ in sources)
    notes = []
    if dropped:
        notes.append(f"Rows whose question also appears in our training file were dropped ({dropped} in all).")
    notes.append("Slice-scale runs, not a full retrain.")
    note_lines = wrap_lines(" ".join(notes), 11.5, 400, W - 32)

    BAR, PITCH, GAP, BAND = 9, 12, 14, 28
    block = PITCH * len(heads)
    top = 128
    body = sum(BAND + len(srcs) * (block + GAP) for _, srcs in groups)
    axis_y = top + body
    box_h = 62 if (tc and to and tr) else 0
    H = axis_y + 30 + box_h + (14 if box_h else 0) + 16 * len(note_lines) + 36
    c = Canvas(W, H, T, "heldout")
    if all(g >= 0 for g in g_ours + g_router):
        title = (f"Held-out gains over {CLM_NAME}: {OURS_NAME.split()[0]} {rng(g_ours)} points, "
                 f"{ROUTER_NAME.split()[0]} {rng(g_router)}")
    else:
        title = f"{CLM_NAME}, {OURS_NAME} and {ROUTER_NAME} on tests no head was trained on"
    total = sum(rows_n[k] for k, *_ in sources)
    chances = {k: D.chance(k) for k, *_ in sources}
    n_opts = sorted({round(100 / v) for v in chances.values() if v})
    opts_txt = (f"{n_opts[0]} to {n_opts[-1]} options per question" if len(n_opts) > 1 else
                f"{n_opts[0]} options per question" if n_opts else "up to 10 options per question")
    c.header(title, f"{total:,} rows that we built as multiple-choice tests from public sets: "
                    f"{D.heldout_what([k for k, *_ in sources])}. Top-1 accuracy in %, {opts_txt}. These are not the "
                    f"benchmarks' own metrics. {ROUTER_NAME} is our fully post-trained head; {OURS_NAME} blends "
                    f"{round(RELEASE_ALPHA * 100):g}% of it into {CLM_NAME}'s head.")

    x0, x1 = 212, 668
    top_v = max(max(v[k] for k, *_ in sources) for v in acc.values())
    vmax = 20 * math.ceil(top_v / 20)
    X = lambda v: x0 + v / vmax * (x1 - x0)  # noqa: E731

    # legend
    lx, ly = x0, top - 14
    for _, name, col in heads:
        c.rect(lx, ly - 10, 12, 12, col, r=2.5)
        lx += 18 + c.text(lx + 18, ly, name, size=11.5, weight=500, color=T["ink2"]) + 22
    if any(v is not None for v in chances.values()):
        c.line([lx + 2, lx + 2], [ly - 10.5, ly + 2.5], T["ink"], lw=1.5, cap="butt")
        c.text(lx + 10, ly, "chance", size=11.5, weight=500, color=T["ink2"])
    c.text(W - 16, ly, f"value, change vs {CLM_NAME}", size=11, color=T["muted"], ha="right")

    y = top
    shown = {}
    grids, n0 = [], len(c.boxes)  # each group's grid is drawn last, so its lines can break around the value labels
    for group, srcs in groups:
        c.rect(12, y + 6, W - 24, 19, T["box"], r=4)
        c.text(18, y + 20, group, size=11, weight=600, color=T["ink"])
        y += BAND
        g_h = len(srcs) * (block + GAP) - GAP
        grids.append((c.z(), y + 2, y + g_h + 6))
        # name the kind under a source only where the group mixes kinds (intent and support routing)
        mixed = len({HELDOUT_KIND.get(key) for key, _ in srcs}) > 1
        for key, name in srcs:
            yb = y + 4
            c.text(16, yb + 14, name, size=12.5, weight=600)
            sub = f"{rows_n[key]:,} rows" + (f" {DOT} {HELDOUT_KIND[key]}" if mixed and key in HELDOUT_KIND else "")
            c.text(16, yb + 29, sub, size=10.5, color=T["muted"])
            base = acc[CLM_TREX_RUN][key]
            for i, (h, hname, col) in enumerate(heads):
                yc = yb + PITCH * i + PITCH / 2
                v = acc[h][key]
                c.hbar(x0, yc, X(v) - x0, BAR, col)
                is_clm = h == CLM_TREX_RUN
                parts = [(f"{v:.1f}", 400 if is_clm else 600, T["ink2"] if is_clm else T["ink"])]
                if not is_clm:
                    parts.append((f"  {signed(round(v - base, 1))}", 400, T["ink2"]))
                c.rich(X(v) + 7, yc + 3.8, parts, size=11)
                shown.setdefault(name, {"rows": rows_n[key]})[hname] = f"{v:.1f}"
            if chances.get(key) is not None:  # a random pick among this source's options
                c.line([X(chances[key]), X(chances[key])], [yb - 1, yb + block + 1], T["ink"], lw=1.5, cap="butt")
                shown[name]["chance"] = f"{chances[key]:.1f}"
            y += block + GAP
    labels = [b for b in c.boxes[n0:] if b[0] >= x0]
    for grid_z, g0, g1 in grids:
        for v in range(0, int(vmax) + 1, 20):
            c.vgrid(X(v), g0, g1, T["axis"] if v == 0 else T["grid"], labels, z=grid_z)
    for v in range(0, int(vmax) + 1, 20):
        c.text(X(v), axis_y + 12, f"{v}%", size=10.5, color=T["muted"], ha="center")

    # what the bigger gains cost: T-Rex
    y = axis_y + 30
    trex_shown = None
    if box_h:
        c.rect(16, y, W - 32, box_h, T["box"], r=6, edge=T["box_edge"], lw=1)
        c.rich(30, y + 23, [(f"What {ROUTER_NAME} gives up: T-Rex.", 600, T["ink"]),
                            ("  Games survived with the safety shield on:", 400, T["ink2"])], size=12)
        tx = 30
        trex_shown = {}
        for (h, name, col), run in zip(heads, (tc, to, tr)):
            c.rect(tx, y + 35, 12, 12, col, r=2.5)
            txt = f"{run['survived']:g} of {run['seeds']:g}"
            tx += 18 + c.rich(tx + 18, y + 45, [(name + "  ", 400, T["ink2"]), (txt, 600, T["ink"])], size=12) + 30
            trex_shown[name] = txt
        y += box_h + 14
    yy = y + 12
    for s in note_lines:
        c.text(16, yy, s, size=11.5, color=T["ink2"])
        yy += 16
    c.footer()
    return c, dict(title=title, top1=shown, trex_shield_on=trex_shown,
                   gain_range={OURS_NAME: rng(g_ours), ROUTER_NAME: rng(g_router)})


# ---------------------------------------------------------------------------------------------
# 3. blend-tradeoff: why the blend stops at 15%
# ---------------------------------------------------------------------------------------------
AXIS_BREAK = 0.3        # blend weights below this get most of the x axis: five of seven blends sit there
AXIS_LEFT_SHARE = 0.72


def fig_blend(T, D: Data):
    pts = D.blend_points()
    tp = [p for p in pts if p["trex"]]
    series = [(k, nm, kind) for k, nm, kind in BLEND_SERIES if all(p["held"] and k in p["held"] for p in pts)]
    if len(tp) < 2 or not series:
        raise Skip("needs trex.json and heldout.json for at least two blend weights")
    is_rel = lambda p: abs(p["alpha"] - RELEASE_ALPHA) < 1e-9  # noqa: E731
    rel = next((p for p in pts if is_rel(p)), None)
    H = 612
    c = Canvas(W, H, T, "blend-tradeoff")
    # this title holds only while every blend above the released one lost at least one game
    above = [p for p in tp if p["alpha"] > RELEASE_ALPHA]
    n_games = sorted({p["trex"]["seeds"] for p in tp})
    if rel and above and all(p["trex"]["survived"] < p["trex"]["seeds"] for p in above):
        title = "Held-out accuracy rises with every step of the blend. T-Rex survival does not"
    else:
        title = "Blend weight against T-Rex survival and held-out accuracy"
    sub_txt = (f"Weight blends of {CLM_NAME}'s post-trained head (0) and our fully post-trained head "
               f"(1, released as {ROUTER_NAME}). Each point is a measured blend; none had extra training.")
    if len(n_games) == 1:
        sub_txt += f" T-Rex is {n_games[0]:g} games per blend, so one game either way is noise."
    c.header(title, sub_txt)

    x0, x1 = 60, 616
    alphas = [p["alpha"] for p in pts]
    broken = sum(a < AXIS_BREAK for a in alphas) >= 3 and any(a > AXIS_BREAK for a in alphas)
    wl = (x1 - x0) * AXIS_LEFT_SHARE if broken else 0.0
    xb = x0 + wl

    def X(a):
        if not broken:
            return x0 + a * (x1 - x0)
        if a <= AXIS_BREAK:
            return x0 + a / AXIS_BREAK * wl
        return xb + (a - AXIS_BREAK) / (1 - AXIS_BREAK) * (x1 - xb)

    PIP_R, PIP_PITCH = 4.5, 13
    n_max = max(p["trex"]["seeds"] for p in tp)
    ty_bot = 236
    ty_top = ty_bot - PIP_PITCH * n_max
    by_top, by_bot = 312, 502
    head_col = lambda p: T["clm"] if p["alpha"] == 0 else T["router"] if p["alpha"] == 1 else T["ours"]  # noqa: E731

    # ---- frame: baselines, the axis break, the released blend -----------------------------------
    top_v = max(p["held"][k] for p in pts for k, _, _ in series)
    vmax = 20 * math.ceil(top_v / 20)
    BY = lambda v: by_bot - v / vmax * (by_bot - by_top)  # noqa: E731

    def baseline(xa, xz, yb):
        """A baseline. Where the axis is stretched it has a gap with two slanted strokes: an axis break."""
        if not broken:
            return c.hair(xa, yb, xz, yb, T["axis"])
        c.hair(xa, yb, xb - 5, yb, T["axis"])
        c.hair(xb + 5, yb, xz, yb, T["axis"])
        for dx in (-4, 2):
            c.line([xb + dx - 1.5, xb + dx + 3.5], [yb + 5, yb - 5], T["muted"], lw=1.2, cap="butt")

    grid_lines = [(x0, BY(v), x1, BY(v)) for v in range(20, int(vmax) + 1, 20)]
    for g in grid_lines:
        c.hair(g[0], g[1], g[2], g[3], T["grid"])
    for v in range(0, int(vmax) + 1, 20):
        c.text(x0 - 10, BY(v) + 4, f"{v}%", size=10.5, color=T["muted"], ha="right")
    baseline(x0, x1, by_bot)
    baseline(x0 - 8, x1 + 8, ty_bot)
    xc = X(RELEASE_ALPHA)
    if rel:
        c.hair(xc, ty_bot, xc, by_bot, T["ink2"], lw=1)

    # ---- T-Rex: one dot per game ----------------------------------------------------------------
    seeds = sorted({p["trex"]["seeds"] for p in tp})
    h1 = (f"T-Rex games survived out of {seeds[0]:g}, safety shield on" if len(seeds) == 1
          else "T-Rex games survived, safety shield on")
    c.rich(x0 - 8, ty_top - 52, [(h1, 600, T["ink"]), ("   one dot per game, filled if survived", 400, T["ink2"])],
           size=12.5)
    trex_shown, more_shown = {}, {}
    for p in tp:
        x, run = X(p["alpha"]), p["trex"]
        for k in range(int(run["seeds"])):
            yk = ty_bot - PIP_PITCH * k - PIP_PITCH / 2
            if k < run["survived"]:
                c.dot(x, yk, PIP_R, head_col(p), ring=1.5)
            else:
                c.ring(x, yk, PIP_R - 0.6, T["muted"], lw=1.2)
        lab = f"{run['survived']:g}/{run['seeds']:g}"
        c.text(x, ty_top - 7, lab, size=11.5, weight=700 if is_rel(p) else 600, ha="center")
        trex_shown[alpha_txt(p["alpha"])] = lab
        for m in p["more"]:  # a later, longer run of the same head: it replays the first run's seeds
            s0 = MD_ONLY["trex_seeds_start_at"]
            s = f"seeds {s0:g} to {s0 + m['seeds'] - 1:g}: {m['survived']:g}/{m['seeds']:g}"
            w = text_w(s, 11, 500)
            tx = min(max(x - w / 2, 16), W - 16 - w)
            c.text(tx, ty_top - 26, s, size=11, weight=500, color=T["ink2"])
            more_shown[alpha_txt(p["alpha"])] = f"{m['survived']:g}/{m['seeds']:g}"

    # ---- held-out accuracy ----------------------------------------------------------------------
    c.text(x0 - 8, by_top - 24, "Held-out accuracy, top-1", size=12.5, weight=600)
    exs = [X(p["alpha"]) for p in pts]

    def path(k):
        """The line through the measured points, bent where it crosses the axis break."""
        xs, ys = [], []
        for i, p in enumerate(pts):
            if broken and i and pts[i - 1]["alpha"] < AXIS_BREAK < p["alpha"]:
                q = pts[i - 1]
                f = (AXIS_BREAK - q["alpha"]) / (p["alpha"] - q["alpha"])
                xs.append(xb)
                ys.append(BY(q["held"][k] + f * (p["held"][k] - q["held"][k])))
            xs.append(exs[i])
            ys.append(BY(p["held"][k]))
        return xs, ys

    obstacles = []
    paths = {k: path(k) for k, _, _ in series}
    for k, _, _ in series:
        obstacles += poly_boxes(*paths[k])
    if rel:
        obstacles += poly_boxes([xc, xc], [by_top, by_bot], half=1.5)
    for k, _, _ in series:
        c.line(*paths[k], T["ours"], lw=2)
    for k, _, _ in series:
        for p, x in zip(pts, exs):
            yv = BY(p["held"][k])
            if p["alpha"] == 0:
                c.ring(x, yv, 5.0, T["clm"], lw=2)
            elif is_rel(p) or p["alpha"] == 1:
                c.dot(x, yv, 5.5, head_col(p), ring=2)
            else:
                c.dot(x, yv, 3.5, T["ours"], ring=1.5)
    held_shown = {nm: {alpha_txt(p["alpha"]): f"{p['held'][k]:.1f}" for p in pts} for k, nm, _ in series}
    # values at the left end (CLM) and at the released blend
    for k, _, _ in series:
        c.place(exs[0], BY(pts[0]["held"][k]), f"{pts[0]['held'][k]:.1f}", obstacles, order="SE,NE,E,N", dist=9,
                size=11, weight=500, color=T["ink2"], soft=grid_lines)
    if rel:
        j = pts.index(rel)
        for k, _, _ in series:
            c.place(exs[j], BY(rel["held"][k]), f"{rel['held'][k]:.1f}", obstacles, order="NW,SE,NE,SW", dist=8,
                    size=11.5, weight=700, soft=grid_lines)
    # names and end values in the right margin
    ends = spread([BY(pts[-1]["held"][k]) for k, _, _ in series], 34)
    for (k, nm, kind), ye in zip(series, ends):
        c.rich(x1 + 14, ye + 2, [(f"{pts[-1]['held'][k]:.1f}", 700, T["ink"]), ("  " + nm, 600, T["ink"])], size=11.5)
        c.text(x1 + 14, ye + 16, f"{kind}, {D.heldout['rows'][k]:,} rows", size=10.5, color=T["muted"])

    # ---- x axis ---------------------------------------------------------------------------------
    for p in pts:
        x = X(p["alpha"])
        c.hair(x, by_bot, x, by_bot + 4, T["axis"])
        strong = is_rel(p)
        c.text(x, by_bot + 19, alpha_txt(p["alpha"]), size=11.5 if strong else 11, weight=700 if strong else 400,
               color=T["ink"] if strong else T["muted"], ha="center")
    names = [(0.0, CLM_NAME, 500, T["ink2"]), (1.0, ROUTER_NAME, 500, T["ink2"])]
    if rel:
        names.append((RELEASE_ALPHA, OURS_NAME, 700, T["ink"]))
    for a, nm, wt, col in names:
        if any(abs(p["alpha"] - a) < 1e-9 for p in pts):
            c.text(X(a), by_bot + 36, nm, size=11.5, weight=wt, color=col, ha="center")
    cap = "share of our head in the blend"
    if broken:
        n_low = sum(a < AXIS_BREAK for a in alphas)
        cap += f" {DOT} the axis is stretched below {AXIS_BREAK:g}, where {n_low} of the {len(alphas)} blends sit"
    c.text((x0 + x1) / 2, by_bot + 58, cap, size=11.5, color=T["muted"], ha="center")
    c.footer()
    return c, dict(title=title, trex_survived=trex_shown, trex_second_run=more_shown, heldout_top1=held_shown,
                   released=alpha_txt(RELEASE_ALPHA) if rel else None)


# ---------------------------------------------------------------------------------------------
# 4. quantization
# ---------------------------------------------------------------------------------------------
def fig_quant(T, D: Data):
    lat, qh = D.latency, D.quant_heads
    if not (isinstance(lat, dict) and isinstance(qh, dict)):
        raise Skip("needs backbone_latency_l40s.json and quant_heads_mid.json")
    variants = [(k, LATENCY_LABELS.get(k, (k, None))) for k in lat if isinstance(lat[k], dict)]
    lengths = sorted({int(t) for k, _ in variants for t in lat[k]})
    acc = {k: load_json(D.res / f) if f else None for k, (_, f) in variants}
    base_key = variants[0][0]
    ref = acc.get(base_key)
    at = str(256 if 256 in lengths else lengths[len(lengths) // 2])   # the prompt length quoted in the title
    name = lambda k: LATENCY_LABELS.get(k, (k,))[0]  # noqa: E731
    has = lambda a, m: bool(a) and is_num(a.get(m))  # noqa: E731

    def cost(k):
        """Tricky Q&A points lost against the unquantized backbone, or None if not measured."""
        if k == base_key or not (has(acc.get(k), "tricky_top1") and has(ref, "tricky_top1")):
            return None
        return -points_between(ref["tricky_top1"], acc[k]["tricky_top1"])

    costs = {k: cost(k) for k, _ in variants}
    cheap = [k for k in costs if costs[k] is not None and costs[k] < FLAT_BELOW]
    dear = [k for k in costs if costs[k] is not None and costs[k] >= FLAT_BELOW]
    fp32 = qh.get("fp32", {}).get("tricky_top1")
    broke = [k for k, v in qh.items() if fp32 and is_num(v.get("tricky_top1")) and (fp32 - v["tricky_top1"]) * 100 >= 5]

    rows = [(k, lab, grp) for k, lab, grp in QUANT_HEAD_ROWS if k in qh]
    rows += [(k, k, "Other") for k in qh if k not in {r[0] for r in rows}]
    n_groups = len({g for _, _, g in rows})
    body_h = 27 * len(rows) + 30 * n_groups
    H = 400 + body_h + 62
    c = Canvas(W, H, T, "quantization")
    if cheap:
        best = min(cheap, key=lambda k: lat[k][at])
        title = (f"{name(best)} backbone: {up1(lat[best][at])} ms, not {up1(lat[base_key][at])} ms, at {at} tokens, "
                 f"for {costs[best]:.1f} points." + "".join(f" {name(k)} costs {costs[k]:.1f}." for k in dear))
    else:
        fast = min((k for k, _ in variants), key=lambda k: lat[k][at])
        title = f"{name(fast)} backbone: {up1(lat[fast][at])} ms, not {up1(lat[base_key][at])} ms, at {at} tokens"
    c.header(title, "Top: time for one backbone pass at three prompt lengths, and what each quantized backbone costs in "
                    "tricky Q&A accuracy. Bottom: the same for the small head and the cached option vectors."
                    + (" Quantized activations break the head." if "int8-dyn" in broke else ""))

    # ---- A: backbone latency ------------------------------------------------------------------
    px0, px1, py_top, py_bot = 60, 440, 152, 304
    c.rich(16, py_top - 30, [("Backbone latency per state, ms", 600, T["ink"]),
                             (f"   {MD_ONLY['latency_setup']}", 400, T["ink2"])], size=12.5)
    top_ms = max(lat[k][str(t)] for k, _ in variants for t in lengths if str(t) in lat[k])
    ymax = 5 * (int(top_ms / 5) + 1)
    Y = lambda v: py_bot - v / ymax * (py_bot - py_top)  # noqa: E731
    for v in range(0, int(ymax) + 1, 5):
        c.hair(px0, Y(v), px1, Y(v), T["axis"] if v == 0 else T["grid"])
        c.text(px0 - 8, Y(v) + 4, f"{v}", size=10.5, color=T["muted"], ha="right")
    cols = [T["clm"], T["soft"], T["ours"]] + [T["ours"]] * 5
    bw, gap = 28, 4
    gw = len(variants) * bw + (len(variants) - 1) * gap
    lat_shown = {}
    for gi, t in enumerate(lengths):
        gx = px0 + (px1 - px0) * (gi + 0.5) / len(lengths)
        for vi, (k, (lab, _)) in enumerate(variants):
            if str(t) not in lat[k]:
                continue
            v = lat[k][str(t)]
            bx = gx - gw / 2 + vi * (bw + gap) + bw / 2
            c.vbar(bx, py_bot, py_bot - Y(v), bw, cols[vi])
            c.text(bx, Y(v) - 5, up1(v), size=10.5, color=T["ink2"], ha="center")
            lat_shown.setdefault(lab, {})[str(t)] = up1(v)
        c.text(gx, py_bot + 17, f"{t} tokens", size=11, color=T["ink2"], ha="center")

    # ---- B: what each backbone costs in accuracy; the bars double as the key for A --------------
    rx = 484
    c.text(rx, py_top - 30, "Tricky Q&A top-1 with each backbone, %", size=12.5, weight=600)
    bar_x, bar_w = rx + 20, 150
    acc_shown = {}
    yv = py_top + 6
    for vi, (k, (lab, _)) in enumerate(variants):
        a = acc.get(k)
        c.rect(rx, yv - 10, 12, 12, cols[vi], r=2.5)
        c.rich(rx + 20, yv, [(lab, 600, T["ink"])] + ([(f"  {CLM_NAME} default", 400, T["ink2"])] if k == base_key else []),
               size=12.5)
        if has(a, "tricky_top1"):
            L = a["tricky_top1"] * bar_w
            c.hbar(bar_x, yv + 17, L, 9, cols[vi])
            parts = [(pct(a["tricky_top1"]), 600, T["ink"])]
            if costs[k] is not None:
                big = costs[k] >= FLAT_BELOW
                parts.append((f"   {signed(-costs[k])} points", 600 if big else 400, T["bad_text"] if big else T["ink2"]))
            elif k == base_key:
                parts.append(("   reference", 400, T["ink2"]))
            c.rich(bar_x + L + 8, yv + 21, parts, size=12)
            acc_shown[lab] = dict(tricky=pct(a["tricky_top1"]), change=signed(-costs[k]) if costs[k] is not None else None,
                                  general=pct(a["gen_top1"]) if has(a, "gen_top1") else None)
        else:
            c.text(bar_x, yv + 21, "accuracy not measured", size=11.5, color=T["muted"])
            acc_shown[lab] = None
        yv += 48
    c.hair(bar_x, py_top + 12, bar_x, yv - 48 + 26, T["axis"])
    if ref:
        gen = [pct(acc[k]["gen_top1"]) for k, _ in variants if has(acc.get(k), "gen_top1")]
        note = f"{MD_ONLY['quant_head_name']}, unchanged, scored on vectors from each backbone."
        if len(gen) == len(variants):
            note += f" General Q&A: {', '.join(gen[:-1])} and {gen[-1]}."
        c.wrap(rx, py_bot + 3, note, size=10.5, color=T["muted"], max_w=W - 16 - rx, lh=14)

    # ---- C: head and action cache -------------------------------------------------------------
    y = py_bot + 70
    c.text(16, y, f"{MD_ONLY['quant_head_name']} and its action cache: tricky Q&A top-1 after quantization",
           size=12.5, weight=600)
    bx0, bx1 = 222, 478
    BX = lambda v: bx0 + v * (bx1 - bx0)  # noqa: E731
    col_head, col_cache = 652, 784
    y += 24
    c.text(col_head, y, "Head size", size=11.5, weight=600, color=T["ink2"], ha="right")
    c.text(col_cache, y, "Cache, 10k actions", size=11.5, weight=600, color=T["ink2"], ha="right")
    y += 4
    head_shown = {}
    last_group = None
    base = qh.get("fp32", {})
    grids, n0 = [], len(c.boxes)  # each group's grid is drawn last, so its lines can break around the value labels
    for k, lab, grp in rows:
        if grp != last_group:
            c.rect(12, y + 6, W - 24, 19, T["box"], r=4)
            c.text(18, y + 20, grp, size=11, weight=600, color=T["ink2"])
            y += 30
            n_in = sum(1 for r in rows if r[2] == grp)
            grids.append((c.z(), y - 2, y + 27 * n_in + 2))
            last_group = grp
        v = qh[k]
        yc = y + 13.5
        tr = v.get("tricky_top1")
        is_base, is_bad = k == "fp32", k in broke
        c.text(16, yc + 4.5, lab, size=12.5)
        if is_num(tr):
            c.hbar(bx0, yc, BX(tr) - bx0, 14, T["clm"] if is_base else T["bad"] if is_bad else T["ours"])
            parts = [(pct_up(tr), 600, T["ink"])]
            if not is_base and fp32:
                d = round(float(pct_up(tr)) - float(pct_up(fp32)), 1)
                if is_bad:
                    parts.append((f"   {signed(d)} points" + (": breaks the head" if k == "int8-dyn" else ""),
                                  600, T["bad_text"]))
                elif d != 0:
                    parts.append((f"   {signed(d)}", 400, T["ink2"]))
            c.rich(BX(tr) + 8, yc + 4.5, parts, size=12)
        for cx, key in ((col_head, "head_MB"), (col_cache, "cache_MB_10000_actions")):
            if is_num(v.get(key)):
                smaller = is_num(base.get(key)) and v[key] < base[key] - 1e-9
                c.text(cx, yc + 4.5, f"{up1(v[key])} MB", size=12, weight=600 if smaller else 400,
                       color=T["ink"] if smaller else T["muted"], ha="right")
        head_shown[lab] = dict(tricky=pct_up(tr) if is_num(tr) else None,
                               head_MB=up1(v["head_MB"]) if is_num(v.get("head_MB")) else None,
                               cache_MB=up1(v["cache_MB_10000_actions"]) if is_num(v.get("cache_MB_10000_actions")) else None)
        y += 27
    labels = [b for b in c.boxes[n0:] if b[0] >= bx0]
    for grid_z, g0, g1 in grids:
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            c.vgrid(BX(v), g0, g1, T["axis"] if v == 0 else T["grid"], labels, z=grid_z)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.text(BX(v), y + 18, f"{v * 100:g}%", size=10.5, color=T["muted"], ha="center")
    c.footer()
    return c, dict(title=title, latency_ms=lat_shown, backbone_accuracy=acc_shown, heads=head_shown)


# ---------------------------------------------------------------------------------------------
# 5. head-layouts: the CPU sweep
# ---------------------------------------------------------------------------------------------
def _sweep_acc(d, key):
    """Accuracy of a sweep run; Matryoshka runs report one value per read-out size, use the full one."""
    if is_num(d.get(key)):
        return d[key]
    dims = d.get("cfg", {}).get("matryoshka") or []
    if dims and is_num(d.get(f"{key}@{max(dims)}")):
        return d[f"{key}@{max(dims)}"]
    return None


def _nice_range(lo, hi, pad=1.2):
    """Axis limits and tick step for a dot plot: padded, on round numbers, at most 6 intervals."""
    for step in (1, 2, 5, 10, 20, 50):
        ylo, yhi = math.floor((lo - pad) / step) * step, math.ceil((hi + pad) / step) * step
        if (yhi - ylo) / step <= 6:
            break
    return ylo, yhi, step


def fig_layouts(T, D: Data):
    sw = {k: d for k, d in D.sweep.items() if _sweep_acc(d, "gen_top1") is not None and is_num(d.get("params"))}
    base, shared = sw.get("base_w1536_p512"), sw.get("shared_p512")
    mrl = next((d for d in sw.values() if d.get("cfg", {}).get("matryoshka")), None)
    if not (base and shared):
        raise Skip("needs cpu_sweep/base_w1536_p512.json and cpu_sweep/shared_p512.json")
    H = 614
    c = Canvas(W, H, T, "head-layouts")
    acc = lambda d, key: float(pct_up(_sweep_acc(d, key)))  # noqa: E731
    dg = acc(shared, "gen_top1") - acc(base, "gen_top1")
    ratio = shared["params"] / base["params"]
    share_txt = "half the parameters" if 0.45 <= ratio <= 0.55 else f"{ratio * 100:.0f}% of the parameters"
    steps = base.get("step")
    c.header(f"After {steps:,} steps, one shared head is {abs(dg):.1f} points {'ahead' if dg >= 0 else 'behind'} "
             f"with {share_txt}",
             f"CPU sweep, every layout trained from scratch on the same budget ({MD_ONLY['cpu_sweep_pairs']} pairs, "
             f"{steps:,} steps, one seed). Early training: this ranks how fast a layout learns, not where it ends up.")

    # ---- left: accuracy against parameter count, two small scatters ----------------------------
    px0, px1 = 56, 440
    pmax = 24.0
    PX = lambda p: px0 + (p / 1e6) / pmax * (px1 - px0)  # noqa: E731
    panels = [("gen_top1", "General Q&A, top-1 %", 152, 318), ("tricky_top1", "Tricky Q&A, top-1 %", 376, 496)]
    lg_y = 104
    c.dot(px0 + 5, lg_y - 4, 4.5, T["ours"], ring=0)
    w1 = c.text(px0 + 15, lg_y, "one shared head", size=11.5, weight=500, color=T["ink2"])
    c.dot(px0 + 15 + w1 + 20, lg_y - 4, 4.5, T["clm"], ring=0)
    w2 = c.text(px0 + 15 + w1 + 30, lg_y, "separate state and action heads", size=11.5, weight=500, color=T["ink2"])
    lx = px0 + 15 + w1 + 30 + w2 + 26
    c.ring(lx, lg_y - 4, 7, T["clm"], lw=1.5)
    c.dot(lx, lg_y - 4, 3.5, T["clm"], ring=0)
    c.text(lx + 13, lg_y, f"{CLM_NAME}'s layout", size=11.5, weight=500, color=T["ink2"])
    shown = {}
    prefer = {"shared_p512": "W,NW,N", "shared_adapter_p512": "E,SE,NE", "base_w1536_p512": "S,SW,SE,W",
              "mrl_p1024": "N,E,NE,NW", "w768_p512": "S,SW,W,N", "depth2_p512": "S,W,SW,N", "p128": "NW,W,N,SW"}
    for pi, (key, name, y_top, y_bot) in enumerate(panels):
        vals = {k: acc(d, key) for k, d in sw.items() if _sweep_acc(d, key) is not None}
        ylo, yhi, step = _nice_range(min(vals.values()), max(vals.values()))
        PY = lambda v: y_bot - (v - ylo) / (yhi - ylo) * (y_bot - y_top)  # noqa: E731
        c.text(px0, y_top - 14, name, size=12, weight=600)
        v, grid_lines = ylo, []
        while v <= yhi + 1e-9:
            c.hair(px0, PY(v), px1, PY(v), T["axis"] if v == ylo else T["grid"])
            grid_lines.append((px0, PY(v), px1, PY(v)))
            c.text(px0 - 8, PY(v) + 4, f"{v:g}", size=10.5, color=T["muted"], ha="right")
            v += step
        for p in range(0, int(pmax) + 1, 5):
            c.hair(PX(p * 1e6), y_bot, PX(p * 1e6), y_bot + 4, T["axis"])
            if pi == len(panels) - 1:
                c.text(PX(p * 1e6), y_bot + 18, f"{p}M", size=10.5, color=T["muted"], ha="center")
        pts = {k: (PX(sw[k]["params"]), PY(vals[k])) for k in vals}
        obstacles = [(x - 6, y - 6, x + 6, y + 6) for x, y in pts.values()]
        # an elbow from the baseline to the shared head spells out the two gains
        if pi == 0 and "base_w1536_p512" in pts and "shared_p512" in pts:
            (bx, by), (sx, sy) = pts["base_w1536_p512"], pts["shared_p512"]
            c.line([bx, bx, sx + 10], [by - 11, sy, sy], T["axis"], lw=1.25, cap="butt")
            obstacles += poly_boxes([bx, bx, sx], [by, sy, sy], half=3)
            b1 = c.box_of(bx + 8, (by + sy) / 2 + 4, f"{signed(dg)} points", 11.5, 600)
            c.text(bx + 8, (by + sy) / 2 + 4, f"{signed(dg)} points", size=11.5, weight=600)
            b2 = c.box_of((bx + sx) / 2 + 20, sy - 8, share_txt, 11.5, 600, "center")
            c.text((bx + sx) / 2 + 20, sy - 8, share_txt, size=11.5, weight=600, ha="center")
            obstacles += [b1, b2]
        for k, (x, y) in pts.items():
            is_shared = str(sw[k].get("cfg", {}).get("tower", "")).startswith("shared")
            if k == "base_w1536_p512":
                c.ring(x, y, 8, T["clm"], lw=1.5)
            c.dot(x, y, 4.5, T["ours"] if is_shared else T["clm"], ring=2)
        for k in sorted(pts, key=lambda k: (k not in ("shared_p512", "base_w1536_p512"), k)):
            x, y = pts[k]
            strong = k in ("shared_p512", "base_w1536_p512")
            c.place(x, y, f"{LAYOUT_LABELS.get(k, k)} {vals[k]:.1f}", obstacles,
                    order=prefer.get(k, "E,NE,SE,W,N,S") + ",E,W,N,S,NE,NW,SE,SW", dist=9 if k == "base_w1536_p512" else 8,
                    size=11, weight=600 if strong else 400, color=T["ink"] if strong else T["ink2"], soft=grid_lines)
        shown[name] = {LAYOUT_LABELS.get(k, k): dict(params_M=up1(sw[k]["params"] / 1e6), acc=f"{vals[k]:.1f}")
                       for k in vals}
    c.text((px0 + px1) / 2, panels[-1][3] + 38, "head parameters", size=11.5, color=T["muted"], ha="center")

    # ---- right: one Matryoshka head read at several sizes ---------------------------------------
    if mrl:
        dims = sorted(mrl["cfg"]["matryoshka"])
        mx0, mx1, my_top, my_bot = 564, 772, 152, 496
        MX = lambda i: mx0 + (mx1 - mx0) * i / (len(dims) - 1)  # noqa: E731
        top_v = max(mrl[f"gen_top1@{d}"] for d in dims) * 100
        ymax = 10 * math.ceil((top_v + 8) / 10)
        MY = lambda v: my_bot - v / ymax * (my_bot - my_top)  # noqa: E731
        hx = mx0 - 34
        c.text(hx, my_top - 28, "One Matryoshka head, read at", size=12, weight=600)
        c.text(hx, my_top - 12, f"{dims[0]} to {dims[-1]} dims: flat", size=12, weight=600)
        for v in range(0, int(ymax) + 1, 10):
            c.hair(mx0 - 12, MY(v), mx1 + 12, MY(v), T["axis"] if v == 0 else T["grid"])
            c.text(mx0 - 20, MY(v) + 4, f"{v}", size=10.5, color=T["muted"], ha="right")
        m_shown = {}
        for key, name in (("gen_top1", "General Q&A"), ("tricky_top1", "Tricky Q&A")):
            ys = [MY(mrl[f"{key}@{d}"] * 100) for d in dims]
            xs = [MX(i) for i in range(len(dims))]
            c.line(xs, ys, T["ours"], lw=2)
            for x, y in zip(xs, ys):
                c.dot(x, y, 4, T["ours"], ring=2)
            c.text(xs[0] - 4, min(ys) - 12, name, size=11.5, weight=600)
            for i in (0, len(dims) - 1):
                c.text(xs[i], ys[i] + 19, pct_up(mrl[f"{key}@{dims[i]}"]), size=11, weight=500, color=T["ink2"], ha="center")
            m_shown[name] = {str(d): pct_up(mrl[f"{key}@{d}"]) for d in dims}
        for i, d in enumerate(dims):
            c.hair(MX(i), my_bot, MX(i), my_bot + 4, T["axis"])
            c.text(MX(i), my_bot + 18, f"{d}", size=10.5, color=T["muted"], ha="center")
        c.text((mx0 + mx1) / 2, my_bot + 38, "output dims used", size=11.5, color=T["muted"], ha="center")
        proj = base.get("cfg", {}).get("proj")
        if proj and proj % dims[0] == 0:
            c.text(hx, my_bot + 62, f"A {dims[0]}-dim action cache is {proj // dims[0]}x smaller", size=11.5, color=T["ink2"])
            c.text(hx, my_bot + 78, f"than {CLM_NAME}'s {proj}-dim one.", size=11.5, color=T["ink2"])
        shown["matryoshka"] = m_shown
    c.footer(f"Source: results/innovations/cpu_sweep/, measured {MD_ONLY['measured_day1']} {DOT} {REPO}")
    return c, shown


# ---------------------------------------------------------------------------------------------
# 6. backbones
# ---------------------------------------------------------------------------------------------
def fig_backbones(T, D: Data):
    bb = {k: d for k, d in D.backbones.items() if is_num(d.get("tricky_top1")) and is_num(d.get("gen_top1"))}
    if len(bb) < 2:
        raise Skip("needs at least two bb_*.json files")
    order = sorted(bb, key=lambda k: -bb[k]["tricky_top1"])
    best = order[0]
    name = lambda k: BACKBONE_LABELS.get(k, k)  # noqa: E731
    RH = 42
    H = 152 + RH * len(order) + 84
    c = Canvas(W, H, T, "backbones")
    steps = bb[best].get("step")
    title = (f"{name(best)} edges out Qwen3-8B under the same small head" if best != "qwen3-8b" and "qwen3-8b" in bb
             else "The same small head on three frozen backbones")
    c.header(title, f"Screening signal only: the same head trained from scratch for {steps:,} steps on a "
                    f"{MD_ONLY['backbone_slice_pairs']}-pair slice of each backbone's vectors. The slice is tiny and the "
                    f"heads overfit it.")
    subs = {k: f"hidden size {bb[k].get('cfg', {}).get('in_dim', '?')}" + (f" {DOT} {CLM_NAME}'s backbone" if k == "qwen3-8b" else "")
            for k in order}
    left = 16 + max([text_w(name(k), 12.5, 600) for k in order] + [text_w(subs[k], 10.5) for k in order]) + 22
    pw = (W - 16 - left - 40) / 2
    panels = [("tricky_top1", "Tricky Q&A, top-1 %", left, left + pw), ("gen_top1", "General Q&A, top-1 %", left + pw + 40, W - 16)]
    top = 126
    ref = D.clm if (D.clm and "qwen3-8b" in bb and D.clm.get("tricky_n") == bb["qwen3-8b"].get("tricky_n")) else None
    shown = {}
    for key, pname, x0, x1 in panels:
        X = lambda v: x0 + v * (x1 - x0 - 30)  # noqa: E731
        c.text(x0, top - 6, pname, size=12, weight=600)
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            c.text(X(v), top + 28 + RH * len(order), f"{v * 100:g}%", size=10.5, color=T["muted"], ha="center")
        grid_z, n0 = c.z(), len(c.boxes)  # the grid is drawn last, so its lines can break around the value labels
        for i, k in enumerate(order):
            yc = top + 31 + RH * i
            v = bb[k][key]
            c.hbar(x0, yc, X(v) - x0, 16, T["clm"] if k == "qwen3-8b" else T["ours"])
            c.text(X(v) + 8, yc + 4.5, pct_up(v), size=12.5, weight=600)
            shown.setdefault(name(k), {})[key] = pct_up(v)
        if ref and is_num(ref.get(key)):
            yc = top + 31 + RH * order.index("qwen3-8b")
            c.ring(X(ref[key]), yc, 5.5, T["clm"], lw=2)
            c.text(X(ref[key]), yc - 12, pct(ref[key]), size=11, weight=500, color=T["ink2"], ha="center")
            shown.setdefault("reference ring", {})[key] = pct(ref[key])
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            c.vgrid(X(v), top + 8, top + 12 + RH * len(order), T["axis"] if v == 0 else T["grid"], c.boxes[n0:],
                    z=grid_z)
    for i, k in enumerate(order):
        yc = top + 31 + RH * i
        c.text(16, yc + 1, name(k), size=12.5, weight=600 if k == best else 400)
        c.text(16, yc + 15, subs[k], size=10.5, color=T["muted"])
    y = top + 56 + RH * len(order)
    if ref:
        c.ring(22, y - 4, 5.5, T["clm"], lw=2)
        c.text(36, y, f"{CLM_NAME}'s fully trained head on Qwen3-8B, same test, for scale. The screening heads are "
                      f"far from converged.", size=11.5, color=T["ink2"])
    c.footer(when=MD_ONLY["measured_day1"])
    return c, shown


# ---------------------------------------------------------------------------------------------
# 7. architecture (a diagram, not a chart)
# ---------------------------------------------------------------------------------------------
def fig_architecture(T, D: Data):
    src = D.ours or D.clm or {}
    cfg = src.get("cfg") or {}
    if not all(is_num(cfg.get(k)) for k in ("in_dim", "width", "depth", "proj")) or not is_num(src.get("params")):
        raise Skip(f"needs the head sizes stored in wise_{RELEASE_ALPHA:g}.json or {CLM_EVAL_RUN}.json")
    in_dim, width, depth, proj = cfg["in_dim"], cfg["width"], cfg["depth"], cfg["proj"]
    per_head = f"{src['params'] / 2 / 1e6:.2f}M parameters"   # two heads of equal size
    H = 446
    c = Canvas(W, H, T, "architecture")
    c.header(f"How {CLM_NAME} decides: one frozen backbone, two small heads, one softmax",
             f"{OURS_NAME} and {ROUTER_NAME} change only the two heads (blue). The backbone and the serving "
             f"engine stay the same.")
    y1, y2, bh = 136, 248, 82                       # the state lane and the option lane
    mid = (y1 + y2) / 2
    col = T["muted"]

    def box(x, yc, w, h, title, lines, ours=False, stack=False):
        fill, edge = (T["ours_box"], T["ours_box_edge"]) if ours else (T["box"], T["box_edge"])
        if stack:
            for off in (8, 4):
                c.rect(x + off, yc - h / 2 + off, w, h, fill, r=8, edge=edge, lw=1)
        c.rect(x, yc - h / 2, w, h, fill, r=8, edge=edge, lw=1.5 if ours else 1)
        ty = yc - (len(lines) * 15.5) / 2 + 4
        c.text(x + w / 2, ty, title, size=12.5, weight=600, ha="center")
        for i, s in enumerate(lines):
            c.text(x + w / 2, ty + 17 + i * 15.5, s, size=11, color=T["ink2"], ha="center")

    ax0, aw = 16, 138          # inputs
    bx0, bw = 184, 108         # backbone
    hx0, hw = 386, 158         # heads
    sx0, sw_ = 592, 78         # similarity
    ox0, ow = 696, 88          # softmax
    box(ax0, y1, aw, bh, "State", ["the situation", "+ instructions"])
    box(ax0, y2, aw, bh, "Options", ["K candidate actions,", "one description each"], stack=True)
    c.rect(bx0, y1 - bh / 2, bw, (y2 - y1) + bh, T["box"], r=8, edge=T["box_edge"], lw=1)
    c.text(bx0 + bw / 2, mid - 20, "Qwen3-8B", size=13, weight=600, ha="center")
    for i, s in enumerate(("frozen backbone", "last-token", "hidden state")):
        c.text(bx0 + bw / 2, mid - 2 + i * 15.5, s, size=11, color=T["ink2"], ha="center")
    mlp = f" {ARROW} ".join([str(in_dim)] + [str(width)] * (depth - 1) + [str(proj)])
    for yc, nm in ((y1, "State head"), (y2, "Action head")):
        box(hx0, yc, hw, bh, nm, [mlp, "MLP, GELU, LayerNorm", per_head], ours=True)
    c.rect(sx0, y1 - bh / 2, sw_, (y2 - y1) + bh, T["box"], r=8, edge=T["box_edge"], lw=1)
    c.text(sx0 + sw_ / 2, mid - 12, "cosine", size=12.5, weight=600, ha="center")
    c.text(sx0 + sw_ / 2, mid + 5, f"{TIMES} learned", size=11, color=T["ink2"], ha="center")
    c.text(sx0 + sw_ / 2, mid + 20.5, "scale", size=11, color=T["ink2"], ha="center")
    box(ox0, mid, ow, 92, "softmax", ["over the", "K options"])
    for yc in (y1, y2):
        c.arrow(ax0 + aw + (11 if yc == y2 else 3), yc, bx0 - 3, yc, col)
        c.arrow(bx0 + bw + 3, yc, hx0 - 3, yc, col)
        vx = (bx0 + bw + hx0) / 2
        c.text(vx, yc - 9, f"{in_dim}-d vector", size=10.5, weight=500, color=T["ink2"], ha="center")
        c.text(vx, yc + 18, "L2-normalized", size=10.5, color=T["muted"], ha="center")
        c.arrow(hx0 + hw + 3, yc, sx0 - 3, yc, col)
        c.text((hx0 + hw + sx0) / 2, yc - 9, f"{proj}-d", size=10.5, weight=500, color=T["ink2"], ha="center")
    c.arrow(sx0 + sw_ + 3, mid, ox0 - 3, mid, col)
    c.arrow(ox0 + ow / 2, mid + 46 + 3, ox0 + ow / 2, mid + 46 + 17, col)
    c.text(ox0 + ow / 2, mid + 80, "one probability", size=11, weight=500, color=T["ink2"], ha="center")
    c.text(ox0 + ow / 2, mid + 95, "per option", size=11, weight=500, color=T["ink2"], ha="center")

    # captions
    y = 336
    c.rect(16, y - 10, 12, 12, T["ours_box"], r=2.5, edge=T["ours_box_edge"], lw=1.5)
    c.rich(36, y, [(f"{OURS_NAME} heads", 600, T["ink"]),
                   (f" = {1 - RELEASE_ALPHA:g} {TIMES} CLM's post-trained heads + {RELEASE_ALPHA:g} {TIMES} our "
                    f"post-trained heads.", 400, T["ink"]),
                   (" Weights are blended. No extra training.", 400, T["ink2"])], size=12)
    y += 18
    c.rich(36, y, [(f"{ROUTER_NAME} heads", 600, T["ink"]), (" = our post-trained heads, not blended.", 400, T["ink"])],
           size=12)
    y += 24
    c.rect(16, y - 10, 12, 12, T["box"], r=2.5, edge=T["box_edge"], lw=1)
    c.text(36, y, f"Unchanged from {CLM_NAME}. Option vectors can be cached, so a decision costs one backbone pass.",
           size=12, color=T["ink2"])
    lat = D.trex_latency_range()
    if lat:
        lo, hi = f"{lat[0]:.0f}", f"{lat[1]:.0f}"
        c.text(36, y + 18, f"Median latency in our T-Rex runs: {lo if lo == hi else f'{lo} to {hi}'} ms per decision "
                           f"on one L40S.", size=12, color=T["ink2"])
    c.footer(f"Model: Kwok et al., Contrastive Language Models, 2026 {DOT} diagram: {REPO}")
    return c, dict(in_dim=in_dim, width=width, proj=proj, per_head=per_head,
                   latency_ms_p50_range=list(lat) if lat else None)


# ---------------------------------------------------------------------------------------------
# 8. social preview (PNG, 1280 x 640, dark)
# ---------------------------------------------------------------------------------------------
def social_preview(D: Data, out: Path):
    heads = (D.heldout or {}).get("heads", {})
    tc, to = D.trex_pooled(CLM_TREX_RUN), D.trex_pooled(OURS_RUN)
    if not (CLM_TREX_RUN in heads and OURS_RUN in heads and tc and to):
        raise Skip("needs heldout.json and trex.json")
    T = THEMES["dark"]
    SW, SH = 1280, 640
    c = Canvas(SW, SH, T, "social-preview")
    c.rect(0, 0, SW, SH, T["surface"])
    c.rect(0, 0, SW, 6, T["ours"])
    c.text(72, 96, "samstealth / system-one-model-tweaks", size=22, weight=500, color=T["muted"])
    c.text(70, 172, PROJECT, size=66, weight=700)
    c.text(72, 220, f"Five measured tweaks to {CLM_NAME}, the open System One decision model.",
           size=27, color=T["ink2"])

    # three stat tiles: two held-out tests and the T-Rex count
    tiles = []
    for key, label in (("bfcl", "Tool choice, held-out test"), ("massive", "Intent routing, held-out test")):
        if key in heads[CLM_TREX_RUN] and key in heads[OURS_RUN]:
            name = next((n for k, n, *_ in HELDOUT_SOURCES if k == key), key.upper())
            tiles.append(dict(before=f"{heads[CLM_TREX_RUN][key]:.1f}", after=f"{heads[OURS_RUN][key]:.1f}%",
                              label=label, sub=f"{name.split()[0]}: {CLM_NAME} {ARROW} {OURS_NAME}"))
    tiles.append(dict(after=f"{to['survived']:g} of {to['seeds']:g}", label="T-Rex games, shield on",
                      sub=f"survived. {CLM_NAME}: {tc['survived']:g} of {tc['seeds']:g}"))
    x, tw, ty = 72, 392, 352
    for i, t in enumerate(tiles):
        tx = x + i * tw
        if i:
            c.hair(tx - 28, ty - 52, tx - 28, ty + 82, T["axis"])
        vx = tx - 2
        if t.get("before"):
            vx += c.text(vx, ty, t["before"], size=52, weight=600, color=T["muted"]) + 14
            vx += c.text(vx, ty - 3, ARROW, size=36, weight=500, color=T["muted"]) + 14
        c.text(vx, ty, t["after"], size=52, weight=700)
        c.text(tx, ty + 42, t["label"], size=22, weight=600)
        c.text(tx, ty + 72, t["sub"], size=18, color=T["ink2"])

    off_c, off_o = D.trex_first(CLM_TREX_RUN, "noshield"), D.trex_first(OURS_RUN, "noshield")
    c.text(72, SH - 116, f"{OURS_NAME} = {round((1 - RELEASE_ALPHA) * 100):g}% {CLM_NAME}'s head + "
                         f"{round(RELEASE_ALPHA * 100):g}% our post-trained copy of it. No extra training.",
           size=21, color=T["ink2"])
    c.text(72, SH - 84, f"Every experiment in the repo: about ${MD_ONLY['modal_spend_usd']} of cloud GPU time "
                        f"in total.", size=21, color=T["ink2"])
    caveat = "Slice-scale runs, not a full retrain."
    if off_c and off_o and off_c["mean_best_score"] > off_o["mean_best_score"]:
        caveat += f" With the safety shield off, {CLM_NAME} still plays T-Rex far better."
    c.text(72, SH - 48, caveat, size=18.5, color=T["muted"])
    c.check()
    c.fig.savefig(out, format="png", dpi=72, facecolor=T["surface"])
    plt.close(c.fig)
    return dict(tiles=[{k: v for k, v in t.items()} for t in tiles], cost_usd=f"about {MD_ONLY['modal_spend_usd']}")


# ---------------------------------------------------------------------------------------------
FIGURES = {
    "hero": (fig_hero, f"{OURS_NAME} vs {CLM_NAME}: held-out tests, agent benchmarks and in-distribution tests"),
    "heldout": (fig_heldout, f"{CLM_NAME}, {OURS_NAME} and {ROUTER_NAME} on eight held-out choice tests"),
    "blend-tradeoff": (fig_blend, "Blend weight against T-Rex survival and held-out accuracy"),
    "quantization": (fig_quant, "Backbone latency and accuracy, and head accuracy, under quantization"),
    "head-layouts": (fig_layouts, "CPU sweep of head layouts"),
    "backbones": (fig_backbones, "The same head on three frozen backbones"),
    "architecture": (fig_architecture, f"Architecture of {CLM_NAME} and what {OURS_NAME} changes"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "innovations")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    ap.add_argument("--only", default="", help="comma-separated figure names (default: all)")
    ap.add_argument("--numbers", action="store_true", help="print the numbers each chart shows, as JSON")
    args = ap.parse_args(argv)

    plt.rcParams["svg.fonttype"] = "path"       # text as paths: identical on every machine
    plt.rcParams["svg.hashsalt"] = "s1-tweaks"  # stable ids, so re-runs give small diffs
    if not HAVE_INTER:
        warn("the Inter font is not installed; using DejaVu Sans, so text widths will differ a little")

    D = Data(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = only - set(FIGURES) - {"social-preview"}
    if unknown:
        warn(f"unknown figure name: {', '.join(sorted(unknown))}. Choose from {', '.join(FIGURES)}, social-preview")
    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p  # noqa: E731
    log = sys.stderr if args.numbers else sys.stdout   # with --numbers, stdout holds only the JSON
    numbers, failed = {}, 0
    for name, (fn, desc) in FIGURES.items():
        if only and name not in only:
            continue
        for theme in ("light", "dark"):
            try:
                c, shown = fn(THEMES[theme], D)
            except Skip as e:
                warn(f"{name}: skipped, {e}")
                failed += 1
                break
            path = args.out / f"{name}-{theme}.svg"
            c.save_svg(path, desc, f"{desc}. Generated by scripts/make_figures.py from results/innovations/.")
            numbers[name] = shown
            print(f"wrote {rel(path)}", file=log)
    if not only or "social-preview" in only:
        try:
            path = args.out / "social-preview.png"
            numbers["social-preview"] = social_preview(D, path)
            print(f"wrote {rel(path)}", file=log)
        except Skip as e:
            warn(f"social-preview: skipped, {e}")
            failed += 1
    if args.numbers:
        print(json.dumps(numbers, indent=1, ensure_ascii=False))
    return 1 if (failed or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
