"""The no-GPU demo: real decisions, scored on your CPU in a few seconds.

What it does. It loads a small sample of real decisions whose Qwen3-8B vectors were computed in
advance on a GPU (examples/data/sample_decisions.npz and .jsonl). It scores every decision with
CLM's official head and with this repo's two heads (S1-Tweaks v0.1 and S1-Router v0.1), prints a
few decisions side by side, then prints the accuracy of each head per source.

What it does not do. It does not run the Qwen3-8B backbone. Turning new text into vectors needs a
GPU: see examples/route_text.py.

    python examples/demo_cpu.py
    python examples/demo_cpu.py --head /path/to/s1_tweaks_v0.1.pt    # one local head against CLM's
    python examples/demo_cpu.py --head s1-tweaks --head my_head.pt   # repeat --head to compare several
    python examples/demo_cpu.py --show 6 --disagreements             # decisions where the heads differ
    python examples/demo_cpu.py --json hits.json                     # also save the hits per head and source
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so the demo runs from a clone without "pip install -e ."

from innovations.scorer import (CLM_NAME, S1_ROUTER_FILE, S1_ROUTER_NAME, S1_TWEAKS_FILE,  # noqa: E402
                                S1_TWEAKS_NAME, load_head)

DATA_DIR = os.path.join(HERE, "data")
NAME = "sample_decisions"
RAW_URL = "https://github.com/samstealth/system-one-model-tweaks/raw/HEAD/examples/data/"


def sample_paths(data):
    """--data is a folder holding the two files, or the path of the .npz (the .jsonl sits next to it)."""
    base = data[:-4] if data.endswith(".npz") else os.path.join(data, NAME)
    return base + ".npz", base + ".jsonl"


def load_sample(data=DATA_DIR):
    npz, jsonl = sample_paths(data)
    missing = [p for p in (npz, jsonl) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "Sample data not found.\n" + "".join(f"  missing: {p}\n" for p in missing) +
            "These two files hold Qwen3-8B vectors and the text of a small set of real decisions.\n"
            "They ship with the repo in examples/data/. To get them:\n"
            "  git pull\n"
            "or download them into examples/data/:\n"
            "  mkdir -p examples/data\n"
            f"  curl -L -o examples/data/{NAME}.npz {RAW_URL}{NAME}.npz\n"
            f"  curl -L -o examples/data/{NAME}.jsonl {RAW_URL}{NAME}.jsonl\n"
            "or point at a copy elsewhere: --data /path/to/folder")
    try:
        z = np.load(npz)
        q, opts, n_opts, gold = z["q"], z["opts"], z["n_opts"].astype(np.int64), z["gold"].astype(np.int64)
    except Exception as e:
        raise SystemExit(f"{npz} is not a readable sample file ({type(e).__name__}: {e}).\n"
                         "It needs the arrays q, opts, n_opts and gold. If the file is a Git LFS pointer, run: git lfs pull")
    with open(jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n = len(q)
    problems = []
    if q.ndim != 2 or opts.ndim != 3 or opts.shape[0] != n or opts.shape[2] != q.shape[1]:
        problems.append(f"q is {q.shape} and opts is {opts.shape}; expected (N, D) and (N, K, D)")
    if len(n_opts) != n or len(gold) != n or len(rows) != n:
        problems.append(f"{n} state vectors, {len(n_opts)} option counts, {len(gold)} gold labels, {len(rows)} text rows")
    if not problems:
        if (gold < 0).any() or (gold >= n_opts).any() or (n_opts > opts.shape[1]).any():
            problems.append("a gold index or an option count is out of range")
        bad = [i for i, r in enumerate(rows) if len(r["options"]) != n_opts[i] or r["gold"] != gold[i]]
        if bad:
            problems.append(f"text row {bad[0]} does not match its vectors (and {len(bad) - 1} more)")
    if problems:
        raise SystemExit(f"The sample in {os.path.dirname(npz)} is inconsistent: " + "; ".join(problems))
    return q, opts, n_opts, gold, rows


def clip(text, head=360, tail=200):
    """Keep the start and the end of a long text. The question sits at the end of a state."""
    text = text.strip()
    return text if len(text) <= head + tail + 20 else text[:head].rstrip() + " [...] " + text[-tail:].lstrip()


def one_line(text, width):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


SHOW_FIRST = ("glaive_fc", "banking77", "heldout_bfcl")  # one tool choice, one routing, one held-out decision


def choose_rows(rows, picks, show, disagreements):
    """Which decisions to print: one of each source in turn, or only those where the heads differ.

    Tool choice, routing and a held-out source come first. Within a source, a state that is one
    request comes before a state that carries earlier turns of a dialogue: about 4 in 10 Glaive
    states do, with an end-of-text marker (README, Limitations). The accuracy table uses every row.
    """
    idx = list(range(len(rows)))
    if disagreements and len(picks) > 1:
        names = list(picks)
        idx = [i for i in idx if len({int(picks[h][i]) for h in names}) > 1]
    by_source = {}
    for i in idx:
        by_source.setdefault(rows[i]["source"], []).append(i)
    by_source = {s: sorted(by_source[s], key=lambda i: ("<|endoftext|>" in rows[i]["state"], i))
                 for s in [s for s in SHOW_FIRST if s in by_source] + [s for s in by_source if s not in SHOW_FIRST]}
    out = []
    while len(out) < show and any(by_source.values()):
        for src in list(by_source):
            if by_source[src] and len(out) < show:
                out.append(by_source[src].pop(0))
    return out


def col(name):
    """Width of a head's column: room for the name and for '100.0% *'."""
    return max(17, len(name) + 3)


def print_decision(i, row, n_total, probs, picks, width=100):
    names = list(probs)
    print(f"\n--- decision {i + 1} of {n_total}, source: {row['source']} " + "-" * 20)
    print("State:")
    for line in clip(row["state"]).splitlines():
        print("    " + line)
    opt_w = max(30, width - 6 - sum(col(n) for n in names))
    print("\n  " + " # " + " " + "option".ljust(opt_w) + "".join((n + "  ").rjust(col(n)) for n in names))
    for j, text in enumerate(row["options"]):
        tag = "  <- gold" if j == row["gold"] else ""
        label = one_line(text, opt_w - len(tag)) + tag
        cells = "".join((f"{100 * probs[h][i, j]:.1f}%" + (" *" if picks[h][i] == j else "  ")).rjust(col(h)) for h in names)
        print(f"  {j:>2}  " + label.ljust(opt_w) + cells)
    verdict = "   ".join(f"{h}: {'right' if picks[h][i] == row['gold'] else 'wrong'}" for h in names)
    print(f"  * = that head's pick.   {verdict}")


def accuracy_table(rows, n_opts, gold, picks):
    names = list(picks)
    sources = sorted({r["source"] for r in rows})
    src = np.array([r["source"] for r in rows])
    w = max([len(s) for s in sources] + [len("source"), 3])
    print("  " + "source".ljust(w) + "n".rjust(7) + "chance".rjust(9) + "".join(h.rjust(col(h)) for h in names))
    acc = {h: {} for h in names}
    hits = {h: {} for h in names}  # the layout of results/innovations/demo_sample.json
    for s in sources + ["all"]:
        m = np.ones(len(rows), bool) if s == "all" else src == s
        line = "  " + s.ljust(w) + str(int(m.sum())).rjust(7) + f"{100 * (1 / n_opts[m]).mean():.1f}%".rjust(9)
        for h in names:
            acc[h][s] = float((picks[h][m] == gold[m]).mean())
            hits[h][s] = {"n": int(m.sum()), "hits": int((picks[h][m] == gold[m]).sum())}
            line += f"{100 * acc[h][s]:.1f}%".rjust(col(h))
        print(line)
    return acc, hits


def main(argv=None):
    p = argparse.ArgumentParser(description="Score a sample of real decisions with CLM's head, S1-Tweaks v0.1 and "
                                            "S1-Router v0.1 on a CPU.")
    p.add_argument("--data", default=DATA_DIR, help="folder with sample_decisions.npz and .jsonl (default: examples/data)")
    p.add_argument("--clm-head", default="clm", help='CLM\'s head: "clm" downloads it from Hugging Face, or pass a local .pt file')
    p.add_argument("--head", action="append", default=None,
                   help='a head to compare with CLM\'s: "s1-tweaks", "s1-router", a local .pt file or a URL. '
                        "Repeat to compare several. Default: s1-tweaks and s1-router, each downloaded from the release page.")
    p.add_argument("--show", type=int, default=3, help="how many decisions to print (default 3)")
    p.add_argument("--disagreements", action="store_true", help="print only decisions where the heads pick differently")
    p.add_argument("--json", default=None, metavar="FILE",
                   help="also write the hits of each head per source to FILE, as JSON")
    a = p.parse_args(argv)

    q, opts, n_opts, gold, rows = load_sample(a.data)
    print(f"Loaded {len(rows)} decisions from {sample_paths(a.data)[0]} (vectors of size {q.shape[1]}, up to {opts.shape[1]} options each).")
    if any(r.get("synthetic") or str(r["source"]).startswith("synthetic") for r in rows):
        print("NOTE: this is a SYNTHETIC sample (random vectors, not Qwen3-8B output). The accuracies below mean nothing.")

    heads = []
    known = {"clm": CLM_NAME, "s1-tweaks": S1_TWEAKS_NAME, "s1-router": S1_ROUTER_NAME,
             S1_TWEAKS_FILE: S1_TWEAKS_NAME, S1_ROUTER_FILE: S1_ROUTER_NAME}  # a local copy keeps its display name
    for src in [a.clm_head] + (a.head or ["s1-tweaks", "s1-router"]):
        label = known.get(src.lower()) or known.get(os.path.basename(src))
        try:
            heads.append(load_head(src, name=label))
        except (OSError, ValueError) as e:
            print(f"\nSkipping {label or src}: {e}")
    if not heads:
        raise SystemExit("\nNo head could be loaded, so there is nothing to score. See the messages above.")
    seen = {}
    for h in heads:  # two files with the same name still get their own column
        seen[h.name] = seen.get(h.name, 0) + 1
        if seen[h.name] > 1:
            h.name += f" ({seen[h.name]})"
    for h in heads:
        print("Head  " + h.describe())

    t = time.perf_counter()
    try:
        probs = {h.name: h.probs(q, opts, n_opts) for h in heads}
    except ValueError as e:
        raise SystemExit(f"\nCould not score the sample: {e}")
    took = time.perf_counter() - t
    picks = {name: P.argmax(1) for name, P in probs.items()}

    shown = choose_rows(rows, picks, a.show, a.disagreements)
    if a.disagreements and len(heads) < 2:
        print("\n--disagreements needs two heads; showing the first decisions instead.")
    elif a.disagreements and not shown:
        print("\nThe heads pick the same option on every decision in this sample.")
    for i in shown:
        print_decision(i, rows[i], len(rows), probs, picks)

    print(f"\nTop-1 accuracy on this sample ({len(rows)} decisions). Chance is the mean of 1 / number of options.")
    acc, hits = accuracy_table(rows, n_opts, gold, picks)
    print(f"\nScored {len(rows)} decisions with {len(heads)} head{'s' if len(heads) > 1 else ''} in {took:.2f} s on CPU.")
    print("Only the heads ran here. The backbone vectors were computed in advance.")
    print("A sample this small gives rough numbers. Measured results and their limits: innovations/RESULTS.md")
    if a.json:
        out = {"note": f"examples/demo_cpu.py --json on {os.path.basename(sample_paths(a.data)[0])} ({len(rows)} decisions), "
                       f"CPU, heads only, {time.strftime('%Y-%m-%d')}", "heads": hits}
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print(f"Wrote the hits per head and source to {a.json}")
    return {"n": len(rows), "seconds": took, "accuracy": acc, "hits": hits, "shown": shown}


if __name__ == "__main__":
    main()
