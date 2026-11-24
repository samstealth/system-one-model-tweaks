"""Text in, probabilities out. Needs a GPU with 24 GB or more for the default backbone.

It embeds one state and a list of options with Qwen/Qwen3-8B through transformers, using the
published recipe, then scores them with a head:

  text      state  = f"{state}\\n\\n{instructions}"      option = f"{key}: {description}"
  tokens    no special tokens, at most 2047 per text; a long state keeps its tail, a long option its head
  vector    last token of the final hidden layer, L2 normalized (4096 numbers for Qwen3-8B)
  score     softmax over the options of  scale * cosine(state_head(state), action_head(option))

    pip install "transformers>=4.51" accelerate
    python examples/route_text.py                      # built-in example: an agent choosing a tool (all heads pick create_ticket)
    python examples/route_text.py --head clm           # CLM's official head only
    python examples/route_text.py --state "Customer: my invoice was charged twice." \\
        --instructions "Which team should handle this?" \\
        --option "billing: Charges, invoices, refunds" --option "technical: Bugs and outages"

Option text: the choice data in this repo writes an option as "key: description", so this script
does the same. The official CLM engine sends an option's description to the encoder as written.

This script is for trying things out: it embeds texts one at a time with plain transformers. For
serving, use the official CLM engine (vLLM pooling server plus clm-serve --ckpt), see MODEL_CARD.md.

Smoke test of the code path on a CPU (a small backbone and a RANDOM head, so the numbers mean nothing):

    python examples/route_text.py --backbone Qwen/Qwen3-0.6B --head random --device cpu
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so the script runs from a clone without "pip install -e ."

from innovations.heads import HeadConfig, TwoTower  # noqa: E402
from innovations.scorer import Scorer, load_head  # noqa: E402

DEFAULT_BACKBONE = "Qwen/Qwen3-8B"
CAP = 2047  # tokens per text, the serving default (vLLM --max-model-len 2048)

# A realistic decision: the agent has one request and a handful of tools, and must pick one.
# Measured on one L40S with Qwen/Qwen3-8B in bf16 (results/innovations/route_text_check.json,
# example C): every head picks create_ticket, with probability 0.72 (CLM-8B), 0.82 (S1-Tweaks v0.1)
# and 0.96 (S1-Router v0.1).
EXAMPLE = {
    "state": ("User request: The checkout page has been down for 20 minutes and I cannot fix it myself. "
              "Get the on-call engineer to look at it."),
    "instructions": "Which tool should the agent call next?",
    "options": {
        "run_sql_query": "Run a read-only SQL query against the staging database and return the rows.",
        "search_logs": "Search application and job logs by time range and keyword.",
        "restart_service": "Restart a named service on the staging cluster.",
        "create_ticket": "Open a ticket for the on-call engineer with a title and a description.",
        "send_email": "Send an email to a person or a mailing list.",
        "no_tool": "Answer directly without calling a tool.",
    },
}


def state_text(state, instructions):
    """Context first, question last, separated by a blank line: CLM's serving layout."""
    s, i = state.strip(), (instructions or "").strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def option_text(key, description):
    """The option layout the choice data in this repo was built with: 'key: description'."""
    return f"{key}: {description}" if description else key


def token_ids(tok, text, keep="head", cap=CAP):
    """Token ids without special tokens. A long state keeps its tail (the question), a long option its head."""
    ids = list(tok(text, add_special_tokens=False)["input_ids"])
    if not ids:
        ids = list(tok(" ", add_special_tokens=False)["input_ids"])
    return ids[-cap:] if keep == "tail" else ids[:cap]


@torch.no_grad()
def embed(model, tok, texts, keep="head", device="cpu", cap=CAP):
    """Last-token pooling of the final hidden layer, L2 normalized. One text per forward pass, so no
    padding is involved. Returns float32 (len(texts), hidden_size)."""
    out = []
    for text in texts:
        ids = torch.tensor([token_ids(tok, text, keep, cap)], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        out.append(torch.nn.functional.normalize(hidden[0, -1].float(), dim=-1).cpu())
    return torch.stack(out).numpy()


def backbone_hidden_size(name):
    """Vector size of a backbone, read from its config without loading the weights."""
    try:
        from transformers import AutoConfig
    except ImportError:
        raise SystemExit('This script needs transformers. Install it with: pip install "transformers>=4.51" accelerate')
    return AutoConfig.from_pretrained(name).hidden_size


def load_backbone(name, device):
    """Tokenizer and model through transformers. bf16 on a GPU (the dtype the vectors in this repo were made with), fp32 on a CPU."""
    from transformers import AutoModel, AutoTokenizer
    dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(name)
    kw = {}
    try:
        import accelerate  # noqa: F401  with accelerate the weights load straight onto the device
        kw["device_map"] = {"": device}
    except ImportError:
        pass
    try:
        model = AutoModel.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:  # transformers before the torch_dtype -> dtype rename
        model = AutoModel.from_pretrained(name, torch_dtype=dtype, **kw)
    if not kw:
        model = model.to(device)
    return tok, model.eval(), dtype


def random_head(in_dim, seed=0):
    """An untrained head of the right input size. For smoke tests of the code path only."""
    torch.manual_seed(seed)
    return Scorer(TwoTower(HeadConfig(in_dim=in_dim)), "RANDOM head (untrained)")


def load_heads(specs, in_dim):
    """--head values to Scorers. With no --head, try CLM's head and this repo's two heads and keep what loads."""
    explicit = bool(specs)
    heads = []
    for spec in specs or ["clm", "s1-tweaks", "s1-router"]:
        if spec == "random":
            heads.append(random_head(in_dim))
            continue
        try:
            heads.append(load_head(spec))
        except (OSError, ValueError) as e:  # not published yet, no network, wrong path, not a head file
            if explicit:
                raise SystemExit(str(e))
            print(f"Skipping {spec}: {e}\n", file=sys.stderr)
    if not heads:
        raise SystemExit("No head could be loaded. Pass --head /path/to/head.pt")
    for i, h in enumerate(heads):
        if h.name in [g.name for g in heads[:i]]:
            h.name += f" ({i + 1})"  # two files with the same name still get their own column
        if h.in_dim != in_dim:
            raise SystemExit(f"{h.name} expects {h.in_dim}-d vectors but this backbone produces {in_dim}-d vectors. "
                             "A head only works with the backbone it was trained against "
                             "(Qwen/Qwen3-8B for CLM-8B and S1-Tweaks v0.1).")
    return heads


def parse_options(pairs):
    out = {}
    for p in pairs:
        key, _, desc = p.partition(":")  # "key: description", or a bare key with no description
        if not key.strip():
            raise SystemExit(f'--option needs the form "key: description", got {p!r}')
        out[key.strip()] = desc.strip()
    if len(out) < 2:
        raise SystemExit("Give at least two different --option values.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Embed a state and options with Qwen3-8B, then score the options with a System One head.")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face id or local path of the backbone (default Qwen/Qwen3-8B)")
    p.add_argument("--head", action="append", default=None,
                   help='"clm", "s1-tweaks", "s1-router", a local .pt file, an https URL, hf://org/repo/file.pt, or "random" '
                        "(smoke test). Repeat to compare heads. Default: clm, s1-tweaks and s1-router, whichever can be loaded.")
    p.add_argument("--device", default=None, help="cuda, cuda:0 or cpu (default: cuda)")
    p.add_argument("--state", default=None, help="the situation, as plain text")
    p.add_argument("--instructions", default=None, help="the question asked about the state")
    p.add_argument("--option", action="append", default=[], metavar='"key: description"', help="one option; repeat for each")
    p.add_argument("--temperature", type=float, default=1.0, help="divides the logits before the softmax (default 1)")
    p.add_argument("--json", action="store_true", help="print the result as JSON instead of a table")
    a = p.parse_args(argv)

    if a.device is None:
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU found. " + (
                "Qwen/Qwen3-8B needs a GPU with 24 GB or more.\n" if a.backbone == DEFAULT_BACKBONE else "\n") +
                "  No GPU: python examples/demo_cpu.py scores precomputed vectors on a CPU.\n"
                "  To run this script on a CPU anyway, pass --device cpu (very slow with the 8B backbone).")
        a.device = "cuda"

    if a.state is None and not a.option:
        state, instructions, options = EXAMPLE["state"], EXAMPLE["instructions"], EXAMPLE["options"]
    elif a.state is None or not a.option:
        raise SystemExit("Pass both --state and at least two --option values, or neither to run the built-in example.")
    else:
        state, instructions, options = a.state, a.instructions, parse_options(a.option)
    s_text = state_text(state, instructions)
    o_texts = [option_text(k, d) for k, d in options.items()]

    in_dim = backbone_hidden_size(a.backbone)
    heads = load_heads(a.head, in_dim)  # before the backbone, so a bad --head fails in seconds
    t = time.perf_counter()
    tok, model, dtype = load_backbone(a.backbone, a.device)
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    s_vec = embed(model, tok, [s_text], "tail", a.device)[0]
    o_vecs = embed(model, tok, o_texts, "head", a.device)
    t_embed = time.perf_counter() - t
    t = time.perf_counter()
    probs = {h.name: h.probs(s_vec, o_vecs, temperature=a.temperature) for h in heads}
    t_head = time.perf_counter() - t

    keys = list(options)
    result = {"backbone": a.backbone, "device": str(a.device), "dtype": str(dtype).replace("torch.", ""),
              "state_text": s_text, "option_texts": o_texts,
              "heads": {n: {"pick": keys[int(np.argmax(pr))], "probabilities": {k: float(v) for k, v in zip(keys, pr)}}
                        for n, pr in probs.items()},
              "seconds": {"load_backbone": round(t_load, 2), "embed": round(t_embed, 3), "heads": round(t_head, 4)}}
    if a.json:
        print(json.dumps(result, indent=1))
        return result

    print(f"Backbone  {a.backbone} on {a.device} ({result['dtype']}), vectors of size {in_dim}")
    for h in heads:
        print("Head      " + h.describe())
    if any(h.name.startswith("RANDOM") for h in heads):
        print("NOTE: a RANDOM head is untrained. Its probabilities mean nothing. This run only checks the code path.")
    print("\nState:")
    for line in s_text.splitlines():
        print("    " + line)
    names = list(probs)
    w = min(max(len(o) for o in o_texts), 78)
    print("\n  " + "option".ljust(w) + "".join(n.rjust(max(17, len(n) + 2)) for n in names))
    for j, o in enumerate(o_texts):
        cells = "".join((f"{100 * probs[n][j]:.1f}%" + (" *" if int(np.argmax(probs[n])) == j else "  ")).rjust(max(17, len(n) + 2))
                        for n in names)
        print("  " + (o if len(o) <= w else o[:w - 3] + "...").ljust(w) + cells)
    print("  * = that head's pick")
    print(f"\nLoaded the backbone in {t_load:.1f} s. Embedded {1 + len(o_texts)} texts in {t_embed:.2f} s "
          f"(one at a time, plain transformers). Heads: {1000 * t_head:.1f} ms.")
    return result


if __name__ == "__main__":
    main()
