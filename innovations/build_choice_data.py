"""Innovation 5: listwise post-training data in the exact format the CLM engine serves.

Why: at serving time the engine embeds  state = f"{state}\\n\\n{instructions}"  and every option as
f"{key}: {description}" (CLM src/clm/schema.py), then softmaxes over a handful of options from the
SAME question. Our reconstructed post-training (pipeline/build_agent_data.py) only ever saw one
agent action per state with random in-batch negatives, so the head never learned to separate
sibling options of one question. That matches the failure we see: Q&A holds at 69% but T-Rex
label reading collapses (10% planner agreement against 66.8% for the authors' head).

This builder writes rows of {source, split, state, options[], gold}, where the other options of
the same question are the hard negatives. Sources (all public, ungated on Hugging Face):
  banking77      ticket routing, 77 intents                         (enterprise router use case)
  clinc_oos      intent routing, 150 intents + out-of-scope
  glaive_fc      tool selection: user turn -> the function called, negatives from the tool pool
  arc, csqa, mmlu_aux   multiple-choice reasoning, options as actions
  labeled        options carry inline labels ("Recommended", "Unsafe", "Correct", ...), the state
                 asks for the labelled one. Generic label reading; it never uses T-Rex text, but
                 it does teach the skill T-Rex's "labeled" prompt tests, so report T-Rex as
                 in-domain when this source is on.

Held out for testing: banking77 test, clinc test, ARC-Challenge test, CSQA validation, and 3%
of glaive / labeled rows. Output: <out>/choice_{train,test}.jsonl

    python -m innovations.build_choice_data --src /home/user/data/choice_src --out /home/user/data/choice
"""
import argparse
import ast
import json
import os
import random
import re
import zlib

import pyarrow.parquet as pq

K = 10  # options kept per row (gold + up to 9 negatives), matching the hard-negative layout


def label_names(readme, config=None):
    txt = open(readme).read()
    if config:
        txt = txt[txt.index(f"config_name: {config}"):]
    block = txt[txt.index("names:"):]
    names = {}
    for line in block.splitlines()[1:]:
        m = re.match(r"\s+'(\d+)': (.+)", line)
        if not m:
            if names:
                break
            continue
        names[m.group(1)] = m.group(2).strip()
    return names


def humanize(intent):
    return intent.replace("_", " ")


def pick(options, gold, rng, k=K):
    """Keep gold plus k-1 random others, shuffle, return (options, gold_index)."""
    others = [o for o in options if o != gold]
    rng.shuffle(others)
    opts = others[:k - 1] + [gold]
    rng.shuffle(opts)
    return opts, opts.index(gold)


def intents(rows, names, source, split, rng, instr):
    out = []
    allnames = sorted(set(names.values()))
    for text, lab in rows:
        gold = names[str(lab)]
        opts, g = pick(allnames, gold, rng)
        out.append({"source": source, "split": split, "state": f"Customer message: {text}\n\n{instr}",
                    "options": [f"{humanize(o)}: Route the ticket to {humanize(o)}." for o in opts], "gold": g})
    return out


def mcq(question, choices, gold_i, source, split, keys=None):
    keys = keys or [chr(65 + i) for i in range(len(choices))]
    return {"source": source, "split": split,
            "state": f"{question}\n\nChoose the correct answer.",
            "options": [f"{k}: {c}" for k, c in zip(keys, choices)], "gold": gold_i}


def glaive(path, rng, limit):
    data = json.load(open(path))
    pool, rows = {}, []
    fn_re = re.compile(r'"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"')
    for d in data:
        for n, desc in fn_re.findall(d.get("system", "")):
            pool.setdefault(n, desc)
    for d in data:
        chat = d.get("chat", "")
        m = re.search(r"USER: (.*?)\n\n\nASSISTANT: <functioncall> \{\"name\": \"([^\"]+)\"", chat, re.S)
        if not m or m.group(2) not in pool:
            continue
        user, fn = m.group(1).strip(), m.group(2)
        cands = [n for n in pool if n != fn]
        negs = rng.sample(cands, K - 2) + ["no_tool"]
        opts = negs + [fn]
        rng.shuffle(opts)
        desc = lambda n: "Answer directly without calling a tool." if n == "no_tool" else pool[n]
        split = "test" if zlib.crc32(user.encode()) % 100 < 3 else "train"
        rows.append({"source": "glaive_fc", "split": split, "state": f"User request: {user}\n\nWhich tool should the agent call next?",
                     "options": [f"{n}: {desc(n)}" for n in opts], "gold": opts.index(fn)})
        if len(rows) >= limit:
            break
    return rows


LABELS = [("Recommended", "Not recommended"), ("Correct", "Incorrect"), ("Safe", "Unsafe"),
          ("Best", "Worse"), ("Approved", "Rejected"), ("Valid", "Invalid")]
ASKS = ["Pick the {pos} option.", "Choose the option marked {pos}.", "Which option is {pos}?",
        "Select the {pos} choice and avoid anything {neg}."]


def labeled(mcq_rows, rng, n):
    """Re-use MCQ option texts but put the answer in an inline label, so the decision depends on reading it."""
    out = []
    for r in rng.sample(mcq_rows, min(n, len(mcq_rows))):
        pos, neg = rng.choice(LABELS)
        g = rng.randrange(len(r["options"]))
        opts = []
        for i, o in enumerate(r["options"]):
            key, body = o.split(": ", 1)
            tag = pos if i == g else neg
            opts.append(f"{key}: {tag}. {body}" if rng.random() < 0.5 else f"{key}: {body}. {tag}.")
        ctx = r["state"].split("\n\n")[0]
        split = "test" if zlib.crc32(ctx.encode()) % 100 < 3 else "train"
        out.append({"source": "labeled", "split": split,
                    "state": f"{ctx}\n\n{rng.choice(ASKS).format(pos=pos.lower(), neg=neg.lower())}",
                    "options": opts, "gold": g})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--glaive", type=int, default=60000)
    p.add_argument("--labeled", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    S = lambda f: os.path.join(a.src, f)
    rows = []

    bnames = label_names(S("banking_readme.md"))
    route = "Which team should handle this ticket?"
    for split, f in (("train", "PolyAI_banking77__data_train-00000-of-00001.parquet"), ("test", "PolyAI_banking77__data_test-00000-of-00001.parquet")):
        t = pq.read_table(S(f)).to_pydict()
        rows += intents(zip(t["text"], t["label"]), bnames, "banking77", split, rng, route)
    cnames = label_names(S("clinc_readme.md"), "plus")
    for split, f in (("train", "clinc_train.parquet"), ("test", "clinc_test.parquet")):
        t = pq.read_table(S(f)).to_pydict()
        rows += intents(zip(t["text"], t["intent"]), cnames, "clinc_oos", split, rng, "Which intent does this message express?")

    mc = []
    for f, src, split in (("allenai_ai2_arc__ARC-Challenge_train-00000-of-00001.parquet", "arc", "train"),
                          ("allenai_ai2_arc__ARC-Easy_train-00000-of-00001.parquet", "arc", "train"),
                          ("allenai_ai2_arc__ARC-Challenge_test-00000-of-00001.parquet", "arc", "test"),
                          ("tau_commonsense_qa__data_train-00000-of-00001.parquet", "csqa", "train"),
                          ("tau_commonsense_qa__data_validation-00000-of-00001.parquet", "csqa", "test")):
        for r in pq.read_table(S(f)).to_pylist():
            ch = r["choices"] if isinstance(r["choices"], dict) else ast.literal_eval(r["choices"])
            if r["answerKey"] not in ch["label"]:
                continue
            mc.append(mcq(r["question"], ch["text"], ch["label"].index(r["answerKey"]), src, split, ch["label"]))
    for r in pq.read_table(S("mmlu_aux.parquet")).column("train").to_pylist():
        r = r if isinstance(r, dict) else ast.literal_eval(r)
        mc.append(mcq(r["question"], r["choices"], int(r["answer"]), "mmlu_aux", "train"))
    rows += mc
    rows += glaive(S("glaiveai_glaive-function-calling-v2__glaive-function-calling-v2.json"), rng, a.glaive)
    rows += labeled([r for r in mc if r["split"] == "train"], rng, a.labeled)

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    with open(os.path.join(a.out, "choice_train.jsonl"), "w") as tr, open(os.path.join(a.out, "choice_test.jsonl"), "w") as te:
        for r in rows:
            (te if r["split"] == "test" else tr).write(json.dumps(r) + "\n")
            counts.setdefault(r["source"], {}).setdefault(r["split"], 0)
            counts[r["source"]][r["split"]] += 1
    json.dump(counts, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
