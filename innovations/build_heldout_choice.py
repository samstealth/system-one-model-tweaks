"""Out-of-distribution choice tests: sources that NO head in this repo was trained on.

The listwise sets in build_choice_data.py have test splits from the same sources we post-train on
(banking77, clinc, ARC, CSQA, MMLU auxiliary_train, glaive, labeled), so gains there are
in-distribution. This builder writes a second, test-only set from public ungated sources that none of
our training touched, with the same row format, state/option templates and K=10 option cap as the
training builder, so a head scored on both sets shows how much of its gain transfers.

  family      source               rows                          template (as in build_choice_data.py)
  tool        heldout_bfcl         BFCL v3 multiple + live_multiple, single-call items     glaive_fc
  routing     heldout_massive      Amazon MASSIVE en-US test, 59 intents                    clinc_oos
              heldout_hwu64        HWU64 test, 64 intents                                   clinc_oos
              heldout_bitext       Bitext customer support, 27 intents, 2,000 sampled       banking77
  mcq         heldout_sciq         SciQ test                                                arc/csqa
              heldout_qasc         QASC validation (8 options)                              arc/csqa
              heldout_mmlu         MMLU test, 2,000 sampled                                 arc/csqa
              heldout_medmcqa      MedMCQA validation, 2,000 sampled                        arc/csqa

OpenBookQA is deliberately NOT used: MMLU's auxiliary_train (our mmlu_aux) was built from ARC, MC_TEST,
OBQA and RACE. Any held-out row whose question/utterance text also occurs in choice_train.jsonl is
dropped, and the drop counts are printed and saved in counts.json.

    python -m innovations.build_heldout_choice --out heldout --train choice/choice_train.jsonl
    modal volume put clm-data heldout /heldout
    modal run innovations/modal_embed.py --job choice --src /data/heldout --root /data/emb/qwen3-8b-heldout
    # score any head with --choice-eval /data/emb/qwen3-8b-heldout (its own root, so nothing trains on it)
"""
import argparse
import csv
import gzip
import json
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

from innovations.build_choice_data import K, humanize, intents, mcq, pick

B = "https://huggingface.co/datasets"
FILES = {
    "bfcl_multiple.json": "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_multiple.json",
    "bfcl_multiple_ans.json": "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_multiple.json",
    "bfcl_live_multiple.json": "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_live_multiple.json",
    "bfcl_live_multiple_ans.json": "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_live_multiple.json",
    "massive_en_test.json.gz": "mteb/amazon_massive_intent/resolve/main/test/en.json.gz",
    "hwu64_test.parquet": "DeepPavlov/hwu64/resolve/main/data/test-00000-of-00001.parquet",
    "hwu64_intents.parquet": "DeepPavlov/hwu64/resolve/main/intents/intents-00000-of-00001.parquet",
    "bitext.csv": "bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    "sciq_test.parquet": "allenai/sciq/resolve/main/data/test-00000-of-00001.parquet",
    "qasc_validation.parquet": "allenai/qasc/resolve/main/data/validation-00000-of-00001.parquet",
    "mmlu_test.parquet": "cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
    "medmcqa_validation.parquet": "openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet",
}
SAMPLE = 2000
ROUTE = "Which team should handle this ticket?"   # banking77's instruction in build_choice_data.py
INTENT = "Which intent does this message express?"  # clinc_oos's instruction


def fetch(dst):
    os.makedirs(dst, exist_ok=True)
    for name, path in FILES.items():
        p = os.path.join(dst, name)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"{B}/{path}", p)
    return dst


def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def key_of(row):
    """The text a duplicate would share: the user/customer message, or the question before the instruction."""
    s = row["state"].split("\n\n")[0]
    for pre in ("Customer message: ", "User request: "):
        if s.startswith(pre):
            s = s[len(pre):]
    return norm(s)


def bfcl(src, rng):
    """Tool choice: gold = the single function the reference answer calls. Negatives: the item's own
    sibling functions first (the realistic, hard ones), then random functions from the whole BFCL pool,
    then "no_tool", K options in total, exactly the glaive_fc template."""
    items = []
    for f in ("bfcl_multiple", "bfcl_live_multiple"):
        ans = {a["id"]: a["ground_truth"] for a in jsonl(os.path.join(src, f + "_ans.json"))}
        for d in jsonl(os.path.join(src, f + ".json")):
            gt = ans.get(d["id"])
            if not gt or len(gt) != 1:
                continue
            gold = next(iter(gt[0]))
            turns = [t for conv in d["question"] for t in conv if t.get("role") == "user"]
            fns = {fn["name"]: (fn.get("description") or "").strip() for fn in d["function"]}
            if not turns or gold not in fns:
                continue
            items.append((turns[-1]["content"].strip(), gold, fns))
    pool = {}
    for _, _, fns in items:
        for n, desc in fns.items():
            pool.setdefault(n, desc)
    rows = []
    for user, gold, fns in items:
        sib = [n for n in fns if n != gold]
        rng.shuffle(sib)
        rest = [n for n in pool if n != gold and n not in fns]
        negs = (sib + rng.sample(rest, max(0, K - 2 - len(sib))))[:K - 2] + ["no_tool"]
        opts = negs + [gold]
        rng.shuffle(opts)
        desc = lambda n: "Answer directly without calling a tool." if n == "no_tool" else (fns.get(n) or pool[n])
        rows.append({"source": "heldout_bfcl", "split": "test",
                     "state": f"User request: {user}\n\nWhich tool should the agent call next?",
                     "options": [f"{n}: {desc(n)}" for n in opts], "gold": opts.index(gold)})
    return rows


def routing(src, rng):
    rows = []
    ms = [json.loads(l) for l in gzip.open(os.path.join(src, "massive_en_test.json.gz"), "rt", encoding="utf-8")]
    rows += intents(((x["text"], x["label"]) for x in ms), {x["label"]: x["label"] for x in ms},
                    "heldout_massive", "test", rng, INTENT)
    names = {str(r["id"]): r["name"] for r in pq.read_table(os.path.join(src, "hwu64_intents.parquet")).to_pylist()}
    hw = pq.read_table(os.path.join(src, "hwu64_test.parquet")).to_pylist()
    rows += intents(((r["utterance"], r["label"]) for r in hw), names, "heldout_hwu64", "test", rng, INTENT)
    bt = list(csv.DictReader(open(os.path.join(src, "bitext.csv"), encoding="utf-8")))
    bnames = {r["intent"]: r["intent"] for r in bt}
    bt = random.Random(1).sample(bt, SAMPLE)
    rows += intents(((r["instruction"], r["intent"]) for r in bt), bnames, "heldout_bitext", "test", rng, ROUTE)
    return rows


def mcqs(src, rng):
    rows = []
    for r in pq.read_table(os.path.join(src, "sciq_test.parquet")).to_pylist():
        opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        rng.shuffle(opts)
        rows.append(mcq(r["question"], opts, opts.index(r["correct_answer"]), "heldout_sciq", "test"))
    for r in pq.read_table(os.path.join(src, "qasc_validation.parquet")).to_pylist():
        ch = r["choices"]
        rows.append(mcq(r["question"], ch["text"], ch["label"].index(r["answerKey"]), "heldout_qasc", "test", ch["label"]))
    mm = random.Random(2).sample(pq.read_table(os.path.join(src, "mmlu_test.parquet")).to_pylist(), SAMPLE)
    for r in mm:
        rows.append(mcq(r["question"], list(r["choices"]), int(r["answer"]), "heldout_mmlu", "test"))
    md = random.Random(3).sample(pq.read_table(os.path.join(src, "medmcqa_validation.parquet")).to_pylist(), SAMPLE)
    for r in md:
        rows.append(mcq(r["question"], [r["opa"], r["opb"], r["opc"], r["opd"]], int(r["cop"]), "heldout_medmcqa", "test"))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--train", required=True, help="choice_train.jsonl from build_choice_data.py, for de-duplication")
    p.add_argument("--src", default="", help="folder with the raw files (default: <out>/raw, downloaded if missing)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    src = fetch(a.src or os.path.join(a.out, "raw"))
    rng = random.Random(a.seed)
    rows = bfcl(src, rng) + routing(src, rng) + mcqs(src, rng)
    seen = {key_of(json.loads(l)) for l in open(a.train, encoding="utf-8") if l.strip()}
    counts = {}
    kept = []
    for r in rows:
        c = counts.setdefault(r["source"], {"kept": 0, "dropped_overlap": 0})
        if key_of(r) in seen:
            c["dropped_overlap"] += 1
        else:
            c["kept"] += 1
            kept.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "choice_test.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    json.dump(counts, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
