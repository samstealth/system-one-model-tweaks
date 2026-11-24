#!/usr/bin/env python3
"""Count what the docs say about the data sets, from the built files, and save it as one JSON.

    python scripts/data_checks.py --choice choice --heldout heldout \\
        --qa-test qa_text/qa_text_test.jsonl --out results/innovations/data_checks.json

It reads the files the builders write (docs/REPRODUCE.md, section 1.7) and reports, per source:

  rows        how many rows
  options     how many rows have 3, 4, 5, 8 or 10 options
  chance      the mean of 1 / number of options: the top-1 accuracy of a random pick
  multi_turn  states that carry an ASSISTANT turn and an end-of-text marker (Glaive only)

With --qa-test it also counts how many (doc_id, pair_idx) pairs of the Q&A test slice are in the
published chunks that the post-training runs replayed. That needs three small metadata files from
Hugging Face, about 0.5 MB each.

No GPU. A few seconds on a CPU.
"""
import argparse
import json
import os
import time

REPLAY_REPO = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
REPLAY_REV = "05ca7d05c043350326c5e86c589b9542962747b4"
REPLAYED = (0, 1)   # --replay-chunks shard_00/0,shard_00/1 in the S1-Router run (docs/REPRODUCE.md, 2.6)
CONTROL = (2,)      # a chunk that was not replayed, for comparison


def per_source(path, strip=""):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = len(r["options"])
            d = out.setdefault(r["source"][len(strip):] if r["source"].startswith(strip) else r["source"],
                               {"rows": 0, "options": {}, "inv": 0.0, "multi_turn": 0})
            d["rows"] += 1
            d["options"][str(k)] = d["options"].get(str(k), 0) + 1
            d["inv"] += 1 / k
            d["multi_turn"] += "ASSISTANT:" in r["state"] and "<|endoftext|>" in r["state"]
    for d in out.values():
        d["chance"] = round(d.pop("inv") / d["rows"], 4)
        d["options"] = dict(sorted(d["options"].items(), key=lambda kv: int(kv[0])))
        if not d["multi_turn"]:
            del d["multi_turn"]
    return out


def qa_overlap(test_path):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    with open(test_path, encoding="utf-8") as f:
        pairs = [(r["doc_id"], r["pair_idx"]) for r in map(json.loads, f)]
    chunks, seen = {}, set()
    for i in REPLAYED + CONTROL:
        p = hf_hub_download(REPLAY_REPO, f"shard_00/chunk_{i:04d}_meta.parquet", repo_type="dataset", revision=REPLAY_REV)
        t = pq.read_table(p, columns=["doc_id", "pair_idx"]).to_pydict()
        keys = set(zip(t["doc_id"], t["pair_idx"]))
        hit = [k for k in pairs if k in keys]
        chunks[f"shard_00/{i}"] = {"replayed": i in REPLAYED, "pairs": len(hit), "docs": len({d for d, _ in hit})}
        if i in REPLAYED:
            seen |= set(hit)
    return {"test_pairs": len(pairs), "test_docs": len({d for d, _ in pairs}),
            "repo": REPLAY_REPO, "revision": REPLAY_REV, "chunks": chunks,
            "pairs_in_replayed_chunks": len(seen), "docs_in_replayed_chunks": len({d for d, _ in seen}),
            "share_of_test_pairs": round(len(seen) / len(pairs), 4)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--choice", help="folder written by innovations.build_choice_data (choice_train.jsonl, choice_test.jsonl)")
    p.add_argument("--heldout", help="folder written by innovations.build_heldout_choice (choice_test.jsonl)")
    p.add_argument("--qa-test", help="qa_text_test.jsonl written by innovations.build_qa_text")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = {"note": "scripts/data_checks.py: counts read from the built data files. chance = mean of 1 / number of options. "
                   "multi_turn = states with an ASSISTANT turn and an end-of-text marker.",
           "date": time.strftime("%Y-%m-%d")}
    if a.choice:
        out["choice"] = {s: per_source(os.path.join(a.choice, f"choice_{s}.jsonl")) for s in ("train", "test")}
    if a.heldout:
        out["heldout"] = per_source(os.path.join(a.heldout, "choice_test.jsonl"), strip="heldout_")
    if a.qa_test:
        out["qa_test_slice_in_replay_chunks"] = qa_overlap(a.qa_test)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
