"""Innovation 1 prep: rebuild Q&A text from the public Qwen3 token ids, so any other backbone can embed it.

The published vectors exist only for Qwen3-8B. To compare a new backbone fairly we embed the SAME
pairs with Qwen3-8B and with the new model, train identical heads on each, and score both on the
same doc-disjoint holdout (with hard negatives for the tricky test).

Source: jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens, part_000000 of docs / qa_pairs / neg_pairs.
Output jsonl rows: {doc_id, pair_idx, split, context, question, answer, negatives[]}

    python -m innovations.build_qa_text --out /data/qa_text --train-docs 40000 --test-docs 4000
"""
import argparse
import json
import os
import zlib

REPO = "jackyk02/nemotron-cc-v2.1-hq-dqa-qwen3-tokens"


def main(argv=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--part", default="part_000000")
    p.add_argument("--train-docs", type=int, default=40000)
    p.add_argument("--test-docs", type=int, default=4000)
    p.add_argument("--max-context-tokens", type=int, default=1536)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    f = {k: hf_hub_download(REPO, f"{k}/{a.part}.parquet", repo_type="dataset") for k in ("docs", "qa_pairs", "neg_pairs")}

    # Pick docs by hash so the split is stable and doc-disjoint.
    qa = pq.ParquetFile(f["qa_pairs"])
    want_train, want_test, chosen = a.train_docs, a.test_docs, {}
    for rg in range(qa.num_row_groups):
        for d in qa.read_row_group(rg, columns=["doc_id"]).column("doc_id").to_pylist():
            if d in chosen:
                continue
            is_test = zlib.crc32(d.encode()) % 10 == 0
            if is_test and want_test > 0:
                chosen[d] = "test"; want_test -= 1
            elif not is_test and want_train > 0:
                chosen[d] = "train"; want_train -= 1
        if want_train <= 0 and want_test <= 0:
            break

    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)
    ctx = {}
    docs = pq.ParquetFile(f["docs"])
    for rg in range(docs.num_row_groups):
        t = docs.read_row_group(rg, columns=["doc_id", "context_ids"]).to_pydict()
        for d, ids in zip(t["doc_id"], t["context_ids"]):
            if d in chosen:
                ctx[d] = dec(ids[: a.max_context_tokens])
        if len(ctx) == len(chosen):
            break

    negs = {}
    nf = pq.ParquetFile(f["neg_pairs"])
    for rg in range(nf.num_row_groups):
        t = nf.read_row_group(rg, columns=["doc_id", "pair_idx", "neg_idx", "neg_ids"]).to_pydict()
        for d, pi, ni, ids in zip(t["doc_id"], t["pair_idx"], t["neg_idx"], t["neg_ids"]):
            if chosen.get(d) == "test":
                negs.setdefault((d, pi), {})[ni] = dec(ids)

    n = {"train": 0, "test": 0}
    with open(os.path.join(a.out, "qa_text_train.jsonl"), "w") as tr, open(os.path.join(a.out, "qa_text_test.jsonl"), "w") as te:
        for rg in range(qa.num_row_groups):
            t = qa.read_row_group(rg, columns=["doc_id", "pair_idx", "question_ids", "answer_ids"]).to_pydict()
            for d, pi, q, an in zip(t["doc_id"], t["pair_idx"], t["question_ids"], t["answer_ids"]):
                s = chosen.get(d)
                if s is None or d not in ctx:
                    continue
                row = {"doc_id": d, "pair_idx": pi, "split": s, "context": ctx[d], "question": dec(q), "answer": dec(an)}
                if s == "test":
                    row["negatives"] = [v for _, v in sorted(negs.get((d, pi), {}).items())]
                (te if s == "test" else tr).write(json.dumps(row) + "\n")
                n[s] += 1
    print(json.dumps(n))


if __name__ == "__main__":
    main()
