"""GPU side of innovations 1, 3 and 5: embed text with any backbone (optionally quantized) on Modal,
writing the same chunk layout the trainer reads, so `innovations.train --root /data/emb/<tag>` just works.

Jobs
  qa       Q&A text from build_qa_text.py. Train pairs -> qa/shard_00/chunk_0000..; holdout with hard
           negatives -> qa/shard_00/chunk_0028 + neg/shard_00/chunk_0028 (the trainer's "tricky" test)
           and qa/shard_05/chunk_0083 (the "general" test).
  choice   listwise rows from build_choice_data.py: state -> q, gold option -> a, the other options
           of the same question -> n (hard negatives). Train -> qa/shard_10/.., test -> qa/shard_11/..
  latency  batch-1 time to embed one state, for each backbone / quantization.

Backbones (verify ids on the Hub before a long run; they were checked on 22 Nov 2026):
  qwen3-8b          Qwen/Qwen3-8B                       baseline, hidden 4096
  qwen3-8b-fp8      Qwen/Qwen3-8B-FP8                   official FP8 checkpoint
  qwen3-8b-awq      Qwen/Qwen3-8B-AWQ                   official 4-bit AWQ checkpoint
  qwen3-4b          Qwen/Qwen3-4B                       smaller backbone, hidden 2560
  inkling-2-mini    thinkingmachines/Inkling-2-Mini     26.4B MoE, 3.4B active, looped layers
  muse-glimmer-30b  meta-models/Muse-Glimmer-30B        Apache 2.0, single-GPU
  (Inkling-2-Small 594B and Muse-Spark-Open 816B need 4-8 big GPUs; see innovations/PLAN.md)

    modal run --detach modal_embed.py --job qa --backbone qwen3-8b
    modal run --detach modal_embed.py --job qa --backbone inkling-2-mini --gpu H100
    modal run --detach modal_embed.py --job choice --backbone qwen3-8b
    modal run modal_embed.py --job latency --backbone qwen3-8b,qwen3-8b-fp8,qwen3-8b-awq
    modal run modal_embed.py --job heads      # the authors' heads into /data/heads
"""
import json
import os
import time

import modal

BACKBONES = {
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B"),
    "qwen3-8b-fp8": dict(repo="Qwen/Qwen3-8B-FP8"),
    "qwen3-8b-awq": dict(repo="Qwen/Qwen3-8B-AWQ", quantization="awq_marlin"),
    "qwen3-4b": dict(repo="Qwen/Qwen3-4B"),
    # Released after vllm 0.11 / transformers 4.x: run these with CLM_NEW_STACK=1 (unpinned vllm and
    # transformers) and, if vLLM's pooling runner rejects the architecture, --engine hf.
    # needs vLLM >= 0.35 (unreleased on 22 Nov 2026): run with CLM_IMAGE=vllm/vllm-openai:inkling-2-mini
    "inkling-2-mini": dict(repo="thinkingmachines/Inkling-2-Mini", hidden=2304, new=True, skip_lm_head=True),   # Inkling2LoopForConditionalGeneration
    "muse-glimmer-30b": dict(repo="meta-models/Muse-Glimmer-30B", hidden=6656, new=True),    # MuseGlimmerForConditionalGeneration
}
CAP = 2047  # token cap per side, the repo's serving default

STACK = (["vllm", "transformers", "accelerate"] if os.environ.get("CLM_NEW_STACK") == "1"
         else ["vllm==0.11.0", "transformers>=4.56,<5"])  # the pinned stack is the one the repo verified for Qwen3
if os.environ.get("CLM_IMAGE"):  # e.g. vllm/vllm-openai:inkling-2-mini, for archs no vLLM release supports yet
    image = (modal.Image.from_registry(os.environ["CLM_IMAGE"]).entrypoint([])
             .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")  # the image ships python3 only
             .pip_install("pyarrow", "hf_transfer")  # image python may be newer than pyarrow 18 wheels
             .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))
else:
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install(*STACK, "pyarrow==18.1.0", "numpy", "huggingface_hub", "hf_transfer")
             .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-innovations-embed")


def _skip_lm_head_in_pooling_loader():
    """Workaround for vLLM's Inkling2LoopForEmbedding (nightly, Nov 2026), which raises
    'Unexpected checkpoint tensor: lm_head.weight' because the checkpoint ships its unembedding
    (model.llm.unembed.weight, renamed to lm_head.weight) and the pooling wrapper has no LM head.
    We drop that tensor before load_weights sees it. The engine must run in-process so the patch
    applies to the worker that loads the weights."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm.model_executor.models import ModelRegistry
    names = [a for a in ModelRegistry.get_supported_archs() if "Inkling" in a and "Embed" in a]
    for arch in names:
        cls = ModelRegistry._try_load_model_cls(arch)
        if cls is None or getattr(cls, "_clm_patched", False):
            continue
        orig = cls.load_weights

        def load_weights(self, weights, _orig=orig):
            keep = ((n, w) for n, w in weights if not ("lm_head" in n or n.endswith("unembed.weight")))
            return _orig(self, keep)

        cls.load_weights, cls._clm_patched = load_weights, True
    print("patched lm_head skip for", names, flush=True)


@app.cls(image=image, volumes={"/data": vol}, gpu=os.environ.get("CLM_GPU", "L40S"), cpu=8, memory=64000,
         timeout=6 * 3600)  # gated repos: add secrets=[modal.Secret.from_name("huggingface")]
class Embedder:
    backbone: str = modal.parameter()
    engine: str = modal.parameter(default="vllm")  # vllm | hf (fallback for archs vLLM cannot pool)

    @modal.enter()
    def load(self):
        spec = BACKBONES[self.backbone]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=spec.get("trust_remote_code", False))
        if self.engine == "vllm":
            if spec.get("skip_lm_head"):
                _skip_lm_head_in_pooling_loader()
            from vllm import LLM
            kw = {k: v for k, v in spec.items() if k in ("quantization", "trust_remote_code")}
            if spec.get("new"):
                kw["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}  # text-only use of multimodal models
            self.llm = LLM(model=spec["repo"], runner="pooling", max_model_len=CAP + 1,
                           gpu_memory_utilization=0.88, **kw)
        else:
            import torch
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec["repo"], torch_dtype=torch.bfloat16, device_map="cuda",
                                                   trust_remote_code=spec.get("trust_remote_code", False)).eval()

    def _ids(self, texts, keep="head"):
        out = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            out.append(ids[-CAP:] if keep == "tail" else ids[:CAP])
        return out

    @modal.method()
    def embed(self, texts, keep="head"):
        """Last-token pooling, L2 normalized, float16: the recipe the published vectors use."""
        import numpy as np
        ids = self._ids(texts, keep)
        if self.engine == "vllm":
            from vllm.inputs import TokensPrompt
            res = self.llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
            v = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in res])
        else:
            import torch
            vs = []
            with torch.no_grad():
                for x in ids:
                    h = self.model(torch.tensor([x], device="cuda"), output_hidden_states=True).hidden_states[-1]
                    vs.append(h[0, -1].float().cpu().numpy())
            v = np.stack(vs)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v.astype(np.float16)

    @modal.method()
    def latency(self, lengths=(64, 256, 1024), reps=30):
        out = {}
        for n in lengths:
            text = " ".join(["state"] * n)
            for _ in range(3):
                self.embed.local([text])
            t = time.perf_counter()
            for _ in range(reps):
                self.embed.local([text])
            out[n] = round((time.perf_counter() - t) / reps * 1000, 2)
        return out


def _save(root, shard, idx, q, a, meta, n=None, ncnt=None):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    d = f"{root}/qa/{shard}"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/chunk_{idx:04d}_q.npy", q); np.save(f"{d}/chunk_{idx:04d}_a.npy", a)
    pq.write_table(pa.table(meta), f"{d}/chunk_{idx:04d}_meta.parquet")
    if n is not None:
        nd = f"{root}/neg/{shard}"; os.makedirs(nd, exist_ok=True)
        np.save(f"{nd}/chunk_{idx:04d}_n.npy", n); np.save(f"{nd}/chunk_{idx:04d}_ncnt.npy", ncnt)


def _pad_negs(lists, emb_fn, dim, K=10):
    import numpy as np
    flat = [t for l in lists for t in l[:K]]
    E = emb_fn(flat) if flat else np.zeros((0, dim), np.float16)
    n = np.zeros((len(lists), K, E.shape[1] if len(E) else dim), np.float16)
    cnt = np.zeros(len(lists), np.uint8); k = 0
    for i, l in enumerate(lists):
        m = min(len(l), K); n[i, :m] = E[k:k + m]; cnt[i] = m; k += m
    return n, cnt


@app.function(image=image, volumes={"/data": vol}, timeout=24 * 3600, cpu=4, memory=64000)
def run_job(job, backbone, src, chunk=100000, engine="vllm", state_format="{context}\n\n{question}"):
    import numpy as np
    vol.reload()
    E = Embedder(backbone=backbone, engine=engine)
    emb = lambda texts, keep="head": np.concatenate([E.embed.remote(texts[i:i + 4096], keep) for i in range(0, len(texts), 4096)])
    root = f"/data/emb/{backbone}"
    if job == "qa":
        for split in ("train", "test"):
            rows = [json.loads(l) for l in open(f"{src}/qa_text_{split}.jsonl")]
            states = [state_format.format(**r) for r in rows]
            for c, i in enumerate(range(0, len(rows), chunk)):
                part = rows[i:i + chunk]
                q, a = emb(states[i:i + chunk], "tail"), emb([r["answer"] for r in part])
                meta = {"doc_id": [r["doc_id"] for r in part], "pair_idx": [r["pair_idx"] for r in part]}
                if split == "train":
                    _save(root, "shard_00", c, q, a, meta)
                else:
                    n, cnt = _pad_negs([r.get("negatives", []) for r in part], emb, q.shape[1])
                    _save(root, "shard_00", 28 + c, q, a, meta, n, cnt)
                    _save(root, "shard_05", 83 + c, q, a, meta)
                vol.commit(); print(split, c, len(part), flush=True)
    elif job == "choice":
        for split, shard in (("train", "shard_10"), ("test", "shard_11")):
            rows = [json.loads(l) for l in open(f"{src}/choice_{split}.jsonl")]
            for c, i in enumerate(range(0, len(rows), chunk)):
                part = rows[i:i + chunk]
                q = emb([r["state"] for r in part], "tail")
                a = emb([r["options"][r["gold"]] for r in part])
                negs = [[o for j, o in enumerate(r["options"]) if j != r["gold"]] for r in part]
                n, cnt = _pad_negs(negs, emb, q.shape[1])
                meta = {"doc_id": [f"{r['source']}:{i + j}" for j, r in enumerate(part)], "source": [r["source"] for r in part]}
                _save(root, shard, c, q, a, meta, n, cnt)
                vol.commit(); print(split, c, len(part), flush=True)
    return root


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def fetch_heads():
    """The authors' three released heads, for warm starts and for scoring quantized backbones."""
    import shutil
    from huggingface_hub import hf_hub_download
    os.makedirs("/data/heads", exist_ok=True)
    for f in ("qwen3_8b_pretrained_head.pt", "qwen3_8b_midtrained_head.pt", "qwen3_8b_posttrained_head.pt"):
        shutil.copyfile(hf_hub_download("jackyk02/qwen3_8b_head", f), "/data/heads/" + f)
    vol.commit()
    return sorted(os.listdir("/data/heads"))


@app.local_entrypoint()
def main(job: str = "qa", backbone: str = "qwen3-8b", src: str = "", engine: str = "vllm"):
    if job == "heads":
        print(fetch_heads.remote())
        return
    if job == "latency":
        out = {}
        for b in backbone.split(","):
            out[b] = Embedder(backbone=b, engine=engine).latency.remote()
            print(b, out[b], flush=True)
        print(json.dumps(out, indent=1))
        return
    src = src or {"qa": "/data/qa_text", "choice": "/data/choice"}[job]
    print(run_job.remote(job, backbone, src, engine=engine))
