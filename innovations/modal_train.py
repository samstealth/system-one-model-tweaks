"""Run innovations.train on a Modal GPU against the clm-data volume. Any train.py flag passes through.

    # full-scale embedding-dimension and shared-space runs on the real 62M-pair store
    modal run --detach innovations/modal_train.py --args "--root /data --out /data/runs/mrl --proj 1024 --matryoshka 64,128,256,512,1024 --batch 16384 --steps 30000"
    modal run --detach innovations/modal_train.py --args "--root /data --out /data/runs/shared --tower shared_adapter --batch 16384 --steps 30000"

    # innovation 5: listwise post-training from the authors' mid-trained head, 40% Q&A replay with hard negatives
    modal run --detach innovations/modal_train.py --args "--root /data/emb/qwen3-8b --chunks shard_10/0,shard_10/1,shard_10/2 --hard-negs \
        --init /data/heads/qwen3_8b_midtrained_head.pt --replay-root /data --replay-chunks shard_00/0,shard_00/1,shard_00/2 \
        --replay-negs --replay-frac 0.4 --choice-eval /data/emb/qwen3-8b --lr 0.0003 --steps 400 --batch 4096 --out /data/runs/post_choice"

    # innovation 1: same heads on a new backbone's vectors (made by modal_embed.py --job qa)
    modal run --detach innovations/modal_train.py --args "--root /data/emb/inkling-2-mini --out /data/runs/inkling_pre --steps 3000 --batch 4096"
"""
import os
import shlex

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0")
         .add_local_python_source("innovations"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-innovations-train")


@app.function(image=image, volumes={"/data": vol}, gpu=os.environ.get("CLM_GPU", "L40S"), cpu=16, memory=128000, timeout=86000)
def run(argv):
    from innovations import train
    vol.reload()
    out = train.main(argv)
    vol.commit()
    return out


@app.local_entrypoint()
def main(args: str):
    print(run.remote(shlex.split(args)))
