"""Three T-Rex runs stacked top to bottom, same seed, from the real frames run_trex.py recorded.

Reuses the drawing code of pipeline/render_game.py unchanged (runner, obstacle, panel); only the
layout (three panels instead of two), the time window and the outputs differ.

    modal run innovations/render_trex3.py --probe            # crash times per head and seed
    modal run innovations/render_trex3.py --seed 3 --start 6.5 --seconds 18

Outputs (written to /data/renders/ on the clm-data volume, then downloaded to assets/media/):
  trex-three-heads.mp4   H.264, yuv420p, 30 fps, 1200 px wide
  trex-three-heads.gif   15 fps, 600 px wide
  trex-three-heads.png   a still just after the middle head's first crash in the window
  trex-three-heads.json  run tags, seed, window, crash times inside the window
"""
import gzip
import json
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "fonts-dejavu-core").pip_install("pillow==11.0.0")
         .add_local_file(os.path.join(os.path.dirname(HERE), "pipeline", "render_game.py"), "/root/render_game.py"))
vol = modal.Volume.from_name("clm-data", version=2)
app = modal.App("clm-render-trex3")
FPS = 60  # the game steps run_trex.py recorded per second
PANELS = [("authors", "CLM-8B (original)"),
          ("choice_hn_frompost", "S1-Router v0.1 (post-trained, no blend)"),
          ("wise_0.15", "S1-Tweaks v0.1 (85% CLM-8B + 15% S1-Router)")]
NAME = "trex-three-heads"


def load(tag):
    return json.load(gzip.open(f"/data/trex/{tag}/shield.json.gz", "rt"))


def frames_for(runs, seed):
    """All recorded frames of one seed, in order (a seed may span several Game objects)."""
    out = []
    for r in runs:
        if r["seed"] == seed:
            out += r["frames"]
    return out


def crash_times(frames):
    """Seconds at which the crashed flag switches on."""
    return [round(i / FPS, 2) for i in range(len(frames)) if frames[i][2] and (i == 0 or not frames[i - 1][2])]


@app.function(image=image, volumes={"/data": vol}, cpu=2, memory=8000, timeout=600)
def probe():
    vol.reload()
    out = {}
    for tag, _ in PANELS:
        runs = load(tag)
        out[tag] = {"records_per_seed": {}, "crashes": {}}
        for s in sorted({r["seed"] for r in runs}):
            fr = frames_for(runs, s)
            out[tag]["records_per_seed"][s] = [len(r["frames"]) for r in runs if r["seed"] == s]
            out[tag]["crashes"][s] = crash_times(fr)
    return out


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16000, timeout=1800)
def render(seed, start, seconds, still_at):
    sys.path.insert(0, "/root")
    from PIL import Image, ImageFont
    import render_game as rg
    f, fb = ImageFont.truetype(rg.FONT, 13), ImageFont.truetype(rg.BOLD, 15)
    vol.reload()
    seqs = [frames_for(load(tag), seed) for tag, _ in PANELS]
    a, b = int(start * FPS), int((start + seconds) * FPS)
    b = min([b] + [len(s) for s in seqs])
    out = f"/tmp/{NAME}"
    os.makedirs(out, exist_ok=True)
    gap, ph = 6, rg.H + 48
    sub = f"safety shield on, seed {seed}"
    for i, t in enumerate(range(a, b, 2)):  # 30 fps
        im = Image.new("RGB", (rg.W, 3 * ph + 2 * gap), (220, 216, 205))
        for k, ((tag, title), s) in enumerate(zip(PANELS, seqs)):
            im.paste(rg.panel(s[t], title, sub, t, f, fb), (0, k * (ph + gap)))
        im.save(f"{out}/{i:05d}.png")
    n = (b - a + 1) // 2
    os.makedirs("/data/renders", exist_ok=True)
    mp4, gif, png = (f"/data/renders/{NAME}.{x}" for x in ("mp4", "gif", "png"))
    subprocess.run(f"ffmpeg -y -loglevel error -framerate 30 -i {out}/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 "
                   f"-vf scale=1200:-2:flags=neighbor -movflags +faststart {mp4}", shell=True, check=True)
    for colors in (32, 16, 8):  # keep the GIF under 4 MB
        subprocess.run(f"ffmpeg -y -loglevel error -framerate 30 -i {out}/%05d.png -vf "
                       f"'fps=15,scale=600:-1:flags=neighbor,split[s0][s1];[s0]palettegen=max_colors={colors}[p];[s1][p]paletteuse=dither=none' {gif}",
                       shell=True, check=True)
        if os.path.getsize(gif) < 4_000_000:
            break
    k = min(n - 1, max(0, int((still_at - start) * 30)))
    subprocess.run(f"cp {out}/{k:05d}.png {png}", shell=True, check=True)
    meta = {"source": "innovations/render_trex3.py, frames recorded by pipeline/run_trex.py (official T-Rex courses, 60 s realtime, 60 steps/s)",
            "panels": [{"tag": tag, "title": title} for tag, title in PANELS], "seed": seed, "shield": True,
            "start_s": round(a / FPS, 2), "end_s": round(b / FPS, 2), "still_at_s": round(start + k / 30, 2),
            "crashes_in_window_s": {tag: [c for c in crash_times(s) if a / FPS <= c < b / FPS] for (tag, _), s in zip(PANELS, seqs)},
            "crashes_whole_run_s": {tag: crash_times(s) for (tag, _), s in zip(PANELS, seqs)},
            "frames": n, "gif_colors": colors}
    json.dump(meta, open(f"/data/renders/{NAME}.json", "w"), indent=1)
    vol.commit()
    return {"meta": meta, "bytes": {x: os.path.getsize(f"/data/renders/{NAME}.{x}") for x in ("mp4", "gif", "png")}}


@app.local_entrypoint()
def main(probe_only: bool = False, seed: int = -1, start: float = -1.0, seconds: float = 18.0, still_after: float = 0.3):
    if probe_only or seed < 0:
        print(json.dumps(probe.remote(), indent=1))
        return
    print(json.dumps(render.remote(seed, start, seconds, start + 4.0 + still_after), indent=1))
