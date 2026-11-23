"""Fast CPU checks of innovations/scorer.py and the examples. No network: downloads are stubbed.
Run: python -m pytest -q tests"""
import hashlib
import importlib.util
import io
import json
import os
import pickle
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from innovations import interpolate, scorer
from innovations.heads import HeadConfig, TwoTower
from innovations.scorer import HeadNotAvailable, Scorer, blend, load_head

ROOT = Path(__file__).resolve().parents[1]
D = 32


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_head(path, seed=0, tied=False, in_dim=D):
    """A small random head saved in CLM's checkpoint layout. tied=True makes both sides identical."""
    torch.manual_seed(seed)
    m = TwoTower(HeadConfig(in_dim=in_dim, width=24, proj=8))
    if tied:
        m.action_head.load_state_dict(m.state_head.state_dict())
    m.export_authors(str(path))
    return m.eval()


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise AssertionError("tests must not touch the network")
    monkeypatch.setattr(scorer.urllib.request, "urlopen", refuse)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse)
    monkeypatch.delenv("S1_TWEAKS_HEAD", raising=False)
    monkeypatch.setenv("S1_TWEAKS_CACHE", str(tmp_path / "cache"))


def not_found(req, timeout=None):
    raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)


# ------------------------------------------------------------------------------------ scorer
def test_probs_are_softmax_of_scaled_cosine(tmp_path):
    m = tiny_head(tmp_path / "h.pt")
    s = load_head(tmp_path / "h.pt")
    assert (s.name, s.in_dim, s.proj_dim) == ("h", D, 8)
    rng = np.random.default_rng(0)
    q, o = rng.normal(size=D).astype(np.float32), rng.normal(size=(5, D)).astype(np.float32)
    p = s.probs(q, o)
    with torch.no_grad():
        zq = m.encode_state(F.normalize(torch.from_numpy(q), dim=-1)[None])[0]
        zo = m.encode_action(F.normalize(torch.from_numpy(o), dim=-1))
        ref = torch.softmax(m.scale() * zo @ zq, -1).numpy()
    assert p.shape == (5,) and abs(p.sum() - 1) < 1e-5
    assert np.allclose(p, ref, atol=1e-6)
    assert s.pick(q, o) == int(ref.argmax())
    assert np.allclose(s.probs(q * 7, o * 3), p, atol=1e-5)                 # inputs are L2 normalized for you
    assert np.allclose(s.probs(q.astype(np.float16), torch.from_numpy(o)), p, atol=1e-2)  # float16 and tensors work
    flat = s.probs(q, o, temperature=50.0)
    assert flat.max() < p.max() and abs(flat.sum() - 1) < 1e-5
    assert np.allclose(s.scale * s.embed_options(o) @ s.embed_state(q), s.logits(q, o), atol=1e-4)


def test_batch_padding_and_shared_options(tmp_path):
    tiny_head(tmp_path / "h.pt")
    s = load_head(tmp_path / "h.pt")
    rng = np.random.default_rng(1)
    N, K = 7, 6
    Q, O = rng.normal(size=(N, D)).astype(np.float16), rng.normal(size=(N, K, D)).astype(np.float16)
    n = np.array([2, 6, 3, 4, 5, 2, 6], np.uint8)
    for i in range(N):
        O[i, n[i]:] = 0  # padding rows, as in the sample file
    P = s.probs(Q, O, n)
    assert P.shape == (N, K) and np.allclose(P.sum(1), 1, atol=1e-5)
    for i in range(N):
        assert (P[i, n[i]:] == 0).all()
        assert np.allclose(P[i, :n[i]], s.probs(Q[i], O[i, :n[i]]), atol=1e-5)
    assert (s.pick(Q, O, n) == P.argmax(1)).all() and (s.pick(Q, O, n) < n).all()
    shared = s.probs(Q, O[1])  # one option list for every state
    assert shared.shape == (N, K) and np.allclose(shared[3], s.probs(Q[3], O[1]), atol=1e-5)


def test_input_checks(tmp_path):
    tiny_head(tmp_path / "h.pt")
    s = load_head(tmp_path / "h.pt")
    q, o = np.ones(D, np.float32), np.ones((3, D), np.float32)
    with pytest.raises(ValueError, match="expects 32-d backbone vectors, got 16-d"):
        s.probs(np.ones(16, np.float32), o)
    with pytest.raises(ValueError, match="option lists"):
        s.probs(np.ones((2, D)), np.ones((3, 4, D)))
    with pytest.raises(ValueError, match="n_opts"):
        s.probs(np.ones((2, D)), np.ones((2, 4, D)), [1, 5])
    with pytest.raises(ValueError, match="temperature"):
        s.probs(q, o, temperature=0)
    with pytest.raises(ValueError, match="got 3-d states"):
        s.probs(np.ones((1, 2, D)), o)


def test_default_layout_matches_clm_8b():
    s = Scorer(TwoTower(HeadConfig()), "default")
    assert s.n_params == 18_887_680 and (s.in_dim, s.proj_dim) == (4096, 512)
    assert sum(p.numel() for p in s.model.state_head.parameters()) == 9_443_840
    ck = s.checkpoint()
    assert set(ck) >= {"state_head", "action_head", "logit_scale", "cfg"}
    assert ck["cfg"] == {"hidden_size": 4096, "projection_dim": 512, "width": 1536, "depth": 3,
                         "activation": "gelu", "layernorm": True, "residual": False}
    assert "4096 -> 1536 -> 1536 -> 512" in s.describe()


def test_scale_is_capped_at_100(tmp_path):
    m = tiny_head(tmp_path / "h.pt")
    ck = torch.load(tmp_path / "h.pt", weights_only=True)
    ck["logit_scale"] = torch.tensor(4.6132)  # the value stored in CLM_v0.1-8B.pt; exp() is just above 100
    torch.save(ck, tmp_path / "hot.pt")
    assert load_head(tmp_path / "hot.pt").scale == 100.0
    assert abs(load_head(tmp_path / "h.pt").scale - float(m.scale().detach())) < 1e-6


def test_save_roundtrip_and_trainer_layout(tmp_path):
    tiny_head(tmp_path / "h.pt")
    s = load_head(tmp_path / "h.pt")
    q, o = np.random.default_rng(2).normal(size=(2, 4, D)).astype(np.float32)[:, 0], np.random.default_rng(3).normal(size=(4, D))
    s.save(tmp_path / "out" / "again.pt")
    assert np.allclose(load_head(tmp_path / "out" / "again.pt").probs(q, o), s.probs(q, o))
    m = TwoTower.from_authors(str(tmp_path / "out" / "again.pt"))  # the repo's own loader reads it too
    assert torch.allclose(m.state_head.out.weight, s.model.state_head.out.weight)
    shared = TwoTower(HeadConfig(in_dim=D, width=24, proj=8, tower="shared"))
    shared.save(str(tmp_path / "final.pt"))
    t = load_head(tmp_path / "final.pt")
    assert t.model.cfg.tower == "shared" and t.probs(q, o).shape == (2, 4)
    with pytest.raises(ValueError, match="two separate heads"):
        t.checkpoint()


def test_rejects_unknown_layouts_and_unsafe_files(tmp_path):
    tiny_head(tmp_path / "h.pt")
    ck = torch.load(tmp_path / "h.pt", weights_only=True)
    torch.save({**ck, "cfg": {**ck["cfg"], "residual": True, "activation": "relu"}}, tmp_path / "res.pt")
    with pytest.raises(ValueError, match="unsupported head layout .*activation=relu, residual=True"):
        load_head(tmp_path / "res.pt")
    torch.save({"weights": torch.zeros(2)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="state_head, action_head and logit_scale"):
        load_head(tmp_path / "other.pt")
    with open(tmp_path / "evil.pt", "wb") as f:  # a pickle that would run code under a plain torch.load
        pickle.dump({"state_head": types.SimpleNamespace(run=os.getcwd)}, f, protocol=2)
    with pytest.raises(ValueError, match="weights_only=True"):
        load_head(tmp_path / "evil.pt")
    with pytest.raises(FileNotFoundError, match="No head at"):
        load_head(tmp_path / "missing.pt")


def test_blend_matches_interpolate(tmp_path):
    tiny_head(tmp_path / "a.pt", seed=1)
    tiny_head(tmp_path / "b.pt", seed=2)
    a, b = load_head(tmp_path / "a.pt"), load_head(tmp_path / "b.pt")
    rng = np.random.default_rng(4)
    q, o = rng.normal(size=D), rng.normal(size=(5, D))
    assert np.allclose(blend(a, b, 0.0).probs(q, o), a.probs(q, o), atol=1e-6)
    assert np.allclose(blend(a, b, 1.0).probs(q, o), b.probs(q, o), atol=1e-6)
    w = blend(tmp_path / "a.pt", tmp_path / "b.pt", 0.15)  # paths work too
    assert w.name == "blend(a, b, 0.15)"
    wa, wb, ww = a.model.state_head.inp.weight, b.model.state_head.inp.weight, w.model.state_head.inp.weight
    assert torch.allclose(ww, 0.85 * wa + 0.15 * wb, atol=1e-7)
    assert abs(w.logit_scale - (0.85 * a.logit_scale + 0.15 * b.logit_scale)) < 1e-6
    interpolate.main(["--a", str(tmp_path / "a.pt"), "--b", str(tmp_path / "b.pt"), "--alpha", "0.15", "--out", str(tmp_path / "w.pt")])
    assert np.allclose(load_head(tmp_path / "w.pt").probs(q, o), w.probs(q, o), atol=1e-6)  # same as the CLI that made wise_0.15
    assert not torch.equal(a.model.state_head.inp.weight, ww) and torch.equal(wa, load_head(tmp_path / "a.pt").model.state_head.inp.weight)
    torch.manual_seed(0)
    other = Scorer(TwoTower(HeadConfig(in_dim=D, width=16, proj=8)), "narrow")
    with pytest.raises(ValueError, match="layouts differ"):
        blend(a, other, 0.5)


def test_s1_tweaks_before_the_release_is_live(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer.urllib.request, "urlopen", not_found)
    with pytest.raises(HeadNotAvailable) as e:
        load_head("s1-tweaks")
    msg = str(e.value)
    assert "not published yet" in msg and scorer.S1_TWEAKS_URL in msg and "HTTP 404" in msg
    assert "--head /path/to/s1_tweaks_v0.1.pt" in msg and "S1_TWEAKS_HEAD" in msg
    assert not list((tmp_path / "cache").glob("*"))  # no partial file left behind
    assert scorer.main(["s1-tweaks"]) == 1

    tiny_head(tmp_path / "local.pt")
    monkeypatch.setenv("S1_TWEAKS_HEAD", str(tmp_path / "local.pt"))  # a local copy wins over the download
    s = load_head("s1-tweaks")
    assert s.name == "S1-Tweaks v0.1" and s.source == str(tmp_path / "local.pt")
    monkeypatch.setenv("S1_TWEAKS_HEAD", str(tmp_path / "nope.pt"))
    with pytest.raises(HeadNotAvailable, match="not a file"):
        load_head("s1-tweaks")


def test_downloads_are_cached(tmp_path, monkeypatch):
    tiny_head(tmp_path / "remote.pt")
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return io.BytesIO((tmp_path / "remote.pt").read_bytes())  # BytesIO is a context manager with read()
    monkeypatch.setattr(scorer.urllib.request, "urlopen", fake_urlopen)
    s = load_head("S1-Tweaks")
    assert s.name == "S1-Tweaks v0.1" and calls == [scorer.S1_TWEAKS_URL]
    assert s.source == str(tmp_path / "cache" / "s1_tweaks_v0.1.pt") and not os.path.exists(s.source + ".part")
    load_head("s1-tweaks")
    assert len(calls) == 1  # second call reads the cache
    u = load_head("https://example.com/heads/my_head.pt?download=1")
    assert u.name == "my_head" and len(calls) == 2 and os.path.dirname(u.source) == str(tmp_path / "cache")

    os.remove(s.source)  # a fresh download is checked against the published checksum, once one is set
    monkeypatch.setattr(scorer, "S1_TWEAKS_SHA256", "0" * 64)
    with pytest.raises(HeadNotAvailable, match="does not match the published checksum"):
        load_head("s1-tweaks")
    assert not os.path.exists(s.source)
    monkeypatch.setattr(scorer, "S1_TWEAKS_SHA256", hashlib.sha256((tmp_path / "remote.pt").read_bytes()).hexdigest())
    assert load_head("s1-tweaks").source == s.source

    class Dropped(io.BytesIO):  # a connection that dies halfway leaves no partial file behind
        def read(self, n=-1):
            raise ConnectionResetError("connection reset")
    monkeypatch.setattr(scorer.urllib.request, "urlopen", lambda req, timeout=None: Dropped(b"x"))
    with pytest.raises(HeadNotAvailable, match="ConnectionResetError"):
        load_head("https://example.com/heads/dropped.pt")
    assert not list((tmp_path / "cache").glob("*.part")) and not list((tmp_path / "cache").glob("*dropped*"))

    def offline(req, timeout=None):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(scorer.urllib.request, "urlopen", offline)
    with pytest.raises(HeadNotAvailable, match="Could not download a head from https://example.com/other.pt"):
        load_head("https://example.com/other.pt")
    os.remove(s.source)
    with pytest.raises(HeadNotAvailable, match="Could not download S1-Tweaks v0.1 .*no route to host"):
        load_head("s1-tweaks")


def test_clm_and_hf_sources_use_the_hub_client(tmp_path, monkeypatch):
    tiny_head(tmp_path / "clm.pt")
    seen = []

    def fake_hub(repo, filename, revision=None):
        seen.append((repo, filename, revision))
        return str(tmp_path / "clm.pt")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hub)
    assert load_head("clm").name == "CLM-8B"
    assert load_head("hf://someone/some-repo/sub/dir/head_x.pt").name == "head_x"
    assert seen == [("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt", scorer.CLM_REVISION), ("someone/some-repo", "sub/dir/head_x.pt", None)]
    with pytest.raises(ValueError, match="hf://org/repo/file.pt"):
        load_head("hf://only/two")

    def hub_down(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub_down)
    with pytest.raises(HeadNotAvailable, match="Download it by hand from https://huggingface.co/Contrastive-LM/CLM-v0.1-8B"):
        load_head("clm")


# ------------------------------------------------------------------------------------ examples/demo_cpu.py
@pytest.fixture
def sample(tmp_path):
    """A synthetic sample plus two tiny heads: 'good' has identical sides, so it finds the gold option."""
    syn = example("make_synthetic_sample")
    syn.write_sample(str(tmp_path / "data"), n=40, k=5, dim=D, seed=0, noise=0.3)
    tiny_head(tmp_path / "good.pt", seed=1, tied=True)
    tiny_head(tmp_path / "rand.pt", seed=2)
    return tmp_path


def test_demo_cpu_on_a_synthetic_sample(sample, capsys):
    demo = example("demo_cpu")
    r = demo.main(["--data", str(sample / "data"), "--clm-head", str(sample / "good.pt"), "--head", str(sample / "rand.pt"), "--show", "2"])
    out = capsys.readouterr().out
    assert "SYNTHETIC sample" in out and "Loaded 40 decisions" in out
    assert out.count("--- decision") == 2 and "<- gold" in out and "* = that head's pick" in out
    for word in ("synthetic_tools", "synthetic_routing", "all", "chance", "good", "rand"):
        assert word in out
    z = np.load(sample / "data" / "sample_decisions.npz")
    rows = [json.loads(line) for line in open(sample / "data" / "sample_decisions.jsonl")]
    assert z["q"].dtype == np.float16 and z["opts"].shape == (40, 5, D) and z["n_opts"].dtype == np.uint8 and z["gold"].dtype == np.int64
    assert all(r_["synthetic"] and r_["source"].startswith("synthetic_") and len(r_["options"]) == n for r_, n in zip(rows, z["n_opts"]))
    for name in ("good", "rand"):  # the table equals an independent computation, one decision at a time
        s = load_head(sample / f"{name}.pt")
        hits = [s.pick(z["q"][i], z["opts"][i, :z["n_opts"][i]]) == z["gold"][i] for i in range(40)]
        assert abs(r["accuracy"][name]["all"] - np.mean(hits)) < 1e-9
        tools = [h for h, row in zip(hits, rows) if row["source"] == "synthetic_tools"]
        assert abs(r["accuracy"][name]["synthetic_tools"] - np.mean(tools)) < 1e-9
    assert r["accuracy"]["good"]["all"] >= 0.9 and r["n"] == 40 and r["seconds"] < 5


def test_demo_cpu_says_what_is_missing(tmp_path):
    demo = example("demo_cpu")
    with pytest.raises(SystemExit) as e:
        demo.main(["--data", str(tmp_path / "nowhere")])
    msg = str(e.value)
    assert "Sample data not found" in msg and "git pull" in msg and "--data /path/to/folder" in msg
    assert str(tmp_path / "nowhere" / "sample_decisions.npz") in msg and str(tmp_path / "nowhere" / "sample_decisions.jsonl") in msg


def test_demo_cpu_runs_with_clm_only_while_the_release_is_not_live(sample, capsys, monkeypatch):
    monkeypatch.setattr(scorer.urllib.request, "urlopen", not_found)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, filename, revision=None: str(sample / "good.pt"))
    demo = example("demo_cpu")
    r = demo.main(["--data", str(sample / "data"), "--disagreements"])
    out = capsys.readouterr().out
    assert "Skipping S1-Tweaks v0.1" in out and "not published yet" in out
    assert list(r["accuracy"]) == ["CLM-8B"] and "with 1 head in" in out and "--disagreements needs two heads" in out


def test_demo_cpu_rejects_a_broken_sample(sample, capsys):
    demo = example("demo_cpu")
    jsonl = sample / "data" / "sample_decisions.jsonl"
    lines = jsonl.read_text().splitlines()
    jsonl.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit, match="inconsistent.*39 text rows"):
        demo.main(["--data", str(sample / "data")])
    jsonl.write_text("\n".join(lines) + "\n")
    tiny_head(sample / "wide.pt", in_dim=2 * D)
    with pytest.raises(SystemExit, match="expects 64-d backbone vectors, got 32-d"):
        demo.main(["--data", str(sample / "data" / "sample_decisions.npz"), "--clm-head", str(sample / "wide.pt"), "--head", str(sample / "wide.pt")])
    (sample / "data" / "sample_decisions.npz").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(SystemExit, match="git lfs pull"):
        demo.main(["--data", str(sample / "data")])


def test_demo_cpu_as_a_script_from_another_folder(sample):
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "demo_cpu.py"), "--data", str(sample / "data"),
                        "--clm-head", str(sample / "good.pt"), "--head", str(sample / "rand.pt"), "--show", "1", "--disagreements"],
                       cwd=str(sample), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Top-1 accuracy on this sample (40 decisions)" in r.stdout


def test_synthetic_sample_never_lands_in_the_real_data_folder(tmp_path):
    syn = example("make_synthetic_sample")
    with pytest.raises(SystemExit, match="Refusing to write synthetic data into examples/data"):
        syn.write_sample(str(ROOT / "examples" / "data"))
    npz, jsonl = syn.main(["--out", str(tmp_path / "s"), "--n", "6", "--k", "3", "--dim", "8"])
    assert os.path.dirname(npz) == str(tmp_path / "s") and np.load(npz)["q"].shape == (6, 8)


# ------------------------------------------------------------------------------------ examples/route_text.py
def test_route_text_formats_and_pooling():
    rt = example("route_text")
    assert rt.state_text(" ctx ", "question?") == "ctx\n\nquestion?" and rt.state_text("ctx", None) == "ctx"
    assert rt.option_text("billing", "Charges and refunds") == "billing: Charges and refunds" and rt.option_text("billing", "") == "billing"
    assert rt.parse_options(["a: one: two", "b : three", "bare"]) == {"a": "one: two", "b": "three", "bare": ""}
    with pytest.raises(SystemExit, match="at least two"):
        rt.parse_options(["only: one"])

    calls = []

    def tok(text, add_special_tokens=True):
        calls.append(add_special_tokens)
        return {"input_ids": [ord(c) % 50 for c in text]}

    assert rt.token_ids(tok, "abcdef", "tail", cap=4) == [ord(c) % 50 for c in "cdef"]   # a state keeps its tail
    assert rt.token_ids(tok, "abcdef", "head", cap=4) == [ord(c) % 50 for c in "abcd"]   # an option keeps its head
    assert rt.token_ids(tok, "") == [ord(" ") % 50] and not any(calls)                   # never special tokens

    class Backbone(torch.nn.Module):  # stands in for Qwen: returns per-token hidden states for two "layers"
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(50, 16)

        def forward(self, input_ids, output_hidden_states=False):
            h = self.emb(input_ids)
            return types.SimpleNamespace(hidden_states=(h, 3 * h.cumsum(1)))

    bb = Backbone()
    v = rt.embed(bb, tok, ["hello world", "abc"], "head", "cpu")
    assert v.shape == (2, 16) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-6)
    ids = torch.tensor([[ord(c) % 50 for c in "hello world"]])
    ref = F.normalize((3 * bb.emb(ids).cumsum(1))[0, -1], dim=-1).detach().numpy()  # last token, final layer, unit norm
    assert np.allclose(v[0], ref, atol=1e-6)
    h = rt.random_head(16)
    assert h.in_dim == 16 and h.name.startswith("RANDOM") and abs(h.probs(v[0], v).sum() - 1) < 1e-5
    assert [x.name for x in rt.load_heads(["random", "random"], 16)] == ["RANDOM head (untrained)", "RANDOM head (untrained) (2)"]
    with pytest.raises(SystemExit, match="expects 16-d vectors but this backbone produces 4096-d"):
        rt.load_heads([h], 4096)  # a head only works with the backbone it was trained against
