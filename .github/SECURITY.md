# Security

## What to know before loading weights

A `.pt` file is a Python pickle. Loading one from a source you do not trust can run code on your machine.

- The scorer, the blend script and the demos (`innovations/scorer.py`, `innovations/interpolate.py`,
  `examples/demo_cpu.py`, `examples/route_text.py`) read heads with `torch.load(weights_only=True)`.
  That mode accepts tensors and plain Python types only. It lowers the risk and it is not a proof:
  torch versions before 2.6 had a flaw in it, which is why this package asks for torch 2.6 or newer.
- A fresh download of `s1_tweaks_v0.1.pt` or `s1_router_v0.1.pt` from the release page is checked against
  the sha256 written in `innovations/scorer.py` and in `MODEL_CARD.md`. A file that does not match is not kept.
- The training scripts (`innovations/train.py`, `innovations/heads.py`, parts of `pipeline/`) use
  `weights_only=False`. Only pass them checkpoints you
  made yourself or that come from a source you trust, such as the official CLM repositories on Hugging Face.

## Reporting a problem

Please do not open a public issue for a security problem.

Use GitHub's private reporting instead: open the **Security** tab of this repository and choose
**Report a vulnerability**. Say what you found, how to reproduce it and which commit you tested.

Good things to report: a way to make the loader run code from a head file, a download that is not
checked, a leaked secret in the history. Problems in the CLM engine itself belong with
[the CLM authors](https://github.com/Contrastive-LM/CLM).
