## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it was checked

<!-- The commands you ran and what they printed. For a new result: hardware, seeds, test size, wall-clock and cost. -->

## Checklist

- [ ] `python -m pytest -q tests` passes.
- [ ] Every new or changed number in the docs points to a file under `results/`, and that file is in this pull request.
- [ ] If a result file changed, the figures were regenerated with `python scripts/make_figures.py`.
- [ ] Limits are stated next to the claim: seeds, sample size, and whether the test is held-out or in-distribution.
- [ ] No weights, vectors or secrets are committed (`*.pt` and `*.npy` are ignored on purpose).
