# Testing

Clean-checkout commands are the source of truth. Do not rely on packages
preinstalled on an agent image or a local `.venv`.

## Commands

Core suite (stdlib + pytest). Tests that need NumPy or Torch **skip**:

```bash
uv run --isolated --extra dev pytest -q
```

Full suite (pytest + NumPy + CPU Torch). This is the PR/push CI ML job:

```bash
uv run --isolated --extra full pytest -q
```

`--extra full` is `dev` + `ml` (see `pyproject.toml`). Equivalent:

```bash
uv run --isolated --extra dev --extra ml pytest -q
```

`pip` users can install `requirements-ml.txt` (CPU Torch index documented
there) plus `pytest` and run `pytest -q` from the repo root.

## Generated checkpoints

`*.pt` / `*.pth` are gitignored. Tests that need model bytes either:

1. **Build a deterministic temporary PolicyNet** (`tests/ml_testutil.py`
   `write_tiny_policy_checkpoint`), or
2. **Skip with rationale** when `ml/policy_bc.pt` or experiment
   `iter_000.pt` files are absent, and assert the **committed experiment
   JSON** hashes instead.

Do not commit trained weights. Do not weaken simulator-fidelity,
seed-integrity, model-fingerprint, or security assertions to make the
suite green.
