# RGflow: 4D SU(3)

This branch is the starting point for the 4D SU(3) lattice gauge-theory
study. The repository is intentionally focused on this theory rather than
providing a framework for several gauge groups.

## Environment

Create and use the virtual environment at the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/pytest
```

The root `.venv/` directory is ignored by Git. There is no committed
`uv.lock`; use the Python environment directly for this project.

## Layout

- `docs/` — papers, notes, and analysis documentation
- `presentation/` — notebooks and presentation material
- `src/` — 4D SU(3) implementation
- `temp/` — local scratch data and generated intermediates
- `tests/` — automated tests
- `plan.md` — the current 4D SU(3) research plan

The implementation will be added under `src/rgflow/su3/` as the study is
developed.

## Naive SU(3) downsampling

Install the optional Torch training dependency and run the L24 to L12
experiment with:

```bash
.venv/bin/python -m pip install -e '.[test,train,report]'
.venv/bin/python scripts/downsample/train.py --backend compile
```

Use `--max-configs 5 --epochs 2` for a small smoke run. Results are written
under `artifacts/4dsu3/downsample/`, including the CNN checkpoint, feature
cache, mean-plus-variance metrics, and diagnostic plot. Compare eager and
compiled kernel throughput with:

```bash
.venv/bin/python scripts/downsample/benchmark.py --backend both
```

For a completed checkpoint, `scripts/downsample/evaluate.py` performs the
eager final evaluation and PyQUDA cross-check without recompiling the training
graph. This is useful because the complex SVD projection dominates runtime.
