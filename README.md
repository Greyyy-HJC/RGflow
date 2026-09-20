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
