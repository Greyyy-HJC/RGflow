#!/usr/bin/env python3
"""Point the presentation notebook at a completed run and execute it."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
run = args.run.resolve()
files = ['metrics.json', 'kernel.json', 'observable_distributions.npz', 'fit_history.json']
for name in files:
    if not (run/name).is_file():
        raise FileNotFoundError(run/name)
relative_run = os.path.relpath(run, ROOT)
path = ROOT/'presentation/downsample_stout.ipynb'
notebook = nbformat.read(path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == 'code' and 'RGFLOW_DOWNSAMPLE_RUN' in cell.source:
        if '%matplotlib inline' not in cell.source:
            cell.source = '%matplotlib inline\n' + cell.source
        lines = cell.source.splitlines()
        cell.source = '\n'.join(
            f"RUN = Path(os.environ.get('RGFLOW_DOWNSAMPLE_RUN', ROOT / {relative_run!r}))"
            if line.startswith('RUN = ') else line for line in lines)
        cell.source = '\n'.join(line for line in cell.source.splitlines()
                                if not line.startswith("print('Source run:'"))
        cell.source += "\nprint('Source run:', RUN)"
manager = KernelManager(kernel_name='python3', transport='ipc',
                        ip=str(ROOT/'temp/downsample_notebook_kernel'))
manager.kernel_spec.argv = [str(ROOT/'.venv/bin/python'), '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(notebook, timeout=300, km=manager, resources={'metadata': {'path': str(ROOT)}}).execute()
assert any('image/png' in output.get('data', {}) for cell in notebook.cells
           for output in cell.get('outputs', [])), 'Notebook contains no embedded figures'
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(path)
