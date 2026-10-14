#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run="$PWD/artifacts/4dsu3/upsample/L12_to_L24_2flow_stable_v3"
phase=starting
trap 'status=$?; printf "%s: exit %s\n" "$phase" "$status" > "$run/pipeline_status.txt"' EXIT
cp "$run/numerics.json" "$run/initial_numerics.json"
if [[ ! -f "$run/offload_validation.json" ]]; then
    phase=memory_validation
    printf '%s\n' "$phase" > "$run/pipeline_status.txt"
    .venv/bin/python scripts/upsample/check_training_memory.py --run "$run" > "$run/offload_validation.log" 2>&1
fi
phase=flow1_training
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/train.py --output "$run" --resume --epochs 0 --smeared-moment-epochs 20 --transport-epochs 0 --moment-epochs 0 --learning-rate .003 --mean-metric diagonal --variance-weight 10 --require-stable-inverse --stability-coarse-file "$run/numerical_sample.pt" --skip-evaluation >> "$run/training.log" 2>&1
phase=flow1_calibration
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/calibrate.py --run "$run" --stage 1 --bound .3 --variance-weight 10 --require-stable-inverse --stability-coarse-file "$run/numerical_sample.pt" > "$run/calibration_stage1.log" 2>&1
phase=flow2_training
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/train.py --output "$run" --resume --epochs 0 --smeared-moment-epochs 0 --transport-epochs 0 --moment-epochs 20 --learning-rate .003 --mean-metric diagonal --variance-weight 10 --require-stable-inverse --stability-coarse-file "$run/numerical_sample.pt" --skip-evaluation >> "$run/training.log" 2>&1
phase=flow2_calibration
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/calibrate.py --run "$run" --stage 2 --bound .5 --variance-weight 10 --require-stable-inverse --stability-coarse-file "$run/numerical_sample.pt" > "$run/calibration_stage2.log" 2>&1
phase=validation
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/validate.py --run "$run" --full-lattice-sample "$run/numerical_sample.pt" > "$run/validation.log" 2>&1
phase=evaluation
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/train.py --output "$run" --evaluate-only --pyquda-check > "$run/evaluation.log" 2>&1
phase=notebook
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python scripts/upsample/make_notebook.py --run "$run" > "$run/notebook.log" 2>&1
phase=document
printf '%s\n' "$phase" > "$run/pipeline_status.txt"
.venv/bin/python docs/project_notes/make_figures.py --refresh-data --upsampling-run "$run"
.venv/bin/python docs/project_notes/update_results.py --upsampling-run "$run"
cd docs/project_notes
../../temp/latex-toolchain/tectonic --synctex --keep-logs jinchen_progress.tex
cd ../..
phase=complete
