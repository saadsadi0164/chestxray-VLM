#!/usr/bin/env bash
set -euo pipefail

# ChestXStudio-VLM Phase 4 Reproducible Kaggle T4 Workflow (v3.1)
# Override cache/output paths cleanly via environment variables when needed.
IMG_CACHE_DIR="${IMG_CACHE_DIR:-/kaggle/tmp/training_images}"
DATASET_DIR="${DATASET_DIR:-/kaggle/tmp/rulebook_v3_dataset}"
SMOKE_DIR="${SMOKE_DIR:-/kaggle/tmp/v3_smoke_ckpt}"
OUT_DIR="${OUT_DIR:-/kaggle/working/chestxray_smolvlm_v3_lora}"
REPORTS_DIR="${REPORTS_DIR:-/kaggle/working/reports}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${REPORTS_DIR}"
for f in ./.tmp/reports/test_manifest.jsonl ./.tmp/reports/heldout_patient_ids.json ./.tmp/reports/baseline_v1.json ./.tmp/reports/baseline_v1_phase1.json; do
  if [[ -f "${f}" && ! -f "${REPORTS_DIR}/$(basename "${f}")" ]]; then
    cp "${f}" "${REPORTS_DIR}/$(basename "${f}")"
  fi
done

echo "=== [Step 1/5] Ensuring fixed 300-study held-out test set & patient exclusion list ==="
if [[ ! -f "${REPORTS_DIR}/test_manifest.jsonl" || ! -f "${REPORTS_DIR}/heldout_patient_ids.json" ]]; then
  "${PYTHON_BIN}" execution/build_heldout_test_set.py --out-dir "${REPORTS_DIR}" --normal 100 --abnormal 200 --seed 42
else
  echo "Using existing ${REPORTS_DIR}/test_manifest.jsonl and ${REPORTS_DIR}/heldout_patient_ids.json"
fi

echo "=== [Step 2/5] Building leak-free per-source balanced v3 training dataset ==="
"${PYTHON_BIN}" execution/build_rulebook_dataset.py \
  --chestx-det \
  --padchest-gr \
  --chexpert-hf \
  --siim-pneumo \
  --max-per-source 3500 \
  --max-padchest 1500 \
  --exclude-patients-json "${REPORTS_DIR}/heldout_patient_ids.json" \
  --format-version v3 \
  --pure-ratio 0.80 \
  --cad-aligned-ratio 0.10 \
  --max-normal-ratio 0.20 \
  --min-class-count 400 \
  --strict-min-class \
  --image-size 512 \
  --img-cache-dir "${IMG_CACHE_DIR}" \
  --out-dir "${DATASET_DIR}"

echo "=== [Step 3/5] Running 5-step T4 sanity smoke check ==="
"${PYTHON_BIN}" execution/train_rulebook_vlm.py \
  --train-jsonl "${DATASET_DIR}/train.jsonl" \
  --val-jsonl "${DATASET_DIR}/val.jsonl" \
  --output-dir "${SMOKE_DIR}" \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --connector-lr 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --adapt-vision \
  --aux-loss-weight 1.0 \
  --findings-weight 5.0 \
  --image-size 512 \
  --num-workers 2 \
  --image-splitting \
  --dry-run-steps 5

echo "=== [Step 4/5] Launching full 4-epoch v3 training (best checkpoint by combined_score) ==="
"${PYTHON_BIN}" execution/train_rulebook_vlm.py \
  --train-jsonl "${DATASET_DIR}/train.jsonl" \
  --val-jsonl "${DATASET_DIR}/val.jsonl" \
  --output-dir "${OUT_DIR}" \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --connector-lr 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --adapt-vision \
  --aux-loss-weight 1.0 \
  --findings-weight 5.0 \
  --image-size 512 \
  --num-workers 2 \
  --image-splitting \
  --eval-val-samples 200

echo "=== [Step 5/5] Evaluating trained v3 adapter on fixed 300-study held-out manifest ==="
"${PYTHON_BIN}" execution/benchmark_baseline.py \
  --manifest "${REPORTS_DIR}/test_manifest.jsonl" \
  --out "${REPORTS_DIR}/v3_test.json" \
  --adapter "${OUT_DIR}" \
  --mode pure_vlm \
  --save-every 25 \
  --overwrite

"${PYTHON_BIN}" execution/compare_runs.py --reports-dir "${REPORTS_DIR}" --out "${REPORTS_DIR}/comparison.md"
echo "Done. Check ${REPORTS_DIR}/comparison.md and Acceptance Gates (NormBias<15%, MacroSens>=0.60, FA<25%, Comp>=95%)."
