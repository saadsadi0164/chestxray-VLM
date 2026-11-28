# Directive: Multi-Dataset Anti-Hallucination VLM Training

## Goal
Train `HuggingFaceTB/SmolVLM-500M-Instruct` across diverse multi-institutional chest X-ray datasets covering 25 thoracic etiologies and findings while strictly enforcing the 5-Rule Clinical Rulebook and eliminating "Normal-biased" and laterality hallucinations.

## Why Current Weights Need Multi-Dataset Retraining
- `./chestxray_smolvlm_lora` was trained exclusively on Indiana University (OpenI) outpatient studies, which skew heavily normal (~70-80%) and lack bounding-box spatial annotations or broad acute emergency coverage (`Pneumothorax`, `Mass`, `Fracture`, `Lung Cavity`, `Edema`).
- Training previously targeted only LLM decoder projections (`q/k/v/o/gate/up/down_proj`), leaving the ViT `out_proj` and `connector` modality projector frozen.

## Execution Tools (Layer 3)
1. **Canonical Preprocessor:** [`execution/preprocess.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/preprocess.py)
2. **Dataset Synthesizer:** [`execution/build_rulebook_dataset.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_rulebook_dataset.py)
3. **Rulebook QLoRA/LoRA Trainer:** [`execution/train_rulebook_vlm.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/train_rulebook_vlm.py)
4. **Negation-Aware Validation & Benchmark Tools:** [`execution/report_parser.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/report_parser.py), [`execution/benchmark_baseline.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/benchmark_baseline.py), [`execution/compare_runs.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/compare_runs.py)

## Supported Datasets (25 Unified Thoracic Classes)
- **ChestX-Det** (`--chestx-det`): HuggingFace `natealberti/ChestX-Det` — verified 0-indexed mask IDs `0..9, 11` (skipping upstream-corrupted ID `12`) plus verified 1-to-1 official Deepwise JSON box recovery.
- **PadChest-GR** (`--padchest-gr`): HuggingFace `JasonZZ0601/PadChest-GR-Xray` — study-grouped findings, normalized anatomical zones, and strict Normal validation.
- **CheXpert** (`--chexpert-hf`): HuggingFace `danjacobellis/chexpert` — frontal/AP/PA studies (`Frontal/Lateral == 0`) with ClassLabel integers `0=unlabeled, 1=uncertain, 2=absent, 3=present` (NOT `-1`).
- **SIIM-Pneumothorax** (`--siim-pneumo`): HuggingFace `nwirandx/siim-pneumothorax-256-2400` — localized connected-component pneumothorax masks.
- **VinDr-CXR** (`--vindr-csv`, `--vindr-images`): 22 critical findings + physician bounding boxes.
- **Custom / NIH / MIMIC-CXR / RSNA / OpenI** (`--custom-jsonl`): Normalized JSONL manifest with `image_path` and `findings: [{label, bbox, zone}]`.

## Standard Operating Workflow

### Step 1: Build Balanced 5-Rule CoT Dataset
```bash
source .venv/bin/activate
python execution/build_rulebook_dataset.py \
  --chestx-det \
  --padchest-gr \
  --chexpert-hf \
  --max-per-source 3000 \
  --max-normal-ratio 0.22 \
  --min-class-count 250 \
  --out-dir ./.tmp/rulebook_dataset
```
*(Add `--vindr-csv` / `--vindr-images` or `--custom-jsonl` when running on Kaggle/Cloud GPU with full datasets.)*

### Step 2: Train on Cloud GPU (4-bit NF4 QLoRA + Vision/Projector Adaptation)
```bash
python execution/train_rulebook_vlm.py \
  --train-jsonl ./.tmp/rulebook_dataset/train.jsonl \
  --val-jsonl ./.tmp/rulebook_dataset/val.jsonl \
  --output-dir ./chestxray_smolvlm_lora_v2 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --quantize-4bit \
  --adapt-vision
```

### Step 3: Local Smoke Test (8 GB Apple Silicon Mac)
Do **not** run full multi-dataset training on an 8 GB MacBook (disk/RAM bound). Use `--smoke-sample` and `--dry-run-steps 1` locally only to verify code integrity:
```bash
python execution/build_rulebook_dataset.py --smoke-sample --out-dir ./.tmp/smoke_ds
python execution/train_rulebook_vlm.py \
  --train-jsonl ./.tmp/smoke_ds/train.jsonl \
  --val-jsonl ./.tmp/smoke_ds/val.jsonl \
  --output-dir ./.tmp/smoke_ckpt \
  --dry-run-steps 1 --adapt-vision --grad-accum 1
```

## Anti-Hallucination Training Safeguards
1. **Canonical Aspect-Preserving Inputs:** Both train and validation datasets load inputs through `execution/preprocess.normalize_cxr_to_rgb` (letterboxed `512x512` RGB, 1st–99th percentile contrast stretch, never horizontally flipped).
2. **Assistant-Only Loss Masking & Token Weighting:** Prompt/system/`<image>` tokens are masked (`labels[:prompt_len] = -100`), with decision-token weighting (`boiler=0.4x`, `findings=6.0x`, `decision=8.0x`) and training-only visual connector `AUX_CLASSES` multi-label supervision (`--aux-loss-weight 0.5`).
3. **Class Rebalancing:** `--max-normal-ratio 0.20` and `--min-class-count 300` prevent majority-normal collapse.
4. **3-Mode Authenticator Mix:** Emits Pure-Image CoT, CAD-Aligned Verification, and frequency-matched CAD-Rejection hard negatives with identical prompt phrasing.
5. **Negation-Aware Checkpoint Selection:** Per-epoch pure-mode validation (`--eval-val-samples 200`, `max_eval=2` for `--dry-run-steps > 0`) uses `execution/report_parser` (`compute_clinical_metrics`) to save best checkpoints by `combined_score` (logging both `combined_score` and `diag_val_loss`).
6. **Held-Out Hygiene:** Never train or tune on any patient in `.tmp/reports/test_manifest.jsonl` / `.tmp/reports/heldout_patient_ids.json`.
