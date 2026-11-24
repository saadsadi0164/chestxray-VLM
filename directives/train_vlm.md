# Directive: Multi-Dataset Anti-Hallucination VLM Training

## Goal
Train `HuggingFaceTB/SmolVLM-500M-Instruct` across diverse multi-institutional chest X-ray datasets covering 25 thoracic etiologies and findings while strictly enforcing the 5-Rule Clinical Rulebook and eliminating "Normal-biased" and laterality hallucinations.

## Why Current Weights Need Multi-Dataset Retraining
- `./chestxray_smolvlm_lora` was trained exclusively on Indiana University (OpenI) outpatient studies, which skew heavily normal (~70-80%) and lack bounding-box spatial annotations or broad acute emergency coverage (`Pneumothorax`, `Mass`, `Fracture`, `Lung Cavity`, `Edema`).
- Training previously targeted only LLM decoder projections (`q/k/v/o/gate/up/down_proj`), leaving the ViT `out_proj` and `connector` modality projector frozen.

## Execution Tools (Layer 3)
1. **Dataset Synthesizer:** [`execution/build_rulebook_dataset.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_rulebook_dataset.py)
2. **Rulebook QLoRA/LoRA Trainer:** [`execution/train_rulebook_vlm.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/train_rulebook_vlm.py)

## Supported Datasets (25 Unified Thoracic Classes)
- **ChestX-Det** (`--chestx-det`): HuggingFace `natealberti/ChestX-Det` — 13 pathologies extracted from exact semantic masks (`id2label` 1..13, `255` = background) with pixel bounding boxes.
- **PadChest-GR** (`--padchest-gr`): HuggingFace `JasonZZ0601/PadChest-GR-Xray` — grounded radiological findings, anatomical locations, and English sentences.
- **CheXpert** (`--chexpert-hf`): HuggingFace `danjacobellis/chexpert` — frontal/AP/PA studies across 13 acute/device classes (`label == 3` positives).
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
1. **Assistant-Only Loss Masking:** Prompt/system/`<image>` tokens are masked (`labels[:prompt_len] = -100`) so cross-entropy optimizes only the structured 3-section report.
2. **Class Rebalancing:** `--max-normal-ratio 0.22` and `--min-class-count` prevent majority-normal collapse.
3. **Dual-Conditioned CoT Samples:** Every image yields both a Pure-Image CoT target and a CAD-Grounded CoT target.
4. **Validation Audit Gate:** Post-training evaluation automatically checks 5-Rule structure compliance and true-positive finding mention rate before promoting weights.
