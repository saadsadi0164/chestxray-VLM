# Directive: ChestXStudio-VLM v3 Training & Acceptance Gates (Phase 4)

## Goal
Train `HuggingFaceTB/SmolVLM-500M-Instruct` on leak-free multi-source chest X-ray datasets using the concise v3 (`FINDINGS:` / `IMPRESSION:`) format, exact offset-weighted cross-entropy (`5.0x` on clinical findings), non-flip resolution/compression jitter, and Mean+Max pooled visual connector auxiliary supervision (`--aux-loss-weight 1.0`) to eliminate the **85.0% Normal-Bias** observed in v1 while maintaining zero held-out leakage.

## Fixed Guards & Preprocessing Invariants
1. **Aspect-Preserving Preprocessing:** All train, val, and inference inputs pass through [`execution/preprocess.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/preprocess.py) (`normalize_cxr_to_rgb`, 1st–99th percentile contrast stretch BEFORE symmetric black letterbox padding to `512x512` RGB).
2. **Dataset Label Truth Guards:**
   - `danjacobellis/chexpert`: HF ClassLabel integers `0=unlabeled, 1=uncertain, 2=absent, 3=present` (exclude `1=uncertain` from Normal).
   - `natealberti/ChestX-Det`: 0-indexed mask IDs `0..9, 11` (skipping corrupted ID `12`) + verified 1-to-1 official Deepwise JSON recovery.
3. **Held-Out Patient Hygiene:** Strip all patients listed in `.tmp/reports/heldout_patient_ids.json` via `--exclude-patients-json` BEFORE splitting or balancing; assert zero overlap across `train`, `val`, and `.tmp/reports/test_manifest.jsonl`.
4. **Per-Source Balancing & Partial-Label Hygiene:** Build SIIM-Pneumothorax training records with `positives_only=True` (`has_pneumo == 0` scans are dropped before disk caching because they may harbor unannotated pathologies), balance Normal share within verified multi-label sources (`ChestX-Det`, `CheXpert`, `PadChest-GR`), cap global Normals at `--max-normal-ratio 0.20`, and enforce `--min-class-count 400` (`--strict-min-class`). Source-masked auxiliary visual BCE (`SOURCE_ANNOTATED_CLASSES`) masks unannotated classes to `0.0` per source during training and validation macro-F1.

## Recommended Cloud GPU Config (Kaggle Tesla T4, 16 GB VRAM)
- **Precision:** Native T4 `fp16` (`4-bit=False` default; add `--quantize-4bit` only if VRAM-constrained).
- **Hyperparameters:**
  `--epochs 4 --batch-size 2 --grad-accum 8 --lr 1e-4 --connector-lr 5e-5 --lora-r 32 --lora-alpha 64 --adapt-vision --aux-loss-weight 1.0 --findings-weight 5.0 --image-size 512 --num-workers 2 --image-splitting --eval-val-samples 200`
- **Artifacts Saved Per Best Checkpoint (selected by `combined_score`):**
  - LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`)
  - Tokenizer/processor config (`do_image_splitting = True`, `image_longest_edge = 1024` -> 5 tiles / 320 visual tokens; v1/v2 inference keeps `False`)
  - `rulebook_v3.json` (metadata, `best_combined_score`, `best_diag_val_loss`, etc.)
  - `aux_head.pt` (Mean+Max pooled `Linear(2*H, 26)` visual classifier state dict for downstream Phase 5 auditing)
  - `train_log.jsonl` (10-step metric telemetry)
- **Safety Abort Guards:** Automatically halts if non-finite/NaN loss occurs twice or if `aux_macro_f1 < 0.15` after Epoch 1.

## Reproducible Pipeline (`scripts/run_kaggle_v3.sh`)
1. Ensure fixed 300-study held-out manifest under `/kaggle/working/reports/`: `execution/build_heldout_test_set.py` -> `test_manifest.jsonl` + `heldout_patient_ids.json`.
2. Build balanced leak-free v3.1 JSONL dataset: `execution/build_rulebook_dataset.py` (`--max-per-source 3500 --max-padchest 1500 --format-version v3 --pure-ratio 0.80 --cad-aligned-ratio 0.10 --exclude-patients-json /kaggle/working/reports/heldout_patient_ids.json`).
3. Run 5-step smoke test (`--dry-run-steps 5 --image-splitting --num-workers 2`) and inspect `[Weighting Check]` + `[Aux Visual Head]`.
4. Launch 4-epoch T4 training and evaluate once on `/kaggle/working/reports/test_manifest.jsonl` via `execution/benchmark_baseline.py` -> `/kaggle/working/reports/v3_test.json`, followed by `execution/compare_runs.py`.

## Acceptance Gates (Must Pass Before Merging/Deploying v3)
- **Normal-Bias Rate (Abn -> Normal):** `< 15.0%`
- **Macro Sensitivity (Per-Finding):** `>= 0.60` (`60.0%`)
- **False-Alarm Rate (Norm -> Abn):** `< 25.0%`
- **Format Compliance (`FINDINGS:` / `IMPRESSION:`):** `>= 95.0%`
