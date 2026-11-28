# ChestXStudio-VLM: Project Status, Code Audit Log & AI Handoff Report

> **Purpose:** Self-contained engineering status report for human developers and AI assistants to resume work without skimming or ambiguity.

---

## 1. Executive Summary

- **Project Name:** ChestXStudio-VLM
- **Repository:** `saadsadi0164/chestxray-VLM` (Branch: `main`)
- **Author:** M. Saad Sadi (`scientistsadi@gmail.com`)
- **Local Machine Hardware:** macOS Apple Silicon (8 GB Unified RAM, ~15 GB free disk), PyTorch `2.15.0` with Metal Performance Shaders (`MPS`).

---

## 2. Current Architecture (3-Layer Design per `agent.md`)

```text
chest x-ray VLM/
├── directives/                          # LAYER 1: SOPs (What to do)
│   ├── chest_xray_vlm.md                # Inference & guardrail SOP
│   └── train_vlm.md                     # Multi-dataset QLoRA retraining SOP
├── agent.md / CLAUDE.md / AGENTS.md /   # LAYER 2: Orchestration Instructions
│   └── GEMINI.md
├── execution/                           # LAYER 3: Deterministic Shared Tools
│   ├── __init__.py
│   ├── ensemble_cad.py                  # 4-model CAD ensemble + 16x16 CAM engine
│   ├── analyze_xray.py                  # Unified SmolVLM+LoRA + Clause Anti-Hallucination Auditor
│   ├── build_rulebook_dataset.py        # Multi-dataset 5-Rule CoT dataset synthesizer
│   └── train_rulebook_vlm.py            # QLoRA/LoRA trainer + post-train compliance evaluator
├── app.py                               # Slim Gradio 6.0 UI (Audited Hybrid + Pure VLM)
├── inference.py                         # Slim CLI single-study runner
├── evaluate_ensemble.py                 # Slim CLI batch evaluator (CSV -> .tmp/)
├── tests/
│   └── test_pipeline.py                 # 13 automated unit & regression tests
├── chestxray_smolvlm_lora/              # Current OpenI LoRA adapter (r=16, alpha=32)
├── 1002_IM-0004-1001.dcm.png            # Local test frontal chest X-ray
├── README.md
└── PROGRESS.md
```

---

## 3. Code Audit & Bugs Fixed (November 24, 2026)

1. **Restored Missing QLoRA Adapter in `app.py`:** `app.py` previously loaded the un-adapted `SmolVLM-500M-Instruct` base model without attaching `./chestxray_smolvlm_lora`. Now unified through `execution/analyze_xray.py` (`PeftModel` attached when `use_lora=True`).
2. **Eliminated ~1,100+ Lines of Duplicated Code:** Consolidated `CLASS_CALIBRATION_GATES`, `compute_op_norm`, 16-bit preprocessing, TTA, CAM localization, and heatmap rendering out of `app.py` (910 lines → ~130 lines) and `evaluate_ensemble.py` (732 lines → ~190 lines) into [`execution/ensemble_cad.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/ensemble_cad.py) and [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py).
3. **Fixed `cam_w_sum` CAM Denominator Bug:** `weights[:len(cnn_ensemble_224)]` previously sliced ResNet50's vote weight into the DenseNet CAM denominator for the 7 classes missing from CheXpert/MIMIC DenseNets (`Infiltration`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Nodule`, `Mass`, `Hernia`), shrinking their CAM maps by ~2.5x. Replaced with explicit `cam_weights_224` and `cam_weights_512` accumulators.
4. **Added Native 512x512 ResNet50 16x16 CAM Fusion:** Extracted `layer4` `[2, 2048, 16, 16]` spatial maps and `fc` logits in a single clean pass (without mutating model `op_threshs` / `apply_sigmoid` buffers) and fused with upsampled DenseNet `7x7 -> 16x16` CAMs.
5. **Fixed 16-Bit PNG Clipping & NaN Checks:** Unified 16-bit-safe `extract_grayscale_float32()` across both UI and batch evaluation, and checked `np.isnan(...).any()` across all TTA views.
6. **Fixed Clause-Level Negation & Prompt-Echo Bugs in Anti-Hallucination Guardrail:**
   - Added missing `"Lung Lesion"` stem to `ALL_PATHOLOGY_STEMS`.
   - Split sentences on clause boundaries before negation masking so `"The heart is normal, and there is pneumothorax"` can no longer slip past.
   - Blocked `"Normal-bias"` claims when Stage 1 CAD verified pathologies above Dual-Gate.
   - Filtered out OpenI prompt-echo sentences (`Systematic Search:`, `Step 1: Examine...`).
   - Added dual-adapter prompt routing (`rulebook_v2.json` detection) so OpenI v1 uses its native `user` prompt cleanly while v2 rulebook adapters use `system` + `user` CoT format.
7. **Cleaned Up Repo Clutter:** Removed stray nested clone folder `chest x-ray VLM/`, redundant `.gitkeep` placeholders, dead variables, and moved `evaluate_ensemble.py` outputs under `./.tmp/ensemble_audit_outputs/`.

---

## 4. Multi-Dataset Anti-Hallucination Training Pipeline (Ready for Cloud GPU)

Because full multi-dataset training (`ChestX-Det`, `VinDr-CXR`, `NIH ChestX-ray14`, `CheXpert`, `MIMIC-CXR`, `OpenI`) exceeds the local Mac's **8 GB RAM / 15 GB free disk** budget, the complete training pipeline has been implemented and smoke-tested locally so it can execute directly on Kaggle T4 / Colab / Cloud GPUs:

- **[`execution/build_rulebook_dataset.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_rulebook_dataset.py):**
  - Covers **25 thoracic etiologies & findings** across `ChestX-Det`, `VinDr-CXR`, and custom/NIH/CheXpert/MIMIC/OpenI JSONL manifests.
  - Maps bounding boxes `[x1, y1, x2, y2]` to radiological laterality and zones (`Rule 3: Spatial Grounding`).
  - Rebalances class distribution (`--max-normal-ratio 0.22`, `--min-class-count 200`) to eliminate Normal-biased collapse.
  - Emits both **Pure-Image CoT** and **CAD-Grounded CoT** examples and validates every record against the 5 rules.
- **[`execution/train_rulebook_vlm.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/train_rulebook_vlm.py):**
  - Implements **Assistant-Only Loss Masking** (`labels[:prompt_len] = -100`).
  - Supports `--adapt-vision` (adapting LLM projections + ViT `out_proj` + `connector` modality projector) and `--quantize-4bit` (NF4 QLoRA on CUDA GPUs).
  - Runs automated post-training Rulebook Compliance & Finding Mention audits.
  - Verified locally via 1-step MPS dry-run (`[Epoch 1/2 | Step 1] loss=2.4671`).

---

## 5. Verified Test Results (Post-Refactor)

- **Automated Unit Suite (`python -m unittest discover -s tests -v`):** `13/13` passed in `0.006s`.
- **CLI Audited Inference (`python inference.py --image 1002_IM-0004-1001.dcm.png --mode audited`):** Completed in `5.8s` on MPS; clean 5-rule report with zero prompt echoes.
- **Batch Evaluator (`python evaluate_ensemble.py 1002_IM-0004-1001.dcm.png --save-cam --device mps`):** Completed in `1272 ms` (`STRICT NORMAL`, CSV saved to `./.tmp/ensemble_audit_outputs/ensemble_audit_results.csv`).

---

## 6. v2.2 / v2.2.1 & v3 Phases 0–2 Progress (`fix/normal-bias`)

### Pre-v3 Foundation (`v2.2` / `v2.2.1` on `main` @ `3e4c269`)
- Added unified 3-mode rulebook data emission (`pure`, `cad_aligned`, `cad_reject`), streaming HF loaders (`ChestX-Det` 0-indexed `0..9, 11` skipping corrupted ID `12` + Deepwise JSON recovery, `CheXpert` ClassLabel `0/1/2/3`, `PadChest-GR`, `SIIM-Pneumothorax`), Visual Connector Auxiliary Classification Head (`AUX_CLASSES`), 20x decision-token weighting, T4 `fp16` `GradScaler` + `fp32` aux-head AMP guard, immediate `vis_hook.clear()` VRAM release, and stratified holdout evaluation (`execution/benchmark_etiologies.py`).

### v3 Phase 0 — Fixed Patient-Grouped Held-Out Set & Baseline (`86499a9`)
- Installed `datasets` cleanly in macOS `.venv` (no CUDA pins) and kept all 22 pre-existing tests passing.
- Built fixed 300-study patient-grouped holdout manifest ([`execution/build_heldout_test_set.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_heldout_test_set.py)) -> `.tmp/reports/test_manifest.jsonl`: **100 Normal, 200 Abnormal** across all 8 priority finding groups (`Pneumothorax`, `Effusion`, `Consolidation`, `Cardiomegaly`, `Edema`, `Nodule`/`Mass`, `Atelectasis`, `Fracture`), with **300 unique patient IDs** and zero overlap asserted (`.tmp/reports/heldout_patient_ids.json`).
- Created resumable evaluator [`execution/benchmark_baseline.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/benchmark_baseline.py) (atomic checkpoints every 25 studies) and evaluated `./chestxray_smolvlm_lora` (`pure_vlm`).

### v3 Phase 1 — Canonical Aspect-Preserving Preprocessing & Safe MPS `fp32` (`c519b94`)
- Created [`execution/preprocess.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/preprocess.py) (`normalize_cxr_to_rgb`) supporting PIL, 8/16-bit PNG, JPEG, and DICOM (`pydicom` RescaleSlope/Intercept, `MONOCHROME1` inversion, WindowCenter/Width, graceful fallback), applying 1st–99th percentile contrast stretch on native anatomy followed by symmetric centered letterbox padding BEFORE resizing to `512x512` RGB.
- Unified preprocessing across `build_rulebook_dataset.py`, `train_rulebook_vlm.py` (`RulebookCXRDataset.__getitem__` and validation eval), and `analyze_xray.py` (both v1 and v2/v3 paths).
- Changed `resolve_llm_device` in [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py) so Apple Silicon MPS defaults to `torch.float32` (`allow_mps_bf16=False`).
- Verified byte-identical train/inference tensors and zero-NaN generation on 5 held-out studies (`tests/test_preprocess_and_precision.py`). Re-running baseline benchmark cut Normal False-Alarm Rate roughly in half (`11.0% -> 5.0%` raw; `13.0% -> 7.0%` rescored).

### v3 Phase 2 — Negation-Aware Parser, Combined Checkpoint Scoring & Run Comparison
- Created [`execution/report_parser.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/report_parser.py) (`parse_report_findings`, `compute_clinical_metrics`) returning `"present" | "absent" | "uncertain"` for every canonical pathology class, handling comma-separated negative lists, prepositional negations, trailing passive negations, contrast conjunctions, clinical hedges, and Mode-3 CAD refutations.
- Updated [`execution/train_rulebook_vlm.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/train_rulebook_vlm.py) to score `--eval-val-samples 200` (`max_eval=2` for dry runs) per epoch via `parse_report_findings` and select best checkpoints by `combined_score` (logging both `combined_score` and `diag_val_loss`).
- Added [`execution/compare_runs.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/compare_runs.py), rescored both Phase 0 and Phase 1 runs, and spot-checked 20 reports against human reading:
  - Phase 0 (`baseline_v1`, MPS `fp16`, raw squashed): Normal-Bias **86.5%** (173/200), False-Alarm **13.0%** (13/100), Macro Sens **0.3%**, 5-Rule Comp **0.0%**, Combined **-0.4297**.
  - Phase 1 (`baseline_v1_phase1`, MPS `fp32`, canonical letterbox): Normal-Bias **85.0%** (170/200), False-Alarm **7.0%** (7/100), Macro Sens **0.1%**, 5-Rule Comp **0.0%**, Combined **-0.4239**.

## 7. Next Steps (Phase 3+)
1. Build leak-free multi-dataset training shards excluding `.tmp/reports/heldout_patient_ids.json`.
2. Execute cloud GPU (Kaggle T4 / Colab) QLoRA/LoRA retraining with `execution/train_rulebook_vlm.py` (`combined_score` checkpoint selection).
3. Benchmark retrained v3 adapter on the fixed 300-study held-out `.tmp/reports/test_manifest.jsonl` via `execution/benchmark_baseline.py` and `execution/compare_runs.py`.
