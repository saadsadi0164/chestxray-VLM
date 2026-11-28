# Directive: Chest X-Ray Vision-Language Model Analysis (ChestXStudio)

## Goal
Execute deterministic, hallucination-resistant multimodal analysis of frontal chest radiographs (AP/PA) under the 5-Rule Clinical Rulebook, using a 2-stage architecture:
1. **Stage 1 (Multi-Resolution CAD & 16x16 CAM):** 4-model TorchXRayVision ensemble (`densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_ch`, `resnet50-res512-all`) with 16-bit-safe grayscale extraction, 3-view aspect-preserved TTA, piecewise `op_norm`, Dual-Gate + Quorum voting, and fused 16x16 Class Activation Mapping (CAM).
2. **Stage 2 (LoRA VLM & Guardrail):** `HuggingFaceTB/SmolVLM-500M-Instruct` with local medical QLoRA adapter (`./chestxray_smolvlm_lora`), `do_image_splitting = False`, clause-level anti-hallucination sentence filtering, and strict 5-Rule report synthesis.

## Inputs
- **Radiograph Image**: Frontal projection (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`), including 12-bit/16-bit PNGs.
- **Operating Threshold Mode**: `High Sensitivity` (+0.04), `Balanced Clinical` (0.00), `High Specificity` (-0.05).
- **Pipeline Mode**: `audited` (Stage 1 + Stage 2, default) or `pure_vlm` (standalone SmolVLM+LoRA CoT).

## Execution Tools (Layer 3)
- **Canonical Preprocessing:** [`execution/preprocess.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/preprocess.py)
- **Ensemble CAD Engine:** [`execution/ensemble_cad.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/ensemble_cad.py)
- **Unified VLM + Rulebook Pipeline:** [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py)
- **Negation-Aware Report Parser:** [`execution/report_parser.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/report_parser.py)
- **Held-Out Manifest & Baselines:** [`execution/build_heldout_test_set.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_heldout_test_set.py), [`execution/benchmark_baseline.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/benchmark_baseline.py), [`execution/compare_runs.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/compare_runs.py)
- **Interactive Gradio Workstation:** [`app.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/app.py)
- **Offline CLI Runner:** [`inference.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/inference.py)
- **Batch CSV Evaluator:** [`evaluate_ensemble.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/evaluate_ensemble.py)

## Expected Outputs
- Structured 3-section report enforcing:
  1. Systematic Search (airways, lungs/pleura, mediastinum, heart, osseous cage)
  2. Observation Before Etiology (visual evidence before diagnosis)
  3. Spatial Grounding (laterality & zone from fused 16x16 CAM)
  4. Explicit Negatives (absence of unpresent acute emergencies)
  5. Uncertainty Calibration (probabilistic phrasing tied to Dual-Gate telemetry)
- Batch CSV telemetry and CAM overlays saved under `./.tmp/ensemble_audit_outputs/`.

## Edge Cases & Learnings
- **Single Aspect-Preserving Preprocessing:** All training and inference paths (both v1 and v2/v3 adapters) MUST call `execution/preprocess.normalize_cxr_to_rgb` (1st–99th percentile contrast stretch BEFORE symmetric black letterbox padding and `512x512` RGB resize, with optional `pydicom` RescaleSlope/Intercept, `MONOCHROME1` inversion, and windowing).
- **Safe Apple Silicon Precision:** `resolve_llm_device` defaults MPS to `torch.float32` (`allow_mps_bf16=False`) to avoid fp16/bf16 logit instability and reduce false-alarm hallucinations.
- **Image Splitting:** Always keep `processor.image_processor.do_image_splitting = False` on `SmolVLM-500M-Instruct` — splitting induces anatomical laterality hallucinations.
- **CAM Weight Normalization:** Always track `cam_weights_224[label]` and `cam_weights_512[label]` separately from total ensemble `weights[label]` because 7 of 18 classes are missing from CheXpert/MIMIC DenseNets and 2 classes have NaN thresholds in ResNet50-512.
- **Clause-Level Negation & Parsing:** Always split sentences on clause delimiters before negation masking, and use `execution/report_parser.parse_report_findings` (comma-list continuation + hedge awareness) for benchmark evaluation.
- **Held-Out Patient Hygiene:** Never train or tune on any patient listed in `.tmp/reports/heldout_patient_ids.json`.
- **Offline Cache:** Check `./.tmp/torchxrayvision/models_data` and `~/.torchxrayvision/models_data` before downloading, and use `local_files_only=True` fallback for HuggingFace assets.
- **Medical Disclaimer:** All outputs must carry non-diagnostic research disclaimers.
