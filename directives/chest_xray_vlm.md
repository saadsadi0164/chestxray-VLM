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
- **Ensemble CAD Engine:** [`execution/ensemble_cad.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/ensemble_cad.py)
- **Unified VLM + Rulebook Pipeline:** [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py)
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
- **Image Splitting:** Always keep `processor.image_processor.do_image_splitting = False` on `SmolVLM-500M-Instruct` — splitting induces anatomical laterality hallucinations.
- **CAM Weight Normalization:** Always track `cam_weights_224[label]` and `cam_weights_512[label]` separately from total ensemble `weights[label]` because 7 of 18 classes are missing from CheXpert/MIMIC DenseNets and 2 classes have NaN thresholds in ResNet50-512.
- **Clause-Level Negation:** Always split sentences on clause delimiters before negation masking so a normal first clause cannot mask a hallucinated pathology later in the sentence.
- **Offline Cache:** Check `./.tmp/torchxrayvision/models_data` and `~/.torchxrayvision/models_data` before downloading, and use `local_files_only=True` fallback for HuggingFace assets.
- **Medical Disclaimer:** All outputs must carry non-diagnostic research disclaimers.
