# Directive: Chest X-Ray Vision-Language Model Analysis (ChestXStudio)

## Goal
Execute deterministic, calibrated multimodal analysis of frontal chest radiographs (CXRs) using the locally hosted `SmolVLM-500M-Instruct` model and specialized Indiana University QLoRA adapter (`chestxray_smolvlm_lora`).

## Inputs
1. **Radiograph Image**: Frontal projection (AP or PA view) in `.png`, `.jpg`, or `.dcm.png` format.
2. **Analysis Mode**:
   - `clinical`: Structured professional radiologic findings and impression.
   - `layperson`: Simplified patient breakdown.

## Execution Tools
- **Deterministic Script**: [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py)
- **CLI Runner**: [`inference.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/inference.py)
- **Interactive UI**: [`app.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/app.py)

## Expected Outputs
- Structured report observing the 5-point clinical rulebook:
  1. Systematic Search (airways, lungs/pleura, mediastinum, heart, osseous structures)
  2. Observation Before Etiology (physical findings precedes impression)
  3. Spatial Grounding (anatomical localization)
  4. Explicit Negatives (absence of acute findings)
  5. Uncertainty Calibration
- Intermediate files (if batched) routed to `.tmp/`.

## Edge Cases & Operating Constraints
- **Hardware Fallback**: If Apple Silicon MPS is unavailable, execution must automatically downgrade to CPU with `torch.float32` (avoiding `torch.float16` CPU kernel crashes).
- **Token Decoding**: Never rely on arbitrary substring splitting; slice raw generated IDs past `input_ids.shape[1]`.
- **Medical Disclaimers**: All outputs must carry mandatory non-diagnostic research notices.
