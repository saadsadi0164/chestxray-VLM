# ChestXStudio-VLM: Multi-Resolution Chest X-Ray Analysis & Rulebook VLM

**Author:** M. Saad Sadi  
**Environment:** macOS (Apple Silicon MPS Acceleration) & Kaggle/Cloud GPU (QLoRA)

---

## Overview
`ChestXStudio-VLM` is a locally runnable medical imaging workstation and VLM training framework engineered to interpret frontal chest radiographs under a strict **5-Rule Clinical Rulebook**, preventing hallucinated findings and "Normal-biased" omissions.

### Two-Stage Audited Clinical Pipeline
1. **Stage 1 — Multi-Resolution Vision CAD & 16x16 CAM (`execution/ensemble_cad.py`):**
   - **3× `DenseNet121` (224×224):** `all` (NIH/CheXpert/MIMIC/PadChest/OpenI), `chex` (Stanford CheXpert), `mimic_ch` (MIT MIMIC-CXR)
   - **1× `ResNet50` (512×512):** `resnet50-res512-all` (PadChest/NIH/RSNA/SIIM-Pneumothorax/VinDr-CXR)
   - **16-Bit-Safe 3-View Aspect-Preserved TTA** + piecewise `op_norm` calibration + **Dual-Gate & Multi-Model Quorum** + **16×16 Multi-Resolution Spatial CAM**
2. **Stage 2 — Multimodal VLM & Anti-Hallucination Auditor (`execution/analyze_xray.py`):**
   - `HuggingFaceTB/SmolVLM-500M-Instruct` + local QLoRA medical adapter (`chestxray_smolvlm_lora/`, Rank `r=16`, `α=32`) with global chest context (`do_image_splitting=False`)
   - **Clause-Level Anti-Hallucination Guardrail** validating VLM statements against CAD detections and enforcing the 5-Rule Clinical Rulebook.

---

## 5-Rule Clinical Rulebook
1. **Systematic Search:** Evaluates airways, lungs/pleura, mediastinum, heart, and osseous thoracic cage.
2. **Observation Before Etiology:** Documents physical visual features before stating a clinical diagnosis.
3. **Spatial Grounding:** Grounds every finding to an anatomical zone using fused 16×16 CAMs (or bounding boxes in training).
4. **Explicit Negatives:** Explicitly states absence of unpresent acute emergencies (`pneumothorax`, `pleural effusion`, `consolidation`, `pulmonary edema`, `fracture`).
5. **Uncertainty Calibration:** Calibrates phrasing (`strongly consistent with` vs `suggestive of`) to Dual-Gate probability scores.

---

## Repository Structure
- [`app.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/app.py) — Gradio 6.0 Clinical Workstation UI (Audited Hybrid + Pure VLM modes)
- [`inference.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/inference.py) — Offline CLI single-image diagnostic runner
- [`evaluate_ensemble.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/evaluate_ensemble.py) — Batch CLI evaluator with CSV telemetry & CAM overlay exports
- [`execution/ensemble_cad.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/ensemble_cad.py) — Shared 4-model CAD ensemble & 16×16 CAM engine
- [`execution/analyze_xray.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/analyze_xray.py) — Shared LoRA VLM + clause-level anti-hallucination auditor
- [`execution/build_rulebook_dataset.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/build_rulebook_dataset.py) — Multi-dataset (ChestX-Det, VinDr-CXR, NIH/CheXpert/MIMIC/OpenI) 5-Rule CoT builder
- [`execution/train_rulebook_vlm.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/execution/train_rulebook_vlm.py) — Multi-dataset QLoRA/LoRA fine-tuning pipeline (with assistant-only loss masking & projector/ViT adaptation)
- [`directives/`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/directives) — Layer 1 Standard Operating Procedures
- [`tests/test_pipeline.py`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/tests/test_pipeline.py) — Automated unit & regression test suite
- [`chestxray_smolvlm_lora/`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/chestxray_smolvlm_lora) — Trained QLoRA medical adapter weights

---

## Quick Usage
```bash
source .venv/bin/activate

# 1. Launch Gradio Workstation
python app.py

# 2. Single-study CLI inference
python inference.py --image 1002_IM-0004-1001.dcm.png --mode audited

# 3. Batch folder ensemble audit (outputs saved to ./.tmp/ensemble_audit_outputs/)
python evaluate_ensemble.py 1002_IM-0004-1001.dcm.png --save-cam

# 4. Run regression tests
python -m unittest discover -s tests -v
```

> **Disclaimer:** For academic research and educational use only. Not an FDA/CE-cleared medical device.
