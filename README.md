# ChestXStudio-VLM: Local Vision-Language Model for Chest X-Ray Interpretation

**Author:** M. Saad Sadi  
**Environment:** macOS (Apple Silicon MPS Acceleration) & Kaggle Cloud (Tesla T4 GPU, QLoRA)  

---

## 🚀 Overview
`ChestXStudio-VLM` is a locally hosted, lightweight Vision-Language Model (VLM) engineered to analyze frontal chest X-rays and generate structured clinical reports. Developed as part of **ChestXStudio**, this project bridges medical diagnostics and machine learning by combining a multimodal transformer architecture with a rigorous, constraint-driven clinical reasoning rulebook.

---

## 🛠️ Technical Architecture & Training Pipeline

| Component | Specification |
| :--- | :--- |
| **Base Model** | `HuggingFaceTB/SmolVLM-500M-Instruct` (517M parameters) |
| **Fine-Tuning Method** | Low-Rank Adaptation (**QLoRA**, Rank $r=16$, $\alpha=32$) |
| **Quantization** | 4-bit NormalFloat (`NF4`) via `bitsandbytes` |
| **Target Modules** | Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) & MLPs (`gate_proj`, `up_proj`, `down_proj`) |
| **Dataset** | Indiana University Chest X-Ray Collection (frontal projections, text-repaired) |
| **Local Inference Engine** | PyTorch Metal Performance Shaders (`MPS`) on Apple Silicon (`~5.1s` per study) |

---

## 📋 Strict Clinical Rulebook
During both fine-tuning and inference, the model is constrained by an expert diagnostic rulebook to prevent hallucinations and enforce structured reporting:
1. **Systematic Search:** Evaluates airways, lungs/pleura, mediastinum, heart, and osseous structures.
2. **Observation Before Etiology:** Requires physical visual findings to be explicitly described *before* stating the clinical impression or etiology.
3. **Spatial Grounding:** Mandates anatomical localization for all positive observations.
4. **Explicit Negatives:** Explicitly documents the absence of acute cardiopulmonary abnormalities.

---

## 📂 Repository Structure
* `chestxray_smolvlm_lora/` — Trained QLoRA medical adapter weights (34.37 MB)
* `inference.py` — Local offline inference script (Apple Silicon MPS)
* `README.md` — Project documentation & CV portfolio overview