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

## 6. Next Steps to Complete Model Retraining

1. Run [`directives/train_vlm.md`](file:///Users/saadsadi/Desktop/chest%20x-ray%20VLM/directives/train_vlm.md) on a CUDA GPU (Kaggle / Colab / RunPod) with `ChestX-Det` + `VinDr-CXR` + `NIH` / `CheXpert` / `OpenI`.
2. Save the resulting adapter (`rulebook_v2.json` included automatically) and replace `./chestxray_smolvlm_lora` (or pass `--adapter ./chestxray_smolvlm_lora_v2`).
3. Benchmark before vs. after hallucination rates and per-etiology recall using `evaluate_ensemble.py` + `train_rulebook_vlm.py` validation compliance metrics.
