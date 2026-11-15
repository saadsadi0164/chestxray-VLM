# ChestXStudio-VLM: Local Vision-Language Model for Chest X-Ray Interpretation

**Author:** M. Saad Sadi  
**Environment:** macOS (Apple Silicon MPS) & Kaggle (Tesla T4 QLoRA)  

## Overview
`ChestXStudio-VLM` is a locally run, lightweight Vision-Language Model (VLM) engineered to interpret frontal chest X-rays and generate structured clinical reports. Built as part of **ChestXStudio**, this project bridges medical diagnostics and AI by enforcing a strict clinical rulebook during inference.

## Key Technical Features
- **Base Architecture:** `SmolVLM-500M-Instruct` fine-tuned using Low-Rank Adaptation (**QLoRA**, $r=16$, $\alpha=32$).
- **Quantization & Compute:** 4-bit NormalFloat (`NF4`) quantization via `bitsandbytes` during cloud training; native **Apple Silicon MPS** acceleration for fast, private offline inference on MacBook Air (`~5.1s` per study).
- **Dataset:** Indiana University Chest X-Ray Collection (cleaned, anonymized artifacts repaired, and 3-channel RGB normalized).
- **Clinical Rulebook Enforcement:** 
  1. Systematic search (airways, lungs, mediastinum, heart, bones).
  2. Observation-first reasoning (visual findings preceding etiology).
  3. Spatial grounding and explicit negative reporting.

## Repository Structure
- `inference.py`: Local inference engine utilizing PyTorch MPS and PEFT adapter merging.
- `chestxray_smolvlm_lora/`: Trained QLoRA medical adapter weights (`34.37 MB`).
