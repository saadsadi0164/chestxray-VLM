"""
ChestXStudio: Offline CLI Diagnostic Tool
Path: inference.py

CLI tool for running deterministic batch or single-study analyses
with device-safe execution and Indiana University clinical rulebook adherence.
"""

import os
import sys
import argparse
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from execution.analyze_xray import ChestXRayEngine, find_default_xray

def main():
    parser = argparse.ArgumentParser(description="ChestXStudio-VLM Local Diagnostic CLI")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to frontal chest X-ray image (.png/.jpg). If omitted, scans workspace."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clinical", "layperson"],
        default="clinical",
        help="Output mode: 'clinical' or 'layperson'."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./chestxray_smolvlm_lora",
        help="Path to trained QLoRA adapter directory."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("      CHESTXSTUDIO-VLM: LOCAL DIAGNOSTIC AUDIT & INFERENCE      ")
    print("=" * 65)

    # 1. Resolve image path safely
    target_image_path = args.image
    if not target_image_path:
        target_image_path = find_default_xray()

    if not target_image_path or not os.path.exists(target_image_path):
        print(f"[Error] Target image not found: {target_image_path}")
        print("Please supply an image via `--image <path_to_xray.png>`.")
        sys.exit(1)

    print(f"Study Image:   {target_image_path}")
    print(f"LoRA Adapter:  {args.adapter}")
    print(f"Report Mode:   {args.mode.upper()}")
    print("-" * 65)

    # 2. Initialize Engine & Execute
    engine = ChestXRayEngine(adapter_path=args.adapter)
    raw_img = Image.open(target_image_path)
    result = engine.analyze(image=raw_img, mode=args.mode)

    print("-" * 65)
    print(f"Completed in {result['latency_seconds']}s on {result['device'].upper()}")
    print("=" * 65)
    print("LOCAL RADIOLOGY REPORT:")
    print("=" * 65)
    print(result["report"])
    print("=" * 65)
    print("DISCLAIMER: For research/demonstration only. Not for clinical use.")

if __name__ == "__main__":
    main()
