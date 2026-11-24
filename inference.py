"""
ChestXStudio: Offline CLI Diagnostic Tool
Path: inference.py

Thin CLI entrypoint invoking `ChestXRayPipeline` from `execution/analyze_xray.py`.
Supports both audited multi-resolution hybrid inference and standalone pure-VLM CoT mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from execution.analyze_xray import ChestXRayPipeline, find_default_xray


def main() -> None:
    parser = argparse.ArgumentParser(description="ChestXStudio-VLM Local Diagnostic CLI")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to frontal chest X-ray image (.png/.jpg). Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["audited", "pure_vlm"],
        default="audited",
        help="Pipeline mode: 'audited' (4-model CAD + LoRA VLM + Guardrail) or 'pure_vlm'.",
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        choices=["High Sensitivity", "Balanced Clinical", "High Specificity"],
        default="Balanced Clinical",
        help="CAD Dual-Gate sensitivity operating point (audited mode only).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./chestxray_smolvlm_lora",
        help="Path to trained QLoRA adapter directory.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip SmolVLM commentary in audited mode (fast CAD + Rulebook synthesis only).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("        CHESTXSTUDIO-VLM: LOCAL DIAGNOSTIC AUDIT & INFERENCE        ")
    print("=" * 70)

    target_image_path = args.image or find_default_xray()
    if not target_image_path or not os.path.exists(target_image_path):
        print(f"[Error] Target image not found: {target_image_path}")
        print("Please supply an image via `--image <path_to_xray.png>`.")
        sys.exit(1)

    print(f"Study Image:   {target_image_path}")
    print(f"LoRA Adapter:  {args.adapter}")
    print(f"Pipeline Mode: {args.mode.upper()}")
    print("-" * 70)

    pipeline = ChestXRayPipeline(
        adapter_path=args.adapter,
        load_cad=(args.mode == "audited"),
    )
    raw_img = Image.open(target_image_path)
    result = pipeline.analyze(
        image=raw_img,
        sensitivity_mode=args.sensitivity,
        use_vlm=not args.no_vlm,
        mode=args.mode,
    )

    print("-" * 70)
    print(f"Completed in {result['latency_seconds']}s | Mode: {result['mode'].upper()}")
    print("=" * 70)
    print("LOCAL RADIOLOGY REPORT:")
    print("=" * 70)
    print(result["report"])
    print("=" * 70)
    print("DISCLAIMER: For research/educational use only. Not for clinical diagnosis.")


if __name__ == "__main__":
    main()
