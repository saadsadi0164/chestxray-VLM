"""
ChestXStudio: Standalone 4-Model Multi-Resolution Ensemble Evaluator
Path: evaluate_ensemble.py

CLI batch evaluator delegating all model loading, preprocessing, calibration,
and CAM rendering to `execution/ensemble_cad.py` (zero duplicate logic).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from execution.ensemble_cad import (
    VALID_IMAGE_EXTENSIONS,
    evaluate_single_image,
    load_multi_resolution_ensemble,
    render_cam_heatmap_overlay,
    resolve_cnn_device,
)


def discover_xray_images(target_path: str | None) -> list[Path]:
    if target_path:
        p = Path(target_path).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            return [p]
        if p.is_dir():
            return [
                f for f in sorted(p.rglob("*"))
                if f.is_file()
                and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
                and "ensemble_audit_outputs" not in str(f)
                and ".tmp" not in str(f)
                and ".venv" not in str(f)
            ]
        raise FileNotFoundError(f"No valid images found at specified path: {target_path}")

    cwd = Path.cwd()
    for d in (cwd / "test_images", cwd / "images", cwd / "samples", cwd):
        if d.exists() and d.is_dir():
            found = [
                f for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]
            if found:
                return found
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-Model (224+512) Dual-Gate Chest X-Ray Ensemble on local images."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a folder of X-ray images or a single X-ray file (auto-discovers if omitted).",
    )
    parser.add_argument(
        "--mode",
        choices=["balanced", "sensitive", "specific"],
        default="balanced",
        help="Operating threshold mode: balanced (0.0), sensitive (+0.04), or specific (-0.05).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for CNN ensemble inference (default: auto).",
    )
    parser.add_argument(
        "--save-cam",
        action="store_true",
        help="Save padding-corrected 16x16 CAM heatmap overlays to --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default="./.tmp/ensemble_audit_outputs",
        help="Output directory for CSV telemetry and CAM overlays (default: ./.tmp/ensemble_audit_outputs).",
    )
    args = parser.parse_args()

    sens_map = {"balanced": 0.0, "sensitive": 0.04, "specific": -0.05}
    sens_offset = sens_map[args.mode]
    device_str = resolve_cnn_device(args.device)

    image_paths = discover_xray_images(args.path)
    if not image_paths:
        print(
            "[ChestXStudio Evaluator] No image files found.\n"
            "Usage: python evaluate_ensemble.py /path/to/xray_folder_or_file.png [--mode sensitive] [--save-cam]"
        )
        sys.exit(1)

    ensemble_224, resnet_512 = load_multi_resolution_ensemble(device_str=device_str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ensemble_audit_results.csv"

    print(
        f"Evaluating {len(image_paths)} radiograph(s) | "
        f"Mode: {args.mode.upper()} (offset={sens_offset:+.2f}) | Device: {device_str.upper()}\n"
    )

    csv_rows: list[dict[str, object]] = []
    abnormal_count = 0
    normal_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        t0 = time.perf_counter()
        try:
            pil_img = Image.open(img_path)
            pil_img.load()
        except Exception as exc:
            print(f"[{idx}/{len(image_paths)}] SKIPPED {img_path.name} (Unreadable image: {exc})")
            continue

        res = evaluate_single_image(
            pil_img,
            ensemble_224=ensemble_224,
            resnet_512=resnet_512,
            sensitivity_offset=sens_offset,
            device_str=device_str,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        findings = res["top_findings"]
        near_misses = res["near_misses"]
        meta = res["pad_meta"]

        print("=" * 78)
        status_tag = "PATHOLOGY DETECTED" if findings else "STRICT NORMAL"
        print(
            f"[{idx}/{len(image_paths)}] {img_path.name} ({meta['w']}x{meta['h']} px) | "
            f"{status_tag} | {elapsed_ms:.0f} ms"
        )
        print("-" * 78)

        if findings:
            abnormal_count += 1
            print(f"{'DETECTED PATHOLOGY':<34} | {'CAL %':<7} | {'RAW %':<7} | {'QUORUM':<6} | {'CAM LOCATION'}")
            print("-" * 78)
            for d in findings:
                print(
                    f"{d['clinical_name'].title():<34} | "
                    f"{d['cal_prob']:>6.1%} | "
                    f"{d['raw_prob']:>6.1%} | "
                    f"{d['quorum_str']:^6} | "
                    f"{d['location']}"
                )
            if args.save_cam:
                cam_img = render_cam_heatmap_overlay(res["composite_cam"], meta)
                cam_file = out_dir / f"cam_{img_path.stem}.png"
                cam_img.save(cam_file)
                print(f" -> Saved CAM Overlay: {cam_file}")
        else:
            normal_count += 1
            print(" -> Zero pathologies crossed the Dual-Gate + Quorum threshold.")

        if near_misses:
            nm_str = ", ".join(
                f"{m['clinical_name'].title()} ({m['cal_prob']:.1%} cal / {m['raw_prob']:.1%} raw, gate={m['min_cal']:.0%})"
                for m in near_misses
            )
            print(f" -> Sub-Threshold Watchlist: {nm_str}")

        print("=" * 78 + "\n")

        csv_rows.append({
            "filename": img_path.name,
            "resolution": f"{meta['w']}x{meta['h']}",
            "status": status_tag,
            "num_findings": len(findings),
            "detected_pathologies": " | ".join(d["clinical_name"] for d in findings) if findings else "Normal",
            "locations": " | ".join(d["location"] for d in findings) if findings else "Bilateral lungs (unremarkable)",
            "calibrated_scores": " | ".join(f"{d['cal_prob']:.1%}" for d in findings) if findings else "-",
            "raw_scores": " | ".join(f"{d['raw_prob']:.1%}" for d in findings) if findings else "-",
            "quorums": " | ".join(d["quorum_str"] for d in findings) if findings else "-",
            "top_subthreshold": f"{near_misses[0]['clinical_name']} ({near_misses[0]['cal_prob']:.1%})" if near_misses else "-",
        })

    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(
        f"BATCH AUDIT COMPLETE: {len(csv_rows)} images processed "
        f"({abnormal_count} Abnormal, {normal_count} Strict Normal).\n"
        f"CSV Telemetry Report saved to: {csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()
