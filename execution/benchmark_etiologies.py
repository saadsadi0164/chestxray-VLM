"""
ChestXStudio-VLM: Multi-Etiology Diagnostic Benchmark
Path: execution/benchmark_etiologies.py

Evaluates Pure VLM (LoRA standalone) vs. Stage 1 CAD Ensemble vs. Audited Hybrid
across a labeled manifest of diverse chest X-ray etiologies, tracking:
  - Per-etiology recall (VLM-only vs. Hybrid CAD)
  - "Normal-bias" false-negative rate on abnormal studies
  - Rulebook structural compliance
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from execution.analyze_xray import ChestXRayPipeline, _CLAUSE_SPLIT_RE, _NEGATION_RE
from execution.build_rulebook_dataset import validate_rulebook_compliance

NORMAL_CLAIM_RE = re.compile(
    r"\b(?:no acute cardiopulmonary|no active disease|no acute disease|"
    r"no acute abnormality|normal chest|lungs are clear|no cardiopulmonary disease)\b",
    flags=re.IGNORECASE,
)

ETIOLOGY_SYNONYMS: dict[str, list[str]] = {
    "pneumothorax": ["pneumothora"],
    "effusion": ["effusion", "pleural fluid", "costophrenic blunting", "meniscus"],
    "consolidation": ["consolidat", "airspace opac", "pneumonia"],
    "pneumonia": ["pneumonia", "consolidat", "infiltrat", "airspace opac"],
    "edema": ["edema", "congest", "cephalization", "perihilar"],
    "cardiomegaly": ["cardiomegal", "enlarged heart", "enlarged cardiac", "cardiothoracic"],
    "atelectasis": ["atelecta", "volume loss", "subsegmental", "collapse"],
    "emphysema": ["emphysem", "hyperinflat", "copd"],
    "fibrosis": ["fibros", "reticular", "interstitial"],
    "nodule": ["nodul", "lesion", "granuloma"],
    "mass": ["mass", "lesion"],
    "lung lesion": ["lesion", "nodul", "mass", "opacit"],
    "fracture": ["fractur", "cortical"],
    "calcific": ["calcif", "granuloma"],
    "pleural thickening": ["pleural thickening", "apical cap", "thickening"],
    "support devices": ["device", "pacemaker", "catheter", "tube", "wire", "lead", "hardware"],
    "normal": ["no acute", "normal", "no active"],
}


def matches_etiology(report_text: str, gt_bucket: str) -> bool:
    """Checks positive/unnegated clinical findings only (ignores Rule 4 & Step headers)."""
    if gt_bucket == "normal":
        return bool(NORMAL_CLAIM_RE.search(report_text))
    stems = ETIOLOGY_SYNONYMS.get(gt_bucket, [gt_bucket.lower()])
    for sent in re.split(r"(?<=[.!?])\s+|\n+", report_text):
        s_low = sent.strip().lower()
        if not s_low or re.match(r"^(?:step\s+\d+:|no radiographic evidence of\b)", s_low):
            continue
        for clause in _CLAUSE_SPLIT_RE.split(s_low):
            unnegated = _NEGATION_RE.sub(" ", clause.strip())
            if any(st in unnegated for st in stems):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pure VLM vs Audited Hybrid across CXR etiologies.")
    parser.add_argument("--manifest", default="./.tmp/virtual_test_suite/manifest.json")
    parser.add_argument("--out", default="./.tmp/virtual_test_suite/benchmark_report.json")
    parser.add_argument("--adapter", default="./chestxray_smolvlm_lora")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = ChestXRayPipeline(adapter_path=args.adapter, load_cad=True, use_lora=True)

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(manifest, start=1):
        img_path = item["path"]
        gt_bucket = item["gt_bucket"]
        img = Image.open(img_path)

        pure_res = pipeline.analyze(img, mode="pure_vlm")
        aud_res = pipeline.analyze(img, sensitivity_mode="Balanced Clinical", use_vlm=True, mode="audited")

        pure_txt = pure_res["report"]
        aud_txt = aud_res["report"]

        pure_hit = matches_etiology(pure_txt, gt_bucket)
        pure_normal_bias = (gt_bucket != "normal") and bool(NORMAL_CLAIM_RE.search(pure_txt)) and not pure_hit
        pure_rulebook_ok, pure_rulebook_errs = validate_rulebook_compliance(pure_txt)

        aud_hit = matches_etiology(aud_txt, gt_bucket)
        aud_rulebook_ok, aud_rulebook_errs = validate_rulebook_compliance(aud_txt)

        cad_labels = [f"{d['raw_label']}({d['cal_prob']:.0%}/{d['raw_prob']:.0%})" for d in aud_res["cad_findings"]]
        near_misses = [f"{m['raw_label']}({m['cal_prob']:.0%}/{m['raw_prob']:.0%})" for m in aud_res["near_misses"]]

        rec = {
            "id": item["id"],
            "dataset": item["dataset"],
            "gt_bucket": gt_bucket,
            "gt_label": item["gt_label"],
            "gt_locations": item.get("gt_locations"),
            "gt_sentence": item.get("gt_sentence"),
            "pure_vlm_hit": pure_hit,
            "pure_vlm_normal_bias": pure_normal_bias,
            "pure_vlm_rulebook_ok": pure_rulebook_ok,
            "pure_vlm_rulebook_errs": pure_rulebook_errs,
            "pure_vlm_report": pure_txt,
            "audited_hit": aud_hit,
            "audited_rulebook_ok": aud_rulebook_ok,
            "cad_detected": cad_labels,
            "cad_near_misses": near_misses,
            "audited_report": aud_txt,
        }
        results.append(rec)

        print(
            f"[{idx}/{len(manifest)}] GT={gt_bucket.upper():<18} | "
            f"PureVLM_Hit={str(pure_hit):<5} NormalBias={str(pure_normal_bias):<5} RuleOK={str(pure_rulebook_ok):<5} | "
            f"Audited_Hit={str(aud_hit):<5} CAD={cad_labels}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    abnormal = [r for r in results if r["gt_bucket"] != "normal"]
    print("\n" + "=" * 78)
    print(f"BENCHMARK SUMMARY ({len(results)} studies; {len(abnormal)} abnormal):")
    if abnormal:
        print(f" - Pure VLM Sensitivity (Abnormal):   {sum(r['pure_vlm_hit'] for r in abnormal)}/{len(abnormal)}")
        print(f" - Pure VLM Normal-Bias Rate:         {sum(r['pure_vlm_normal_bias'] for r in abnormal)}/{len(abnormal)}")
        print(f" - Pure VLM 5-Rule Compliance:        {sum(r['pure_vlm_rulebook_ok'] for r in results)}/{len(results)}")
        print(f" - Audited Hybrid Sensitivity:        {sum(r['audited_hit'] for r in abnormal)}/{len(abnormal)}")
        print(f" - Audited 5-Rule Compliance:         {sum(r['audited_rulebook_ok'] for r in results)}/{len(results)}")
    print(f"Full JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
