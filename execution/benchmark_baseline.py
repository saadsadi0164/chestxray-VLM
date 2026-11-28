"""
ChestXStudio-VLM: Fixed Held-Out Baseline & Resumable Evaluator
Path: execution/benchmark_baseline.py

Evaluates Pure VLM (or Audited Hybrid) inference deterministically on a
patient-grouped held-out manifest (default: .tmp/reports/test_manifest.jsonl).
Features:
  - Resumable JSON checkpointing every N studies (default: 25) via atomic write
    so long Apple Silicon MPS or CPU runs never lose progress.
  - Negation-aware positive finding detection (ignores "no", "without",
    "negative for", "free of", "no evidence of", etc.; delegates automatically
    to `execution/report_parser.py` when present).
  - Per-finding Sensitivity, Specificity, and F1 plus overall Normal-Bias Rate,
    False-Alarm Rate, Macro Sensitivity, and 5-Rule Compliance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.analyze_xray import (
    ALL_PATHOLOGY_STEMS,
    ChestXRayPipeline,
    _CLAUSE_SPLIT_RE,
    _NEGATION_RE,
    _PROMPT_ECHO_RE,
    _SECTION_HEADER_RE,
)
from execution.build_rulebook_dataset import canonicalize_label, validate_rulebook_compliance

try:
    from execution.report_parser import compute_clinical_metrics, parse_report_findings

    HAS_REPORT_PARSER = True
except ImportError:
    HAS_REPORT_PARSER = False


# Extra clinical stem aliases mapping VLM text phrasing -> canonical 25-class ontology
_FALLBACK_CLASS_STEMS: dict[str, list[str]] = {
    "Pneumothorax": ["pneumothora"],
    "Effusion": ["effusion", "pleural fluid", "costophrenic blunting", "meniscus"],
    "Consolidation": ["consolidat", "airspace opac", "air space opac", "airspace disease"],
    "Pneumonia": ["pneumonia"],
    "Edema": ["edema", "congest", "kerley", "cephalization"],
    "Atelectasis": ["atelecta", "volume loss", "collapse"],
    "Cardiomegaly": ["cardiomegal", "enlarged heart", "enlarged cardiac", "cardiothoracic"],
    "Enlarged Cardiomediastinum": ["widened mediastin", "enlarged cardiomediastin", "mediastinal widening"],
    "Aortic Enlargement": ["aortic enlargement", "aortic tortuos", "aortic unfolding", "prominent aorta"],
    "Lung Opacity": ["opacit"],
    "Infiltration": ["infiltrat"],
    "ILD": ["interstitial lung", "reticular opacit", "reticulonodular"],
    "Mass": ["mass"],
    "Nodule": ["nodul", "granuloma"],
    "Lung Cavity": ["cavit"],
    "Lung Cyst": ["cyst", "bulla", "bleb"],
    "Lung Lesion": ["lesion"],
    "Emphysema": ["emphysem", "hyperinflat", "flattened hemidiaphragm"],
    "Fibrosis": ["fibros", "reticular", "scarring"],
    "Pleural Thickening": ["pleural thickening", "apical cap"],
    "Calcification": ["calcif"],
    "Fracture": ["fractur", "cortical step"],
    "Hernia": ["hernia"],
    "Support Devices": ["pacemaker", "catheter", "endotracheal tube", "chest tube", "support device", "hardware", "sternotomy"],
}

_NEGATION_FALLBACK_RE = re.compile(
    r"\b(?:no|not|without|negative for|free of|absence of|absent|clear of|"
    r"unremarkable|within normal limits|normal|ruled out|no evidence of|"
    r"no radiographic evidence of|neither)\b.*",
    flags=re.IGNORECASE,
)


def _fallback_parse_report_status(report_text: str) -> dict[str, str]:
    """
    Baseline negation-aware parser used in Phase 0 prior to `report_parser.py`.
    Returns canonical finding -> "present" | "absent".
    """
    present: set[str] = set()
    if not report_text:
        return {}

    for raw_line in re.split(r"(?<=[.!?])\s+|\n+", report_text):
        sent = raw_line.strip()
        if not sent:
            continue
        low = sent.lower()
        if _SECTION_HEADER_RE.match(low) or _PROMPT_ECHO_RE.search(low):
            continue
        if low.startswith("no radiographic evidence of") or low.startswith("no evidence of"):
            continue
        if re.match(r"^(?:cad (?:additionally )?flagged suspected|candidate ensemble cad)\b", low):
            continue

        for clause in _CLAUSE_SPLIT_RE.split(low):
            clause = clause.strip()
            if not clause:
                continue
            unnegated = _NEGATION_FALLBACK_RE.sub(" ", clause)
            unnegated = _NEGATION_RE.sub(" ", unnegated)
            for canon, stems in _FALLBACK_CLASS_STEMS.items():
                if any(st in unnegated for st in stems):
                    present.add(canon)
            for raw_lbl, stems in ALL_PATHOLOGY_STEMS.items():
                canon_lbl = canonicalize_label(raw_lbl) or raw_lbl
                if any(st in unnegated for st in stems):
                    present.add(canon_lbl)

    return {c: "present" for c in sorted(present)}


def normalize_manifest_findings(raw_findings: Any) -> list[str]:
    """Converts string or dict findings list into sorted unique canonical labels."""
    out: set[str] = set()
    for f in raw_findings or []:
        lbl = f.get("label") if isinstance(f, dict) else f
        if not lbl:
            continue
        canon = canonicalize_label(str(lbl))
        if canon:
            out.add(canon)
    return sorted(out)


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Loads either JSONL (`.jsonl`) or JSON list (`.json`) manifest cleanly."""
    p = Path(manifest_path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl" or not text.startswith("["):
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw_items = json.loads(text)

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        study_id = str(item.get("study_id") or item.get("id") or f"study_{idx:04d}")
        patient_id = str(item.get("patient_id") or item.get("group_id") or study_id)
        gt_findings = normalize_manifest_findings(item.get("gt_findings", item.get("findings", [])))
        normalized.append({
            "study_id": study_id,
            "patient_id": patient_id,
            "source": str(item.get("source") or item.get("dataset") or "unknown"),
            "image_path": str(item.get("image_path") or item["path"]),
            "gt_findings": gt_findings,
            "gt_is_normal": len(gt_findings) == 0,
        })
    return normalized


def _score_record(study: dict[str, Any], report_text: str, latency_sec: float) -> dict[str, Any]:
    """Parses a single study report into structured per-study clinical scoring fields."""
    if HAS_REPORT_PARSER:
        status_map = parse_report_findings(report_text)
    else:
        status_map = _fallback_parse_report_status(report_text)

    pred_present = sorted(k for k, v in status_map.items() if v == "present")
    pred_uncertain = sorted(k for k, v in status_map.items() if v == "uncertain")
    rulebook_ok, rulebook_errs = validate_rulebook_compliance(report_text)
    reported_normal = len(pred_present) == 0

    gt_is_normal = len(study["gt_findings"]) == 0
    return {
        "study_id": study["study_id"],
        "patient_id": study["patient_id"],
        "source": study["source"],
        "image_path": study["image_path"],
        "gt_findings": list(study["gt_findings"]),
        "gt_is_normal": gt_is_normal,
        "report": report_text,
        "pred_findings": pred_present,
        "pred_uncertain": pred_uncertain,
        "pred_status": status_map,
        "rulebook_compliant": rulebook_ok,
        "rulebook_errors": rulebook_errs,
        "reported_normal": reported_normal,
        "normal_bias": (not gt_is_normal) and reported_normal,
        "false_alarm": gt_is_normal and (not reported_normal),
        "latency_seconds": round(float(latency_sec), 3),
    }


def _fallback_compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes per-finding Sensitivity/Specificity/F1 and overall rates."""
    if not records:
        return {
            "total_studies": 0,
            "abnormal_studies": 0,
            "normal_studies": 0,
            "normal_bias_rate": 0.0,
            "false_alarm_rate": 0.0,
            "macro_sensitivity": 0.0,
            "rulebook_compliance": 0.0,
            "combined_score": 0.0,
            "per_finding": {},
        }

    classes = sorted({lbl for r in records for lbl in r.get("gt_findings", [])})
    per_finding: dict[str, dict[str, Any]] = {}
    sens_list: list[float] = []

    for cls in classes:
        tp = fp = fn = tn = 0
        for rec in records:
            gt_pos = cls in rec.get("gt_findings", [])
            pred_pos = cls in rec.get("pred_findings", [])
            if gt_pos and pred_pos:
                tp += 1
            elif (not gt_pos) and pred_pos:
                fp += 1
            elif gt_pos and (not pred_pos):
                fn += 1
            else:
                tn += 1

        support = tp + fn
        sens = tp / support if support > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        per_finding[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
        }
        if support > 0:
            sens_list.append(sens)

    abn = [r for r in records if not r.get("gt_is_normal", len(r.get("gt_findings", [])) == 0)]
    norm = [r for r in records if r.get("gt_is_normal", len(r.get("gt_findings", [])) == 0)]

    normal_bias_rate = (sum(1 for r in abn if r.get("normal_bias")) / len(abn)) if abn else 0.0
    false_alarm_rate = (sum(1 for r in norm if r.get("false_alarm")) / len(norm)) if norm else 0.0
    macro_sens = (sum(sens_list) / len(sens_list)) if sens_list else 0.0
    compliance = sum(1 for r in records if r.get("rulebook_compliant")) / len(records)

    fa_penalty = max(0.0, false_alarm_rate - 0.25) * 1.5
    combined = macro_sens - 0.5 * normal_bias_rate - fa_penalty

    return {
        "total_studies": len(records),
        "abnormal_studies": len(abn),
        "normal_studies": len(norm),
        "normal_bias_count": sum(1 for r in abn if r.get("normal_bias")),
        "false_alarm_count": sum(1 for r in norm if r.get("false_alarm")),
        "rulebook_compliant_count": sum(1 for r in records if r.get("rulebook_compliant")),
        "normal_bias_rate": round(normal_bias_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "macro_sensitivity": round(macro_sens, 4),
        "rulebook_compliance": round(compliance, 4),
        "combined_score": round(combined, 4),
        "per_finding": per_finding,
    }


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if HAS_REPORT_PARSER:
        return compute_clinical_metrics(records)
    return _fallback_compute_metrics(records)


def _atomic_save_json(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)


def print_summary_table(summary: dict[str, Any], title: str = "BASELINE BENCHMARK SUMMARY") -> None:
    print("\n" + "=" * 78)
    print(f"{title} ({summary['total_studies']} studies: {summary['abnormal_studies']} abn, {summary['normal_studies']} norm)")
    print("=" * 78)
    print(f"  Normal-Bias Rate (Abn -> Normal) : {summary.get('normal_bias_count', 0)}/{summary['abnormal_studies']} ({summary['normal_bias_rate']:.1%})")
    print(f"  False-Alarm Rate (Norm -> Abn)   : {summary.get('false_alarm_count', 0)}/{summary['normal_studies']} ({summary['false_alarm_rate']:.1%})")
    print(f"  Macro Sensitivity (Per-Finding)  : {summary['macro_sensitivity']:.1%}")
    print(f"  5-Rule Compliance Rate           : {summary.get('rulebook_compliant_count', 0)}/{summary['total_studies']} ({summary['rulebook_compliance']:.1%})")
    print(f"  Combined Selection Score         : {summary.get('combined_score', 0.0):.4f}")
    print("-" * 78)
    print(f"  {'Finding':<26} | {'Support':>7} | {'Sens':>7} | {'Spec':>7} | {'F1':>7}")
    print("-" * 78)
    for cls, st in sorted(summary.get("per_finding", {}).items()):
        print(
            f"  {cls:<26} | {st['support']:>7} | {st['sensitivity']:>6.1%} | "
            f"{st['specificity']:>6.1%} | {st['f1']:>6.3f}"
        )
    print("=" * 78 + "\n")


def rescore_report_file(in_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Re-evaluates already-generated report texts in an existing baseline JSON using current parser."""
    src = Path(in_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    raw_records = data.get("records", [])
    rescored: list[dict[str, Any]] = []
    for r in raw_records:
        study = {
            "study_id": r["study_id"],
            "patient_id": r["patient_id"],
            "source": r["source"],
            "image_path": r["image_path"],
            "gt_findings": normalize_manifest_findings(r.get("gt_findings", [])),
        }
        rescored.append(_score_record(study, r.get("report", ""), r.get("latency_seconds", 0.0)))

    summary = compute_summary(rescored)
    meta = dict(data.get("metadata", {}))
    meta["parser"] = "execution.report_parser" if HAS_REPORT_PARSER else "fallback_v0"
    meta["rescored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {"metadata": meta, "summary": summary, "records": rescored}
    dst = Path(out_path) if out_path else src
    _atomic_save_json(dst, payload)
    print_summary_table(summary, title=f"RESCORED: {dst.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable baseline benchmark on fixed held-out manifest.")
    parser.add_argument("--manifest", default="./.tmp/reports/test_manifest.jsonl")
    parser.add_argument("--out", default="./.tmp/reports/baseline_v1.json")
    parser.add_argument("--adapter", default="./chestxray_smolvlm_lora")
    parser.add_argument("--mode", choices=["pure_vlm", "audited"], default="pure_vlm")
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional subset cap (0 = all).")
    parser.add_argument("--overwrite", action="store_true", help="Ignore prior checkpoint and run from scratch.")
    parser.add_argument("--rescore", type=str, default=None, help="Re-score an existing output JSON without inference.")
    args = parser.parse_args()

    if args.rescore:
        rescore_report_file(args.rescore, args.out if args.out != "./.tmp/reports/baseline_v1.json" else args.rescore)
        return

    studies = load_manifest(args.manifest)
    if args.max_samples > 0:
        studies = studies[: args.max_samples]

    out_path = Path(args.out)
    records_by_id: dict[str, dict[str, Any]] = {}
    if out_path.exists() and not args.overwrite:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("records", []):
                records_by_id[str(r["study_id"])] = r
            if records_by_id:
                print(f"[Resume] Loaded {len(records_by_id)} completed studies from {out_path}")
        except Exception as exc:
            print(f"[Resume] Could not read existing {out_path} ({exc}); starting fresh.")

    pending = [s for s in studies if s["study_id"] not in records_by_id]
    print(f"[Benchmark] Total={len(studies)} | Done={len(records_by_id)} | Remaining={len(pending)}")

    llm_device_str = args.llm_device
    llm_dtype_str = "unknown"

    if pending:
        pipeline = ChestXRayPipeline(
            adapter_path=args.adapter,
            use_lora=True,
            load_cad=(args.mode == "audited"),
            llm_device=args.llm_device,
        )
        llm_device_str = pipeline.llm_device
        llm_dtype_str = str(pipeline.llm_dtype)

        for idx, study in enumerate(pending, start=1):
            with Image.open(study["image_path"]) as img:
                img.load()
                res = pipeline.analyze(img, mode=args.mode)
            rec = _score_record(study, res["report"], res.get("latency_seconds", 0.0))
            records_by_id[study["study_id"]] = rec

            gt_disp = ",".join(study["gt_findings"]) if study["gt_findings"] else "NORMAL"
            pred_disp = ",".join(rec["pred_findings"]) if rec["pred_findings"] else "NORMAL"
            print(
                f"[{len(records_by_id)}/{len(studies)}] {study['study_id'][:24]:<24} | "
                f"GT={gt_disp[:24]:<24} | Pred={pred_disp[:24]:<24} | "
                f"NormBias={rec['normal_bias']} | {rec['latency_seconds']:.2f}s",
                flush=True,
            )

            if idx % max(1, args.save_every) == 0:
                ordered = [records_by_id[s["study_id"]] for s in studies if s["study_id"] in records_by_id]
                summary = compute_summary(ordered)
                payload = {
                    "metadata": {
                        "manifest": args.manifest,
                        "adapter": args.adapter,
                        "mode": args.mode,
                        "llm_device": llm_device_str,
                        "llm_dtype": llm_dtype_str,
                        "parser": "execution.report_parser" if HAS_REPORT_PARSER else "fallback_v0",
                        "total_target_studies": len(studies),
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                    "summary": summary,
                    "records": ordered,
                }
                _atomic_save_json(out_path, payload)
                print(f"[Checkpoint] Saved {len(ordered)}/{len(studies)} studies -> {out_path}", flush=True)

    ordered_final = [records_by_id[s["study_id"]] for s in studies if s["study_id"] in records_by_id]
    final_summary = compute_summary(ordered_final)
    final_payload = {
        "metadata": {
            "manifest": args.manifest,
            "adapter": args.adapter,
            "mode": args.mode,
            "llm_device": llm_device_str,
            "llm_dtype": llm_dtype_str,
            "parser": "execution.report_parser" if HAS_REPORT_PARSER else "fallback_v0",
            "total_target_studies": len(studies),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "summary": final_summary,
        "records": ordered_final,
    }
    _atomic_save_json(out_path, final_payload)
    print(f"[Benchmark] Saved final JSON -> {out_path}")
    print_summary_table(final_summary, title=f"BENCHMARK: {out_path.name}")


if __name__ == "__main__":
    main()
