"""
ChestXStudio-VLM: Multi-Etiology & CAD-Authenticator Acceptance Gate Benchmark
Path: execution/benchmark_etiologies.py

Evaluates Pure VLM (LoRA standalone), CAD-Authenticator False-Positive Rejection,
and Audited Hybrid across either:
  - `./.tmp/virtual_test_suite/manifest.json` (legacy virtual test manifest)
  - `.tmp/eval_holdout/holdout_manifest.json` (unseen ChestX-Det / PadChest-GR test splits)

Tracks all 4 model Acceptance Gates automatically:
  Gate 1: Pure VLM Abnormal Sensitivity >= 75%
  Gate 2: Pure VLM Normal-Bias Rate on Abnormal Studies < 10%
  Gate 3: Pure VLM 5-Rule Format Compliance >= 98%
  Gate 4: CAD-Authenticator False-Positive Rejection >= 85%
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from execution.analyze_xray import ChestXRayPipeline, _CLAUSE_SPLIT_RE, _NEGATION_RE
from execution.build_rulebook_dataset import (
    PATHOLOGY_ONTOLOGY,
    _build_cad_prompt,
    validate_rulebook_compliance,
)

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
    "enlarged cardiomediastinum": ["cardiomediastin", "mediastinal widening", "widened mediastin"],
    "aortic enlargement": ["aort", "tortuous", "unfolding"],
    "atelectasis": ["atelecta", "volume loss", "subsegmental", "collapse"],
    "emphysema": ["emphysem", "hyperinflat", "copd"],
    "fibrosis": ["fibros", "reticular", "interstitial"],
    "ild": ["interstitial", "reticular", "ild"],
    "nodule": ["nodul", "lesion", "granuloma"],
    "mass": ["mass", "lesion"],
    "lung cavity": ["cavit"],
    "lung cyst": ["cyst", "bulla", "bleb"],
    "lung opacity": ["opacit", "consolidat", "infiltrat"],
    "infiltration": ["infiltrat", "opacit"],
    "lung lesion": ["lesion", "nodul", "mass", "opacit"],
    "fracture": ["fractur", "cortical"],
    "calcification": ["calcif", "granuloma"],
    "calcific": ["calcif", "granuloma"],
    "pleural thickening": ["pleural thickening", "apical cap", "thickening"],
    "hernia": ["hernia"],
    "support devices": ["device", "pacemaker", "catheter", "tube", "wire", "lead", "hardware"],
    "normal": ["no acute", "normal", "no active"],
}

FP_INJECTION_POOL: list[tuple[str, str]] = [
    ("Pneumothorax", "right apical pleural space"),
    ("Effusion", "left costophrenic angle and lower pleural space"),
    ("Consolidation", "right lower lung zone / lung base"),
    ("Edema", "bilateral perihilar and basilar lung fields"),
    ("Cardiomegaly", "central mediastinum and cardiac silhouette"),
    ("Mass", "right upper lung zone (apical)"),
    ("Nodule", "left mid lung zone"),
    ("Fracture", "right osseous thoracic cage"),
]

_CAD_FLAG_PREFIX_RE = re.compile(
    r"cad (?:additionally )?flagged suspected [^.;]+[.;]?",
    flags=re.IGNORECASE,
)


def matches_etiology(report_text: str, gt_bucket: str) -> bool:
    """Checks positive/unnegated clinical findings only (ignores Rule 4 & CAD-flag echo clauses)."""
    bucket = gt_bucket.strip().lower()
    if bucket == "normal":
        return bool(NORMAL_CLAIM_RE.search(report_text))
    stems = ETIOLOGY_SYNONYMS.get(bucket, [bucket])
    cleaned_report = _CAD_FLAG_PREFIX_RE.sub(" ", report_text)
    for sent in re.split(r"(?<=[.!?])\s+|\n+", cleaned_report):
        s_low = sent.strip().lower()
        if not s_low or re.match(r"^(?:step\s+\d+:|no radiographic evidence of\b)", s_low):
            continue
        for clause in _CLAUSE_SPLIT_RE.split(s_low):
            unnegated = _NEGATION_RE.sub(" ", clause.strip())
            if any(st in unnegated for st in stems):
                return True
    return False


def _extract_localization_and_impression(report_text: str) -> str:
    """Extracts Sections 2 & 3 so CAD refutation prose in Section 1 is not miscounted as echoing."""
    idx2 = report_text.find("2. Spatial Localization:")
    idx3 = report_text.find("3. Clinical Impression:")
    if idx2 != -1:
        return report_text[idx2:]
    if idx3 != -1:
        return report_text[idx3:]
    return _CAD_FLAG_PREFIX_RE.sub(" ", report_text)


def _normalize_manifest_item(idx: int, item: dict[str, Any]) -> dict[str, Any]:
    """Accepts both virtual_test_suite/manifest.json and holdout_manifest.json schemas."""
    if "path" in item:
        gt_bucket = str(item.get("gt_bucket", "normal")).strip().lower()
        gt_buckets = [gt_bucket] if gt_bucket else ["normal"]
        return {
            "id": str(item.get("id", f"study_{idx:04d}")),
            "dataset": str(item.get("dataset", "virtual_suite")),
            "image_path": item["path"],
            "gt_buckets": gt_buckets,
            "gt_label": str(item.get("gt_label", gt_bucket)),
            "gt_locations": item.get("gt_locations"),
            "gt_sentence": item.get("gt_sentence"),
        }

    findings = item.get("findings") or []
    gt_buckets = sorted({str(f["label"]).strip().lower() for f in findings if f.get("label")})
    if not gt_buckets:
        gt_buckets = ["normal"]
    return {
        "id": str(item.get("group_id") or item.get("id") or f"holdout_{idx:04d}"),
        "dataset": str(item.get("source", "holdout")),
        "image_path": item["image_path"],
        "gt_buckets": gt_buckets,
        "gt_label": ", ".join(gt_buckets),
        "gt_locations": [f.get("zone") for f in findings if f.get("zone")],
        "gt_sentence": item.get("note"),
    }


def run_authenticator_fp_check(
    pipeline: ChestXRayPipeline,
    img: Image.Image,
    gt_buckets: list[str],
    idx: int,
) -> dict[str, Any]:
    """
    Injects a false-positive CAD alert that is NOT in `gt_buckets` and checks whether
    the VLM visually rejects it instead of echoing it into Localization / Impression.
    """
    gt_set = set(gt_buckets)
    chosen_canon, chosen_zone = FP_INJECTION_POOL[idx % len(FP_INJECTION_POOL)]
    for offset in range(len(FP_INJECTION_POOL)):
        cand_canon, cand_zone = FP_INJECTION_POOL[(idx + offset) % len(FP_INJECTION_POOL)]
        if cand_canon.lower() not in gt_set:
            chosen_canon, chosen_zone = cand_canon, cand_zone
            break

    meta = PATHOLOGY_ONTOLOGY[chosen_canon]
    fake_cad_summary = f"{meta['clinical_name']} at {chosen_zone}"
    prompt = _build_cad_prompt(fake_cad_summary)

    # Use 384 tokens so the full 3-section report + refutation is never truncated
    report = pipeline._generate_vlm_text(img, prompt, max_new_tokens=384)
    conclusion = _extract_localization_and_impression(report)
    echoed_fake = matches_etiology(conclusion, chosen_canon.lower())

    is_normal_study = gt_buckets == ["normal"]
    if is_normal_study:
        rejected = (not echoed_fake) and bool(NORMAL_CLAIM_RE.search(conclusion))
    else:
        rejected = not echoed_fake

    return {
        "injected_fake_cad": f"{chosen_canon} ({chosen_zone})",
        "is_normal_study": is_normal_study,
        "echoed_fake_cad": echoed_fake,
        "rejected_fake_cad": rejected,
        "authenticator_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pure VLM, CAD-Authenticator & Audited Hybrid.")
    parser.add_argument("--manifest", default="./.tmp/virtual_test_suite/manifest.json")
    parser.add_argument("--out", default="./.tmp/virtual_test_suite/benchmark_report.json")
    parser.add_argument("--adapter", default="./chestxray_smolvlm_lora")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for fast eval runs (0 = all).")
    parser.add_argument("--skip-cad", action="store_true", help="Skip Stage 1 CNN CAD to benchmark Pure VLM + Authenticator faster.")
    parser.add_argument("--test-authenticator-fp", action="store_true", help="Run Gate 4 false-positive CAD rejection test.")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        raw_manifest = json.load(f)

    if args.max_samples > 0:
        raw_manifest = raw_manifest[: args.max_samples]

    manifest = [_normalize_manifest_item(i, item) for i, item in enumerate(raw_manifest, start=1)]
    pipeline = ChestXRayPipeline(adapter_path=args.adapter, load_cad=not args.skip_cad, use_lora=True)

    results: list[dict[str, Any]] = []
    per_class_hits: dict[str, list[bool]] = defaultdict(list)

    for idx, item in enumerate(manifest, start=1):
        img = Image.open(item["image_path"])
        gt_buckets = item["gt_buckets"]
        is_normal = gt_buckets == ["normal"]

        pure_res = pipeline.analyze(img, mode="pure_vlm")
        pure_txt = pure_res["report"]

        if is_normal:
            pure_hit = matches_etiology(pure_txt, "normal")
            pure_normal_bias = False
        else:
            bucket_hits = {b: matches_etiology(pure_txt, b) for b in gt_buckets}
            for b, h in bucket_hits.items():
                per_class_hits[b].append(h)
            pure_hit = any(bucket_hits.values())
            pure_normal_bias = bool(NORMAL_CLAIM_RE.search(pure_txt)) and not pure_hit

        pure_rulebook_ok, pure_rulebook_errs = validate_rulebook_compliance(pure_txt)

        aud_hit: bool | None = None
        aud_rulebook_ok: bool | None = None
        cad_labels: list[str] = []
        near_misses: list[str] = []
        aud_txt: str | None = None

        if not args.skip_cad:
            aud_res = pipeline.analyze(img, sensitivity_mode="Balanced Clinical", use_vlm=True, mode="audited")
            aud_txt = aud_res["report"]
            aud_hit = (
                matches_etiology(aud_txt, "normal")
                if is_normal
                else any(matches_etiology(aud_txt, b) for b in gt_buckets)
            )
            aud_rulebook_ok, _ = validate_rulebook_compliance(aud_txt)
            cad_labels = [f"{d['raw_label']}({d['cal_prob']:.0%}/{d['raw_prob']:.0%})" for d in aud_res["cad_findings"]]
            near_misses = [f"{m['raw_label']}({m['cal_prob']:.0%}/{m['raw_prob']:.0%})" for m in aud_res["near_misses"]]

        auth_res: dict[str, Any] | None = None
        if args.test_authenticator_fp:
            auth_res = run_authenticator_fp_check(pipeline, img, gt_buckets, idx)

        rec: dict[str, Any] = {
            "id": item["id"],
            "dataset": item["dataset"],
            "gt_buckets": gt_buckets,
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
        if auth_res is not None:
            rec.update(auth_res)
        results.append(rec)

        auth_tag = f" | AuthRej={auth_res['rejected_fake_cad']}" if auth_res else ""
        aud_tag = f" | Audited_Hit={aud_hit}" if not args.skip_cad else ""
        print(
            f"[{idx}/{len(manifest)}] GT={item['gt_label'][:22].upper():<22} | "
            f"PureHit={str(pure_hit):<5} NormBias={str(pure_normal_bias):<5} RuleOK={str(pure_rulebook_ok):<5}"
            f"{aud_tag}{auth_tag}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    abnormal = [r for r in results if r["gt_buckets"] != ["normal"]]
    normal = [r for r in results if r["gt_buckets"] == ["normal"]]

    print("\n" + "=" * 78)
    print(f"BENCHMARK SUMMARY ({len(results)} studies: {len(abnormal)} abnormal, {len(normal)} normal):")
    if abnormal:
        sens = sum(r["pure_vlm_hit"] for r in abnormal) / len(abnormal)
        nbias = sum(r["pure_vlm_normal_bias"] for r in abnormal) / len(abnormal)
        comp = sum(r["pure_vlm_rulebook_ok"] for r in results) / len(results)
        print(f" [Gate 1] Pure VLM Abnormal Sensitivity : {sum(r['pure_vlm_hit'] for r in abnormal)}/{len(abnormal)} ({sens:.1%})  [Target >= 75%]  {'PASS' if sens >= 0.75 else 'FAIL'}")
        print(f" [Gate 2] Pure VLM Normal-Bias Rate     : {sum(r['pure_vlm_normal_bias'] for r in abnormal)}/{len(abnormal)} ({nbias:.1%})  [Target <  10%]  {'PASS' if nbias < 0.10 else 'FAIL'}")
        print(f" [Gate 3] Pure VLM 5-Rule Compliance    : {sum(r['pure_vlm_rulebook_ok'] for r in results)}/{len(results)} ({comp:.1%})  [Target >= 98%]  {'PASS' if comp >= 0.98 else 'FAIL'}")

    if args.test_authenticator_fp:
        norm_auth = [r for r in results if r.get("is_normal_study")]
        all_auth = [r for r in results if "rejected_fake_cad" in r]
        if norm_auth:
            n_rej = sum(r["rejected_fake_cad"] for r in norm_auth) / len(norm_auth)
            print(f" [Gate 4] Auth Normal-FP Rejection      : {sum(r['rejected_fake_cad'] for r in norm_auth)}/{len(norm_auth)} ({n_rej:.1%})  [Target >= 85%]  {'PASS' if n_rej >= 0.85 else 'FAIL'}")
        if all_auth:
            a_rej = sum(r["rejected_fake_cad"] for r in all_auth) / len(all_auth)
            print(f" [Gate 4b] Auth Overall-FP Rejection    : {sum(r['rejected_fake_cad'] for r in all_auth)}/{len(all_auth)} ({a_rej:.1%})")

    if not args.skip_cad and abnormal:
        aud_sens = sum(bool(r["audited_hit"]) for r in abnormal) / len(abnormal)
        aud_comp = sum(bool(r["audited_rulebook_ok"]) for r in results) / len(results)
        print(f" [Hybrid] Audited Abnormal Sensitivity  : {sum(bool(r['audited_hit']) for r in abnormal)}/{len(abnormal)} ({aud_sens:.1%})")
        print(f" [Hybrid] Audited 5-Rule Compliance     : {sum(bool(r['audited_rulebook_ok']) for r in results)}/{len(results)} ({aud_comp:.1%})")

    if per_class_hits:
        print("\nPer-Etiology Pure VLM Recall:")
        for cls_name, hits in sorted(per_class_hits.items()):
            rate = sum(hits) / len(hits)
            print(f"  - {cls_name:<26}: {sum(hits):>3}/{len(hits):<3} ({rate:.1%})")
    print(f"\nFull JSON saved to: {out_path}")


if __name__ == "__main__":
    main()