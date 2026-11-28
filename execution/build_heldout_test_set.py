"""
ChestXStudio-VLM: Fixed Patient-Grouped Held-Out Test Set Builder
Path: execution/build_heldout_test_set.py

Builds a fixed, patient-grouped held-out evaluation manifest of 300 unseen studies
(100 Normal, 200 Abnormal covering >= 8 thoracic etiologies) from unseen test/val splits:
  - natealberti/ChestX-Det (split="test", with verified Deepwise JSON recovery)
  - JasonZZ0601/PadChest-GR-Xray (split="test", grouped strictly by PatientID)
  - nwirandx/siim-pneumothorax-256-2400 (split="test", positive Pneumothorax)
  - danjacobellis/chexpert (split="validation", ClassLabel 0/1/2/3 mapping, grouped by patient ID)

Saves proportional 512px PNGs under `.tmp/reports/test_images/` without pre-applying
Phase 1 contrast stretching so baseline vs. Phase 1 preprocessing differences can be
measured faithfully on identical held-out studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.build_rulebook_dataset import (
    CHESTX_DET_ID2LABEL,
    PATHOLOGY_ONTOLOGY,
    _extract_mask_boxes,
    _fast_image_fingerprint,
    _fetch_official_chestx_det_json,
    _findings_from_deepwise_item,
    _merge_component_zones,
    _normalize_padchest_zone,
    _parse_padchest_locations,
    bbox_to_anatomical_zone,
    canonicalize_label,
)

TARGET_FINDING_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Pneumothorax", ("Pneumothorax",)),
    ("Effusion", ("Effusion",)),
    ("Consolidation", ("Consolidation",)),
    ("Cardiomegaly", ("Cardiomegaly",)),
    ("Edema", ("Edema",)),
    ("Nodule/Mass", ("Nodule", "Mass")),
    ("Atelectasis", ("Atelectasis",)),
    ("Fracture", ("Fracture",)),
]


def _save_raw_aspect_512(pil_img: Image.Image, out_path: Path, max_dim: int = 512) -> tuple[str, int, int]:
    """
    Saves an image with proportional max-dimension <= 512px PNG WITHOUT altering
    raw dynamic range or squashing aspect ratio (safe for 8-bit and 16-bit images).
    Returns: (path_str, saved_w, saved_h)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pil_img)
    h, w = arr.shape[:2]
    scale = min(max_dim / max(w, 1), max_dim / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resample = getattr(Image, "Resampling", Image).BILINEAR
    if not out_path.exists():
        if (new_w, new_h) != (w, h):
            if arr.dtype == np.uint16 or pil_img.mode in ("I", "I;16", "I;16L", "I;16B", "F"):
                if arr.ndim == 3:
                    arr = arr[..., 0]
                f_img = Image.fromarray(arr.astype(np.float32), mode="F")
                resized = np.clip(np.round(np.asarray(f_img.resize((new_w, new_h), resample))), 0, 65535).astype(np.uint16)
                Image.fromarray(resized).save(out_path, format="PNG")
            else:
                pil_img.resize((new_w, new_h), resample).save(out_path, format="PNG")
        else:
            pil_img.save(out_path, format="PNG")
    else:
        with Image.open(out_path) as existing:
            new_w, new_h = existing.size

    return str(out_path), new_w, new_h


def _chestxdet_patient_id(item: dict[str, Any] | None, idx: int) -> str:
    if item and item.get("file_name"):
        stem = Path(str(item["file_name"])).stem
        return f"chestxdet_{stem}"
    return f"chestxdet_test_{idx:06d}"


def _collect_chestx_det_test(cand_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("[Holdout Builder] Streaming natealberti/ChestX-Det (split=test)...", flush=True)
    ds = load_dataset("natealberti/ChestX-Det", split="test", streaming=True)
    out_dir = cand_dir / "chestx_det_test"
    records: list[dict[str, Any]] = []
    mask_sigs: list[tuple[bool, bool, bool, int, int]] = []

    for idx, row in enumerate(ds):
        pil_img = row.get("image")
        if pil_img is None:
            continue
        orig_w, orig_h = pil_img.size
        img_fp = _fast_image_fingerprint(pil_img)
        saved_path, sw, sh = _save_raw_aspect_512(pil_img, out_dir / f"chestxdet_test_{idx:06d}.png", 512)

        lbl_img = row.get("label")
        findings: list[dict[str, Any]] = []
        uniq_set: set[int] = set()
        if lbl_img is not None:
            lbl_arr = np.asarray(lbl_img)
            if lbl_arr.ndim == 3:
                lbl_arr = lbl_arr[..., 0]
            for uid in np.unique(lbl_arr):
                uid_int = int(uid)
                if uid_int == 255:
                    continue
                uniq_set.add(uid_int)
                if uid_int not in CHESTX_DET_ID2LABEL:
                    continue
                canon = canonicalize_label(CHESTX_DET_ID2LABEL[uid_int])
                if not canon:
                    continue
                comp_boxes = _extract_mask_boxes(lbl_arr == uid_int, min_pixels=16)
                comp_zones = [bbox_to_anatomical_zone(b, canon, img_w=orig_w, img_h=orig_h) for b in comp_boxes]
                for z in _merge_component_zones(comp_zones, canon):
                    findings.append({"label": canon, "zone": z})

        mask_sigs.append((len(uniq_set) == 0, 2 in uniq_set, 5 in uniq_set, orig_w, orig_h))
        records.append({
            "study_id": f"chestxdet_test_{idx:06d}",
            "patient_id": f"chestxdet_test_{idx:06d}",
            "source": "ChestX-Det-test",
            "image_path": saved_path,
            "findings": findings,
            "image_width": sw,
            "image_height": sh,
            "image_fingerprint": img_fp,
        })

    official = _fetch_official_chestx_det_json("test")
    if official and len(records) <= len(official):
        candidates = [
            ("native", official[: len(records)]),
            ("filename_sorted", sorted(official, key=lambda x: str(x.get("file_name", "")))[: len(records)]),
        ]
        n = max(1, len(records))
        for order_name, cand in candidates:
            empty_ok = sum(sig[0] == (len(it.get("syms") or []) == 0) for sig, it in zip(mask_sigs, cand)) / n
            cardio_ok = sum(sig[1] == ("Cardiomegaly" in (it.get("syms") or [])) for sig, it in zip(mask_sigs, cand)) / n
            eff_ok = sum(sig[2] == ("Effusion" in (it.get("syms") or [])) for sig, it in zip(mask_sigs, cand)) / n
            if empty_ok >= 0.99 and cardio_ok >= 0.98 and eff_ok >= 0.95:
                for idx, (rec, sig, it) in enumerate(zip(records, mask_sigs, cand)):
                    rec["findings"] = _findings_from_deepwise_item(it, sig[3], sig[4])
                    rec["patient_id"] = _chestxdet_patient_id(it, idx)
                print(
                    f"  [ChestX-Det-test] Deepwise JSON verified ({order_name}: empty={empty_ok:.1%}, cardio={cardio_ok:.1%}, eff={eff_ok:.1%})",
                    flush=True,
                )
                break

    print(f"  [ChestX-Det-test] Collected {len(records)} candidates", flush=True)
    return records


def _collect_chexpert_val(cand_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("[Holdout Builder] Streaming danjacobellis/chexpert (split=validation)...", flush=True)
    ds = load_dataset("danjacobellis/chexpert", split="validation", streaming=True)
    chex_cols = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    ]
    out_dir = cand_dir / "chexpert_val"
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        fl_val = str(row.get("Frontal/Lateral", "")).strip().lower()
        if fl_val not in ("0", "frontal"):
            continue
        pil_img = row.get("image")
        if pil_img is None:
            continue

        findings: list[dict[str, Any]] = []
        has_uncertain = False
        for col in chex_cols:
            val = row.get(col)
            if val == 3 or str(val).strip() == "3":
                canon = canonicalize_label(col)
                if canon:
                    findings.append({"label": canon, "zone": PATHOLOGY_ONTOLOGY[canon]["default_zone"]})
            elif val == 1 or str(val).strip() == "1":
                has_uncertain = True

        no_finding = row.get("No Finding") == 3 or str(row.get("No Finding")).strip() == "3"
        if not findings and (not no_finding or has_uncertain):
            continue

        raw_path = str(row.get("Path") or row.get("path") or "")
        pat_match = re.search(r"patient\d+", raw_path, flags=re.IGNORECASE)
        patient_id = f"chexpert_{pat_match.group(0).lower()}" if pat_match else f"chexpert_val_{idx:06d}"

        img_fp = _fast_image_fingerprint(pil_img)
        saved_path, sw, sh = _save_raw_aspect_512(pil_img, out_dir / f"chexpert_val_{idx:06d}.png", 512)
        records.append({
            "study_id": f"chexpert_val_{idx:06d}",
            "patient_id": patient_id,
            "source": "CheXpert-val",
            "image_path": saved_path,
            "findings": findings,
            "image_width": sw,
            "image_height": sh,
            "image_fingerprint": img_fp,
        })

    print(f"  [CheXpert-val] Collected {len(records)} frontal studies", flush=True)
    return records


def _collect_siim_test(cand_dir: Path, max_positive: int = 45) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print("[Holdout Builder] Streaming nwirandx/siim-pneumothorax-256-2400 (split=test)...", flush=True)
    ds = load_dataset("nwirandx/siim-pneumothorax-256-2400", split="test", streaming=True)
    out_dir = cand_dir / "siim_test"
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if len(records) >= max_positive:
            break
        if int(row.get("has_pneumo", 0) or 0) != 1:
            continue
        pil_img = row.get("image")
        if pil_img is None:
            continue
        orig_w, orig_h = pil_img.size
        img_fp = _fast_image_fingerprint(pil_img)
        saved_path, sw, sh = _save_raw_aspect_512(pil_img, out_dir / f"siim_test_{idx:06d}.png", 512)

        findings: list[dict[str, Any]] = []
        mask_img = row.get("mask")
        comp_boxes: list[list[float]] = []
        if mask_img is not None:
            m_arr = np.asarray(mask_img)
            if m_arr.ndim == 3:
                m_arr = m_arr[..., 0]
            comp_boxes = _extract_mask_boxes(m_arr > 0, min_pixels=8)
        if comp_boxes:
            zones = [bbox_to_anatomical_zone(b, "Pneumothorax", img_w=orig_w, img_h=orig_h) for b in comp_boxes]
            for z in _merge_component_zones(zones, "Pneumothorax"):
                findings.append({"label": "Pneumothorax", "zone": z})
        else:
            findings.append({"label": "Pneumothorax", "zone": PATHOLOGY_ONTOLOGY["Pneumothorax"]["default_zone"]})

        image_id = str(row.get("image_id") or f"{idx:06d}")
        records.append({
            "study_id": f"siim_test_{idx:06d}",
            "patient_id": f"siim_{image_id}",
            "source": "SIIM-Pneumothorax-test",
            "image_path": saved_path,
            "findings": findings,
            "image_width": sw,
            "image_height": sh,
            "image_fingerprint": img_fp,
        })

    print(f"  [SIIM-Pneumothorax-test] Collected {len(records)} positive studies", flush=True)
    return records


def _collect_padchest_test(cand_dir: Path, max_studies: int = 260) -> list[dict[str, Any]]:
    from datasets import load_dataset

    print(f"[Holdout Builder] Streaming JasonZZ0601/PadChest-GR-Xray (split=test, max={max_studies})...", flush=True)
    ds = load_dataset("JasonZZ0601/PadChest-GR-Xray", split="test", streaming=True)
    out_dir = cand_dir / "padchest_test"
    studies: dict[str, dict[str, Any]] = {}
    scanned_rows = 0
    max_scan_rows = max_studies * 4

    for idx, row in enumerate(ds):
        scanned_rows += 1
        pil_img = row.get("image")
        if pil_img is None:
            continue

        pat_raw = str(row.get("PatientID") or row.get("patient_id") or row.get("ImageID") or f"{idx:06d}").strip()
        img_fp = _fast_image_fingerprint(pil_img)
        study_key = f"{pat_raw}_{img_fp}"

        if study_key not in studies and len(studies) >= max_studies:
            break
        if scanned_rows > max_scan_rows and study_key not in studies:
            break

        orig_w, orig_h = pil_img.size
        if study_key not in studies:
            saved_path, sw, sh = _save_raw_aspect_512(
                pil_img,
                out_dir / f"padchest_test_{len(studies):06d}_{img_fp}.png",
                512,
            )
            studies[study_key] = {
                "study_id": f"padchest_test_{len(studies):06d}",
                "patient_id": f"padchest_{pat_raw}",
                "source": "PadChest-GR-test",
                "image_path": saved_path,
                "findings": [],
                "image_width": sw,
                "image_height": sh,
                "image_fingerprint": img_fp,
                "_seen_pairs": set(),
                "_explicit_normal": False,
                "_has_unmapped_abn": False,
            }

        entry = studies[study_key]
        raw_lbl = row.get("label") if row.get("label") is not None else row.get("labels", "")
        parts = [str(x).strip() for x in raw_lbl] if isinstance(raw_lbl, list) else [p.strip() for p in str(raw_lbl).split(",")]
        parts = [p for p in parts if p]
        loc_str = _parse_padchest_locations(row.get("locations"))
        raw_note = str(row.get("sentence_en") or "").strip()
        has_note = bool(raw_note and raw_note.lower() != "none")

        boxes = row.get("boxes") or row.get("bbox") or None
        first_box = None
        if isinstance(boxes, list) and boxes:
            if isinstance(boxes[0], (list, tuple)) and len(boxes[0]) == 4:
                first_box = boxes[0]
            elif len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
                first_box = boxes

        if parts and all(p.lower() in ("normal", "no finding", "no findings") for p in parts):
            if has_note:
                entry["_has_unmapped_abn"] = True
            else:
                entry["_explicit_normal"] = True
            continue

        mapped_any = False
        for part in parts:
            if part.lower() in ("normal", "no finding", "no findings"):
                continue
            canon = canonicalize_label(part)
            if canon:
                mapped_any = True
                zone = bbox_to_anatomical_zone(first_box, canon, img_w=orig_w, img_h=orig_h) if first_box else _normalize_padchest_zone(loc_str, canon)
                pair = (canon, zone)
                if pair not in entry["_seen_pairs"]:
                    entry["_seen_pairs"].add(pair)
                    entry["findings"].append({"label": canon, "zone": zone})
            else:
                entry["_has_unmapped_abn"] = True

        if not mapped_any and (parts or has_note):
            entry["_has_unmapped_abn"] = True

    records: list[dict[str, Any]] = []
    for entry in studies.values():
        if entry["findings"] or (entry["_explicit_normal"] and not entry["_has_unmapped_abn"]):
            records.append({
                "study_id": entry["study_id"],
                "patient_id": entry["patient_id"],
                "source": entry["source"],
                "image_path": entry["image_path"],
                "findings": entry["findings"],
                "image_width": entry["image_width"],
                "image_height": entry["image_height"],
                "image_fingerprint": entry["image_fingerprint"],
            })

    print(f"  [PadChest-GR-test] Collected {len(records)} clean studies ({scanned_rows} rows scanned)", flush=True)
    return records


def select_patient_grouped_holdout(
    candidates: list[dict[str, Any]],
    target_normal: int = 100,
    target_abnormal: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Groups candidates strictly by `patient_id` (max 1 study per patient), deduplicates
    image fingerprints, selects balanced Normal & stratified multi-etiology Abnormal studies,
    and asserts zero patient overlap across all selected records.
    """
    rng = random.Random(seed)
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_fp: set[str] = set()

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for rec in shuffled:
        fp = str(rec.get("image_fingerprint") or "")
        if fp and fp in seen_fp:
            continue
        if fp:
            seen_fp.add(fp)
        pid = str(rec["patient_id"])
        by_patient[pid].append(rec)

    # Collapse strictly to 1 canonical study per unique patient_id
    # If a patient has multiple studies, prefer abnormal (to preserve positive clinical status)
    unique_patient_studies: list[dict[str, Any]] = []
    for pid, group in by_patient.items():
        group_sorted = sorted(group, key=lambda r: (-len(r.get("findings") or []), r["study_id"]))
        chosen = dict(group_sorted[0])
        chosen["patient_id"] = pid
        unique_patient_studies.append(chosen)

    normal_pool = [r for r in unique_patient_studies if not r.get("findings")]
    abnormal_pool = [r for r in unique_patient_studies if r.get("findings")]

    if len(normal_pool) < target_normal:
        raise RuntimeError(f"Insufficient unique-patient normal studies: {len(normal_pool)} < {target_normal}")
    if len(abnormal_pool) < target_abnormal:
        raise RuntimeError(f"Insufficient unique-patient abnormal studies: {len(abnormal_pool)} < {target_abnormal}")

    # Balance normal selection round-robin across sources
    norm_by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in normal_pool:
        norm_by_src[r["source"]].append(r)
    for src_recs in norm_by_src.values():
        rng.shuffle(src_recs)

    selected_normal: list[dict[str, Any]] = []
    src_keys = sorted(norm_by_src.keys())
    ptrs = {k: 0 for k in src_keys}
    while len(selected_normal) < target_normal:
        progressed = False
        for k in src_keys:
            if len(selected_normal) >= target_normal:
                break
            idx = ptrs[k]
            if idx < len(norm_by_src[k]):
                selected_normal.append(norm_by_src[k][idx])
                ptrs[k] += 1
                progressed = True
        if not progressed:
            break

    # Greedy diversity selection for abnormal studies across the 8 target etiology groups
    group_names = [g[0] for g in TARGET_FINDING_GROUPS]
    group_map = {g[0]: set(g[1]) for g in TARGET_FINDING_GROUPS}
    group_counts: Counter[str] = Counter({g: 0 for g in group_names})
    selected_abnormal: list[dict[str, Any]] = []
    remaining_abn = list(abnormal_pool)
    rng.shuffle(remaining_abn)

    def _rec_groups(rec: dict[str, Any]) -> list[str]:
        lbls = {f["label"] for f in rec.get("findings") or []}
        return [g for g in group_names if lbls & group_map[g]]

    # First ensure every target finding group reaches balanced representation (~25 each for 8 groups)
    target_per_group = max(1, target_abnormal // len(group_names))
    while len(selected_abnormal) < target_abnormal and remaining_abn:
        # Pick rarest underrepresented target group first
        under_groups = sorted(group_names, key=lambda g: (group_counts[g], rng.random()))
        best_idx = -1
        best_score = -1.0

        for i, rec in enumerate(remaining_abn):
            r_groups = _rec_groups(rec)
            # Prioritize records hitting the rarest underrepresented target groups
            need_score = sum(
                max(0, target_per_group - group_counts[g]) * 10.0 + (1.0 / (1.0 + group_counts[g]))
                for g in r_groups
            )
            if not r_groups:
                need_score = 0.05
            # Slight bonus if hitting the single rarest group
            if under_groups and under_groups[0] in r_groups:
                need_score += 25.0
            if need_score > best_score:
                best_score = need_score
                best_idx = i

        chosen = remaining_abn.pop(best_idx if best_idx >= 0 else 0)
        selected_abnormal.append(chosen)
        for g in _rec_groups(chosen):
            group_counts[g] += 1

    final_selected = selected_normal + selected_abnormal
    rng.shuffle(final_selected)

    # Strict assertions: exact counts & zero patient overlap
    pids = [r["patient_id"] for r in final_selected]
    assert len(final_selected) == target_normal + target_abnormal, "Unexpected total study count"
    assert len(set(pids)) == len(final_selected), "Patient overlap detected inside held-out manifest!"
    norm_pids = {r["patient_id"] for r in selected_normal}
    abn_pids = {r["patient_id"] for r in selected_abnormal}
    assert norm_pids.isdisjoint(abn_pids), "Patient overlap between Normal and Abnormal studies!"

    return final_selected


def build_manifest(
    out_dir: Path = Path("./.tmp/reports"),
    target_normal: int = 100,
    target_abnormal: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    cand_dir = out_dir / "_cand_cache"
    img_dir = out_dir / "test_images"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    all_cands: list[dict[str, Any]] = []
    all_cands.extend(_collect_chestx_det_test(cand_dir))
    all_cands.extend(_collect_chexpert_val(cand_dir))
    all_cands.extend(_collect_siim_test(cand_dir, max_positive=45))
    all_cands.extend(_collect_padchest_test(cand_dir, max_studies=260))

    selected = select_patient_grouped_holdout(
        all_cands,
        target_normal=target_normal,
        target_abnormal=target_abnormal,
        seed=seed,
    )

    manifest_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected, start=1):
        src_img = Path(rec["image_path"])
        dst_img = img_dir / f"test_{idx:04d}_{src_img.name}"
        shutil.move(str(src_img), str(dst_img))
        gt_findings = sorted({f["label"] for f in rec.get("findings") or []})
        manifest_rows.append({
            "study_id": rec["study_id"],
            "patient_id": rec["patient_id"],
            "source": rec["source"],
            "image_path": str(dst_img),
            "gt_findings": gt_findings,
            "gt_zones": rec.get("findings") or [],
            "gt_is_normal": len(gt_findings) == 0,
            "image_width": rec.get("image_width"),
            "image_height": rec.get("image_height"),
            "image_fingerprint": rec.get("image_fingerprint"),
        })

    if cand_dir.exists():
        shutil.rmtree(cand_dir)

    manifest_path = out_dir / "test_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    pids_path = out_dir / "heldout_patient_ids.json"
    with open(pids_path, "w", encoding="utf-8") as f:
        json.dump(sorted({r["patient_id"] for r in manifest_rows}), f, indent=2)

    class_counts: Counter[str] = Counter()
    src_counts: Counter[str] = Counter()
    for row in manifest_rows:
        src_counts[row["source"]] += 1
        if row["gt_is_normal"]:
            class_counts["Normal"] += 1
        else:
            for lbl in row["gt_findings"]:
                class_counts[lbl] += 1

    covered_groups = [
        g_name for g_name, g_lbls in TARGET_FINDING_GROUPS
        if sum(class_counts[lbl] for lbl in g_lbls) > 0
    ]
    assert len(covered_groups) >= 8, f"Expected >= 8 finding types covered, got {covered_groups}"

    print("\n" + "=" * 72)
    print(f"HELD-OUT TEST MANIFEST BUILT: {len(manifest_rows)} studies ({target_normal} Normal, {target_abnormal} Abnormal)")
    print(f"Unique Patient IDs: {len({r['patient_id'] for r in manifest_rows})} (0 overlap asserted)")
    print(f"Covered Priority Groups ({len(covered_groups)}/8): {covered_groups}")
    print(f"Sources: {dict(src_counts)}")
    print(f"Class Distribution: {dict(class_counts.most_common())}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 72)
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patient-grouped 300-study held-out CXR test set.")
    parser.add_argument("--out-dir", default="./.tmp/reports")
    parser.add_argument("--normal", type=int, default=100)
    parser.add_argument("--abnormal", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifest(Path(args.out_dir), target_normal=args.normal, target_abnormal=args.abnormal, seed=args.seed)


if __name__ == "__main__":
    main()
