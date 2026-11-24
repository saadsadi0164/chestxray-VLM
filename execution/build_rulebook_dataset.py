"""
ChestXStudio-VLM: Multi-Dataset Ingestion & 5-Rule CoT Training Data Synthesizer
Path: execution/build_rulebook_dataset.py

Converts multi-institutional chest X-ray datasets (ChestX-Det, VinDr-CXR,
NIH ChestX-ray14, CheXpert, MIMIC-CXR, OpenI, or custom JSONL) into strictly
formatted 5-Rule Clinical Rulebook training samples for SmolVLM-500M-Instruct.

Key anti-hallucination design features:
  1. Converts bounding boxes [x1, y1, x2, y2] -> exact anatomical zones (Rule 3).
  2. Enforces Observation Before Etiology (Rule 2) & Explicit Negatives (Rule 4).
  3. Rebalances class frequencies (caps Normal ratio, oversamples acute/rare classes)
     to eliminate "Normal-biased" hallucination.
  4. Emits Dual-Mode training examples (Pure-Image CoT + CAD-Grounded CoT).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from execution.analyze_xray import SYSTEM_RULEBOOK

# =====================================================================
# 1. UNIFIED THORACIC PATHOLOGY & OBSERVATION ONTOLOGY (24 ETIOLOGIES)
# =====================================================================
PATHOLOGY_ONTOLOGY: dict[str, dict[str, Any]] = {
    "Pneumothorax": {
        "aliases": ["pneumothorax", "ptx"],
        "observation": "a fine visceral pleural line with peripheral radiolucency and absence of distal pulmonary vascular markings",
        "clinical_name": "pneumothorax",
        "urgency": 1,
        "default_zone": "apical/lateral pleural space",
    },
    "Effusion": {
        "aliases": ["effusion", "pleural effusion"],
        "observation": "homogeneous dependent opacity blunting the costophrenic angle with a concave meniscus contour",
        "clinical_name": "pleural effusion",
        "urgency": 2,
        "default_zone": "lower pleural space and costophrenic recess",
    },
    "Consolidation": {
        "aliases": ["consolidation", "airspace consolidation"],
        "observation": "confluent homogeneous parenchymal airspace opacification obscuring underlying pulmonary vessels",
        "clinical_name": "focal consolidation",
        "urgency": 3,
        "default_zone": "pulmonary parenchymal zone",
    },
    "Pneumonia": {
        "aliases": ["pneumonia"],
        "observation": "patchy-to-confluent alveolar airspace opacity with ill-defined borders",
        "clinical_name": "pneumonia / infectious airspace process",
        "urgency": 3,
        "default_zone": "pulmonary parenchyma",
    },
    "Edema": {
        "aliases": ["edema", "pulmonary edema", "vascular congestion"],
        "observation": "perihilar haziness, indistinct pulmonary vascular margins, and cephalization of upper-lobe vessels",
        "clinical_name": "pulmonary edema / vascular congestion",
        "urgency": 3,
        "default_zone": "bilateral perihilar and basilar lung fields",
    },
    "Atelectasis": {
        "aliases": ["atelectasis", "linear atelectasis", "subsegmental atelectasis"],
        "observation": "linear or wedge-shaped parenchymal density accompanied by subtle regional volume loss",
        "clinical_name": "atelectasis",
        "urgency": 5,
        "default_zone": "mid-to-lower lung zone",
    },
    "Cardiomegaly": {
        "aliases": ["cardiomegaly"],
        "observation": "transverse enlargement of the cardiac silhouette with cardiothoracic ratio exceeding 0.50",
        "clinical_name": "cardiomegaly",
        "urgency": 4,
        "default_zone": "central mediastinum and cardiac silhouette",
    },
    "Enlarged Cardiomediastinum": {
        "aliases": ["enlarged cardiomediastinum", "mediastinal widening", "widened mediastinum"],
        "observation": "widened superior and central cardiomediastinal contour",
        "clinical_name": "enlarged cardiomediastinal silhouette",
        "urgency": 4,
        "default_zone": "central mediastinum",
    },
    "Aortic Enlargement": {
        "aliases": ["aortic enlargement", "aortic tortuosity", "aortic unfolding", "prominent aorta"],
        "observation": "prominence and tortuous unfolding of the thoracic aortic knob contour",
        "clinical_name": "aortic enlargement / tortuosity",
        "urgency": 6,
        "default_zone": "superior mediastinum / aortic arch",
    },
    "Mediastinal Shift": {
        "aliases": ["mediastinal shift", "tracheal deviation"],
        "observation": "lateral displacement of the trachea and central mediastinal structures",
        "clinical_name": "mediastinal shift",
        "urgency": 1,
        "default_zone": "central mediastinum",
    },
    "Lung Opacity": {
        "aliases": ["lung opacity", "airspace opacity", "pulmonary opacity"],
        "observation": "focal or patchy parenchymal opacity partially obscuring vascular markings",
        "clinical_name": "pulmonary airspace opacity",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "Infiltration": {
        "aliases": ["infiltration", "infiltrate"],
        "observation": "ill-defined patchy interstitial and alveolar parenchymal opacification",
        "clinical_name": "pulmonary infiltrate",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "ILD": {
        "aliases": ["ild", "interstitial lung disease", "reticular opacity"],
        "observation": "diffuse reticular and reticulonodular interstitial markings throughout the parenchyma",
        "clinical_name": "interstitial lung abnormality",
        "urgency": 5,
        "default_zone": "bilateral lung fields",
    },
    "Mass": {
        "aliases": ["mass", "pulmonary mass"],
        "observation": "discrete rounded or lobulated focal opacity measuring greater than 3 cm",
        "clinical_name": "pulmonary mass",
        "urgency": 2,
        "default_zone": "pulmonary parenchyma",
    },
    "Nodule": {
        "aliases": ["nodule", "nodule/mass", "diffuse nodule", "pulmonary nodule"],
        "observation": "circumscribed focal nodular radio-opacity with distinct parenchymal borders",
        "clinical_name": "pulmonary nodule",
        "urgency": 3,
        "default_zone": "pulmonary parenchyma",
    },
    "Lung Cavity": {
        "aliases": ["lung cavity", "cavitation", "cavitary lesion"],
        "observation": "thick-walled parenchymal opacity containing a central radiolucent air-filled cavity",
        "clinical_name": "cavitary pulmonary lesion",
        "urgency": 2,
        "default_zone": "upper-to-mid lung zone",
    },
    "Lung Cyst": {
        "aliases": ["lung cyst", "bulla", "bleb"],
        "observation": "thin-walled circumscribed radiolucent air space within the parenchyma",
        "clinical_name": "pulmonary cyst / bulla",
        "urgency": 6,
        "default_zone": "lung parenchyma",
    },
    "Lung Lesion": {
        "aliases": ["lung lesion", "other lesion", "lesion"],
        "observation": "focal parenchymal lesion distorting local pulmonary architecture",
        "clinical_name": "focal pulmonary lesion",
        "urgency": 4,
        "default_zone": "lung parenchyma",
    },
    "Emphysema": {
        "aliases": ["emphysema", "hyperinflation", "copd"],
        "observation": "hyperinflated bilateral lung fields with increased radiolucency and flattened hemidiaphragms",
        "clinical_name": "pulmonary emphysema / hyperinflation",
        "urgency": 5,
        "default_zone": "bilateral lung fields",
    },
    "Fibrosis": {
        "aliases": ["fibrosis", "pulmonary fibrosis"],
        "observation": "coarse linear reticular bands with parenchymal architectural distortion and traction",
        "clinical_name": "pulmonary fibrosis",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "Pleural Thickening": {
        "aliases": ["pleural thickening", "pleural_thickening", "apical cap"],
        "observation": "focal linear thickening along the peripheral or apical pleural margin",
        "clinical_name": "pleural thickening",
        "urgency": 6,
        "default_zone": "peripheral pleural margin",
    },
    "Calcification": {
        "aliases": ["calcification", "calcified granuloma", "aortic calcification"],
        "observation": "well-defined high-attenuation punctate or curvilinear calcific density",
        "clinical_name": "thoracic calcification / calcified granuloma",
        "urgency": 7,
        "default_zone": "lung parenchyma or aortic arch",
    },
    "Fracture": {
        "aliases": ["fracture", "rib fracture", "clavicle fracture"],
        "observation": "focal cortical step-off and lucency disrupting the osseous contour",
        "clinical_name": "rib or clavicular fracture",
        "urgency": 3,
        "default_zone": "osseous thoracic cage",
    },
    "Hernia": {
        "aliases": ["hernia", "hiatal hernia"],
        "observation": "retrocardiac soft-tissue/gas-containing density projecting above the diaphragm",
        "clinical_name": "hiatal hernia",
        "urgency": 6,
        "default_zone": "retrocardiac lower mediastinum",
    },
    "Support Devices": {
        "aliases": ["support devices", "pacemaker", "endotracheal tube", "central line", "chest tube"],
        "observation": "radiopaque medical hardware / support device projecting over the thorax",
        "clinical_name": "thoracic support device",
        "urgency": 7,
        "default_zone": "mediastinum / thoracic wall",
    },
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _meta in PATHOLOGY_ONTOLOGY.items():
    _ALIAS_TO_CANONICAL[_canon.lower()] = _canon
    for _alias in _meta["aliases"]:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canon


def canonicalize_label(raw_label: str) -> str | None:
    """Maps raw dataset label names into the unified 25-pathology ontology."""
    cleaned = raw_label.strip().lower().replace("_", " ")
    if cleaned in ("no finding", "normal", "none", ""):
        return None
    if cleaned in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[cleaned]
    for alias, canon in _ALIAS_TO_CANONICAL.items():
        if alias in cleaned:
            return canon
    return None


# =====================================================================
# 2. BOUNDING-BOX -> ANATOMICAL ZONE GROUNDING (RULE 3)
# =====================================================================
def bbox_to_anatomical_zone(
    bbox: list[float] | tuple[float, float, float, float] | None,
    canonical_label: str,
    img_w: int = 1,
    img_h: int = 1,
) -> str:
    """
    Converts pixel or normalized [x1, y1, x2, y2] bounding boxes into
    radiographically accurate anatomical zones.
    Note: Image-left (cx < 0.5) is Patient's RIGHT; Image-right (cx >= 0.5) is Patient's LEFT.
    """
    if canonical_label in ("Cardiomegaly", "Enlarged Cardiomediastinum", "Mediastinal Shift"):
        return "central mediastinum and cardiac silhouette"
    if canonical_label == "Aortic Enlargement":
        return "superior mediastinum / left aortic arch"
    if canonical_label == "Hernia":
        return "retrocardiac lower mediastinal space"

    if not bbox or len(bbox) != 4:
        return PATHOLOGY_ONTOLOGY[canonical_label]["default_zone"]

    x1, y1, x2, y2 = [float(v) for v in bbox]
    # Normalize if pixel coordinates were provided
    if max(x1, y1, x2, y2) > 1.5:
        x1 /= max(img_w, 1)
        x2 /= max(img_w, 1)
        y1 /= max(img_h, 1)
        y2 /= max(img_h, 1)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    span_x = abs(x2 - x1)

    if span_x > 0.60 or (0.43 <= cx <= 0.57 and canonical_label in ("Edema", "Emphysema", "ILD")):
        side = "bilateral"
    elif cx < 0.45:
        side = "right"
    elif cx > 0.55:
        side = "left"
    else:
        side = "central/parahilar"

    if canonical_label == "Pneumothorax":
        if cy < 0.35:
            return f"{side} apical pleural space"
        if cy < 0.65:
            return f"{side} apicolateral pleural space"
        return f"{side} lateral and basilar pleural space"

    if canonical_label == "Effusion":
        if cy >= 0.50:
            return f"{side} costophrenic angle and lower pleural space"
        return f"{side} pleural space"

    if cy < 0.34:
        zone = "upper lung zone (apical)"
    elif cy < 0.66:
        zone = "mid lung zone"
    else:
        zone = "lower lung zone / lung base"

    if side == "central/parahilar":
        return f"central perihilar {zone}"
    return f"{side} {zone}"


# =====================================================================
# 3. STRICT 5-RULE REPORT SYNTHESIZER & VALIDATOR
# =====================================================================
def build_rulebook_target(
    findings: list[dict[str, Any]],
    raw_report_note: str | None = None,
) -> str:
    """
    Synthesizes a strictly rule-compliant 3-section radiological report from
    grounded dataset annotations (`findings`: [{label, zone, confidence?}]).
    """
    deduped: dict[str, dict[str, Any]] = {}
    for item in findings:
        canon = canonicalize_label(item["label"])
        if not canon:
            continue
        zone = item.get("zone") or PATHOLOGY_ONTOLOGY[canon]["default_zone"]
        if canon not in deduped:
            deduped[canon] = {
                "canon": canon,
                "zones": [zone],
                "urgency": PATHOLOGY_ONTOLOGY[canon]["urgency"],
            }
        elif zone not in deduped[canon]["zones"]:
            deduped[canon]["zones"].append(zone)

    ordered = sorted(deduped.values(), key=lambda x: x["urgency"])
    present_canons = {x["canon"] for x in ordered}

    # Rule 4: Explicit Negatives for acute emergencies not present
    acute_neg_checks = [
        ({"Pneumothorax"}, "pneumothorax"),
        ({"Effusion"}, "pleural effusion"),
        ({"Consolidation", "Pneumonia", "Lung Opacity"}, "focal lobar consolidation"),
        ({"Edema"}, "pulmonary edema"),
        ({"Fracture"}, "acute osseous fracture"),
    ]
    explicit_negatives = [
        name for keys, name in acute_neg_checks
        if not (keys & present_canons)
    ]
    neg_sentence = (
        f"No radiographic evidence of {', '.join(explicit_negatives)}."
        if explicit_negatives else ""
    )

    if not ordered:
        sec1 = (
            "The trachea and central airways are midline and patent. "
            "Bilateral lung fields are well-aerated and clear without focal airspace opacity or pleural abnormality. "
            "The cardiomediastinal silhouette is normal in size and contour, and the thoracic skeleton is intact. "
            f"{neg_sentence}"
        ).strip()
        sec2 = "Bilateral lung fields, costophrenic angles, central mediastinum, and osseous structures (unremarkable)."
        sec3 = "No acute cardiopulmonary abnormality identified on this frontal chest radiograph."
    else:
        obs_sentences: list[str] = ["Trachea and central airways are midline."]
        loc_entries: list[str] = []
        imp_entries: list[str] = []

        for idx, entry in enumerate(ordered, start=1):
            canon = entry["canon"]
            meta = PATHOLOGY_ONTOLOGY[canon]
            zone_str = " and ".join(entry["zones"][:2])
            obs_sentences.append(
                f"Visual inspection shows {meta['observation']} within the {zone_str}."
            )
            loc_entries.append(f"{meta['clinical_name'].capitalize()}: {zone_str}")
            imp_entries.append(
                f"{idx}. Radiographic findings are consistent with {meta['clinical_name']} "
                f"involving the {zone_str}."
            )

        if not ({"Cardiomegaly", "Enlarged Cardiomediastinum", "Aortic Enlargement"} & present_canons):
            obs_sentences.append("The cardiomediastinal silhouette is within normal limits.")
        if raw_report_note:
            clean_note = raw_report_note.strip().rstrip(".")
            if 10 < len(clean_note) < 160:
                obs_sentences.append(f"{clean_note}.")

        if neg_sentence:
            obs_sentences.append(neg_sentence)

        sec1 = " ".join(obs_sentences)
        sec2 = "; ".join(loc_entries) + "."
        sec3 = "\n".join(imp_entries) + "\nClinical correlation is recommended."

    return (
        f"1. Systematic Visual Findings:\n{sec1}\n\n"
        f"2. Spatial Localization:\n{sec2}\n\n"
        f"3. Clinical Impression:\n{sec3}"
    )


def validate_rulebook_compliance(report: str) -> tuple[bool, list[str]]:
    """Programmatically audits a generated/target report against the 5-rule protocol."""
    errors: list[str] = []
    required_headers = [
        "1. Systematic Visual Findings:",
        "2. Spatial Localization:",
        "3. Clinical Impression:",
    ]
    positions = [report.find(h) for h in required_headers]
    for h, pos in zip(required_headers, positions):
        if pos == -1:
            errors.append(f"Missing section header: {h}")
    if all(p != -1 for p in positions) and not (positions[0] < positions[1] < positions[2]):
        errors.append("Sections out of order (Observation must precede Impression).")

    lower = report.lower()
    if "no radiographic evidence of" not in lower and "no acute" not in lower:
        errors.append("Missing explicit negative statement (Rule 4).")

    return (len(errors) == 0), errors


# =====================================================================
# 4. DATASET LOADERS & CLASS-BALANCED SAMPLING
# =====================================================================
CHESTX_DET_ID2LABEL: dict[int, str] = {
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "Diffuse Nodule",
    6: "Effusion",
    7: "Emphysema",
    8: "Fibrosis",
    9: "Fracture",
    10: "Mass",
    11: "Nodule",
    12: "Pleural Thickening",
    13: "Pneumothorax",
}


def load_chestx_det_hf(max_samples: int | None = None) -> list[dict[str, Any]]:
    """
    Loads `natealberti/ChestX-Det` from HuggingFace (13 thoracic classes).
    Extracts exact class IDs and bounding boxes directly from the `label` pixel mask
    (`255` = Background, `1..13` = ChestX-Det pathology categories).
    """
    import numpy as np
    from datasets import load_dataset  # Lazy import so inference runtime stays slim

    ds = load_dataset("natealberti/ChestX-Det", split="train")
    records: list[dict[str, Any]] = []
    out_img_dir = Path("./.tmp/training_images/chestx_det")
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(ds):
        if max_samples and idx >= max_samples:
            break
        pil_img = row.get("image")
        lbl_img = row.get("label")
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size
        img_file = out_img_dir / f"chestx_det_{idx:06d}.png"
        if not img_file.exists():
            pil_img.save(img_file)

        findings: list[dict[str, Any]] = []
        if lbl_img is not None:
            lbl_arr = np.asarray(lbl_img)
            for uid in np.unique(lbl_arr):
                uid_int = int(uid)
                if uid_int == 255 or uid_int not in CHESTX_DET_ID2LABEL:
                    continue
                canon = canonicalize_label(CHESTX_DET_ID2LABEL[uid_int])
                if not canon:
                    continue
                ys, xs = np.where(lbl_arr == uid_int)
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None
                findings.append({
                    "label": canon,
                    "zone": bbox_to_anatomical_zone(box, canon, img_w=img_w, img_h=img_h),
                })

        records.append({
            "source": "ChestX-Det",
            "image_path": str(img_file),
            "findings": findings,
        })
    return records


def load_chexpert_hf(max_samples: int | None = None, frontal_only: bool = True) -> list[dict[str, Any]]:
    """Loads `danjacobellis/chexpert` from HuggingFace (positive label == 3)."""
    from datasets import load_dataset

    ds = load_dataset("danjacobellis/chexpert", split="train")
    chex_cols = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    ]
    out_img_dir = Path("./.tmp/training_images/chexpert")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        if frontal_only and str(row.get("Frontal/Lateral")) != "0":
            continue
        pil_img = row.get("image")
        if pil_img is None:
            continue
        img_file = out_img_dir / f"chexpert_{idx:06d}.png"
        if not img_file.exists():
            pil_img.save(img_file)
        findings = []
        for col in chex_cols:
            if str(row.get(col)) == "3":
                canon = canonicalize_label(col)
                if canon:
                    findings.append({
                        "label": canon,
                        "zone": PATHOLOGY_ONTOLOGY[canon]["default_zone"],
                    })
        records.append({"source": "CheXpert", "image_path": str(img_file), "findings": findings})
    return records


def load_padchest_gr_hf(max_samples: int | None = None) -> list[dict[str, Any]]:
    """Loads `JasonZZ0601/PadChest-GR-Xray` (grounded findings, locations & sentences)."""
    from datasets import load_dataset

    ds = load_dataset("JasonZZ0601/PadChest-GR-Xray", split="train")
    out_img_dir = Path("./.tmp/training_images/padchest_gr")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        pil_img = row.get("image")
        if pil_img is None:
            continue
        img_file = out_img_dir / f"padchest_{idx:06d}.png"
        if not img_file.exists():
            pil_img.save(img_file)
        lbl = str(row.get("label", ""))
        locs = row.get("locations") or []
        loc_str = " ".join(str(x) for x in locs) if isinstance(locs, list) else str(locs)
        findings = []
        for part in lbl.split(","):
            canon = canonicalize_label(part)
            if canon:
                findings.append({
                    "label": canon,
                    "zone": loc_str.strip() or PATHOLOGY_ONTOLOGY[canon]["default_zone"],
                })
        note = str(row.get("sentence_en") or "")
        records.append({
            "source": "PadChest-GR",
            "image_path": str(img_file),
            "findings": findings,
            "note": None if note == "None" else note,
        })
    return records


def load_vindr_csv(csv_path: str, image_dir: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Loads VinDr-CXR train.csv annotations (aggregated by image_id with bounding boxes)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = row["image_id"]
            class_name = row.get("class_name", "")
            canon = canonicalize_label(class_name)
            if canon and row.get("x_min"):
                box = [float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])]
                grouped[img_id].append({"label": canon, "bbox": box})
            else:
                grouped.setdefault(img_id, [])

    records: list[dict[str, Any]] = []
    for img_id, raw_items in grouped.items():
        if max_samples and len(records) >= max_samples:
            break
        img_path = Path(image_dir) / f"{img_id}.png"
        if not img_path.exists():
            img_path = Path(image_dir) / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        findings = [
            {"label": it["label"], "zone": bbox_to_anatomical_zone(it["bbox"], it["label"])}
            for it in raw_items
        ]
        records.append({
            "source": "VinDr-CXR",
            "image_path": str(img_path),
            "findings": findings,
        })
    return records


def load_generic_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """
    Loads custom JSONL annotations:
      {"image_path": "...", "findings": [{"label": "...", "bbox": [x1,y1,x2,y2], "zone": "..."}]}
    """
    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            norm_findings = []
            for f_item in item.get("findings", []):
                canon = canonicalize_label(f_item["label"])
                if not canon:
                    continue
                zone = f_item.get("zone") or bbox_to_anatomical_zone(f_item.get("bbox"), canon)
                norm_findings.append({"label": canon, "zone": zone})
            records.append({
                "source": item.get("source", "custom"),
                "image_path": item["image_path"],
                "findings": norm_findings,
                "note": item.get("note"),
            })
    return records


def balance_records_against_normal_bias(
    records: list[dict[str, Any]],
    max_normal_ratio: float = 0.22,
    min_class_count: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Rebalances training dataset to eradicate "Normal-biased" hallucination:
      - Downsamples Normal-only records to at most `max_normal_ratio` of total abnormal records.
      - Oversamples underrepresented etiologies (Pneumothorax, Mass, Fracture, Cavity, etc.)
        up to `min_class_count` occurrences.
    """
    rng = random.Random(seed)
    normal_recs = [r for r in records if not r["findings"]]
    abnormal_recs = [r for r in records if r["findings"]]

    if not abnormal_recs:
        return records

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in abnormal_recs:
        for f_item in rec["findings"]:
            by_class[f_item["label"]].append(rec)

    balanced = list(abnormal_recs)
    for canon, class_recs in by_class.items():
        if len(class_recs) < min_class_count and len(class_recs) > 0:
            needed = min_class_count - len(class_recs)
            balanced.extend(rng.choices(class_recs, k=needed))

    max_normals = max(1, int(len(balanced) * (max_normal_ratio / max(1e-5, 1.0 - max_normal_ratio))))
    if len(normal_recs) > max_normals:
        normal_recs = rng.sample(normal_recs, max_normals)

    combined = balanced + normal_recs
    rng.shuffle(combined)
    return combined


def format_vlm_chat_sample(
    record: dict[str, Any],
    include_cad_context: bool = False,
) -> dict[str, Any]:
    """Formats a record into SmolVLM chat messages + validates 5-Rule compliance."""
    target_report = build_rulebook_target(record["findings"], raw_report_note=record.get("note"))
    ok, errs = validate_rulebook_compliance(target_report)
    if not ok:
        raise ValueError(f"Generated target failed rulebook validation: {errs}")

    if include_cad_context:
        if not record["findings"]:
            cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold."
        else:
            cad_summary = "; ".join(
                f"{PATHOLOGY_ONTOLOGY[f['label']]['clinical_name']} at {f['zone']}"
                for f in record["findings"][:4]
            )
        user_text = (
            f"Verified Ensemble CAD findings: {cad_summary}.\n"
            "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
        )
    else:
        user_text = "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."

    return {
        "source": record.get("source", "unknown"),
        "image_path": record["image_path"],
        "labels": [f["label"] for f in record["findings"]],
        "cad_conditioned": include_cad_context,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_report}]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced 5-Rule CoT training JSONL for SmolVLM.")
    parser.add_argument("--chestx-det", action="store_true", help="Include HuggingFace natealberti/ChestX-Det.")
    parser.add_argument("--chexpert-hf", action="store_true", help="Include HuggingFace danjacobellis/chexpert.")
    parser.add_argument("--padchest-gr", action="store_true", help="Include HuggingFace JasonZZ0601/PadChest-GR-Xray.")
    parser.add_argument("--max-per-source", type=int, default=None, help="Cap samples per dataset source.")
    parser.add_argument("--vindr-csv", type=str, default=None, help="Path to VinDr-CXR train.csv.")
    parser.add_argument("--vindr-images", type=str, default=None, help="Path to VinDr-CXR images folder.")
    parser.add_argument("--custom-jsonl", type=str, default=None, help="Path to custom JSONL manifest.")
    parser.add_argument("--max-normal-ratio", type=float, default=0.22, help="Max fraction of Normal studies.")
    parser.add_argument("--min-class-count", type=int, default=200, help="Oversample target per etiology.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split fraction.")
    parser.add_argument("--out-dir", type=str, default="./.tmp/rulebook_dataset", help="Output directory.")
    parser.add_argument("--smoke-sample", action="store_true", help="Generate a tiny 2-sample demo dataset from local image.")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if args.smoke_sample:
        sample_img = "./1002_IM-0004-1001.dcm.png"
        if not os.path.exists(sample_img):
            raise FileNotFoundError(f"Smoke test image missing: {sample_img}")
        records = [
            {"source": "smoke_normal", "image_path": sample_img, "findings": []},
            {"source": "smoke_aortic", "image_path": sample_img, "findings": [
                {"label": "Aortic Enlargement", "zone": "superior mediastinum / left aortic arch"}
            ]},
        ]
    else:
        if args.chestx_det:
            records.extend(load_chestx_det_hf(max_samples=args.max_per_source))
        if args.chexpert_hf:
            records.extend(load_chexpert_hf(max_samples=args.max_per_source))
        if args.padchest_gr:
            records.extend(load_padchest_gr_hf(max_samples=args.max_per_source))
        if args.vindr_csv and args.vindr_images:
            records.extend(load_vindr_csv(args.vindr_csv, args.vindr_images, max_samples=args.max_per_source))
        if args.custom_jsonl:
            records.extend(load_generic_jsonl(args.custom_jsonl))

    if not records:
        print("No dataset inputs provided. Pass --chestx-det, --vindr-csv, --custom-jsonl, or --smoke-sample.")
        return

    if not args.smoke_sample:
        records = balance_records_against_normal_bias(
            records,
            max_normal_ratio=args.max_normal_ratio,
            min_class_count=args.min_class_count,
        )

    # Emit BOTH Pure-Image CoT and CAD-Grounded CoT examples for every study
    chat_samples: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for rec in records:
        if not rec["findings"]:
            class_counts["Normal"] += 1
        else:
            for f_item in rec["findings"]:
                class_counts[f_item["label"]] += 1
        chat_samples.append(format_vlm_chat_sample(rec, include_cad_context=False))
        chat_samples.append(format_vlm_chat_sample(rec, include_cad_context=True))

    random.Random(42).shuffle(chat_samples)
    val_count = max(1, int(len(chat_samples) * args.val_ratio))
    val_samples = chat_samples[:val_count]
    train_samples = chat_samples[val_count:] if len(chat_samples) > val_count else chat_samples

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, subset in (("train.jsonl", train_samples), ("val.jsonl", val_samples)):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for sample in subset:
                f.write(json.dumps(sample) + "\n")

    print(f"[Dataset Builder] Wrote {len(train_samples)} train and {len(val_samples)} val samples to {out_dir}")
    print("[Dataset Builder] Class distribution:", dict(class_counts.most_common()))


if __name__ == "__main__":
    main()
