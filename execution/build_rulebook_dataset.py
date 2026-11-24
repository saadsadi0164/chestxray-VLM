"""
ChestXStudio-VLM: Multi-Dataset Ingestion & 5-Rule CoT Training Data Synthesizer
Path: execution/build_rulebook_dataset.py

Converts multi-institutional chest X-ray datasets (ChestX-Det, PadChest-GR,
CheXpert, SIIM-Pneumothorax, VinDr-CXR, or custom JSONL) into strictly formatted
5-Rule Clinical Rulebook training samples for SmolVLM-500M-Instruct.

Upgrades in v2 audited synthesizer:
1. Disk-safe HuggingFace streaming (`streaming=True`) + on-the-fly 512x512 RGB JPEG
   caching into `--img-cache-dir` (safe for Kaggle's 20 GB /kaggle/working cap).
2. Patient/image-level train/val splitting BEFORE oversampling or mode expansion
   (zero data leakage between train.jsonl and val.jsonl).
3. Word-boundary canonical label matching (fixes "ild" matching inside "mild" and
   blocks negated raw labels).
4. 3-Mode "Rulebook + Authenticator" training mix:
   - 40% Pure-Image CoT
   - 35% CAD-Aligned Verification
   - 25% CAD-Authenticator Hard Negatives / Rejection (Normal FP, Abnormal Distractor,
     and Laterality Flip subtypes) using identical CAD prompt phrasing so the VLM
     must inspect pixels rather than memorize prompt cues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from execution.analyze_xray import SYSTEM_RULEBOOK

# =====================================================================
# 1. UNIFIED THORACIC PATHOLOGY & OBSERVATION ONTOLOGY (25 ETIOLOGIES)
# =====================================================================
PATHOLOGY_ONTOLOGY: dict[str, dict[str, Any]] = {
    "Pneumothorax": {
        "aliases": ["pneumothorax", "ptx"],
        "observation": "a fine visceral pleural line with peripheral radiolucency and absence of distal pulmonary vascular markings",
        "neg_visual": "intact peripheral pulmonary vascular markings extending to the chest wall without a visible visceral pleural line",
        "clinical_name": "pneumothorax",
        "urgency": 1,
        "default_zone": "apical/lateral pleural space",
    },
    "Effusion": {
        "aliases": ["effusion", "pleural effusion"],
        "observation": "homogeneous dependent opacity blunting the costophrenic angle with a concave meniscus contour",
        "neg_visual": "sharp, clear costophrenic angles without dependent meniscus blunting",
        "clinical_name": "pleural effusion",
        "urgency": 2,
        "default_zone": "lower pleural space and costophrenic recess",
    },
    "Consolidation": {
        "aliases": ["consolidation", "airspace consolidation"],
        "observation": "confluent homogeneous parenchymal airspace opacification obscuring underlying pulmonary vessels",
        "neg_visual": "clear, well-aerated lung parenchyma without confluent airspace opacification",
        "clinical_name": "focal consolidation",
        "urgency": 3,
        "default_zone": "pulmonary parenchymal zone",
    },
    "Pneumonia": {
        "aliases": ["pneumonia"],
        "observation": "patchy-to-confluent alveolar airspace opacity with ill-defined borders",
        "neg_visual": "no focal alveolar airspace opacity to suggest infectious consolidation",
        "clinical_name": "pneumonia / infectious airspace process",
        "urgency": 3,
        "default_zone": "pulmonary parenchyma",
    },
    "Edema": {
        "aliases": ["edema", "pulmonary edema", "vascular congestion"],
        "observation": "perihilar haziness, indistinct pulmonary vascular margins, and cephalization of upper-lobe vessels",
        "neg_visual": "crisp pulmonary vascular margins without perihilar haziness or Kerley lines",
        "clinical_name": "pulmonary edema / vascular congestion",
        "urgency": 3,
        "default_zone": "bilateral perihilar and basilar lung fields",
    },
    "Atelectasis": {
        "aliases": ["atelectasis", "linear atelectasis", "subsegmental atelectasis"],
        "observation": "linear or wedge-shaped parenchymal density accompanied by subtle regional volume loss",
        "neg_visual": "preserved lung volumes without linear or wedge-shaped parenchymal collapse",
        "clinical_name": "atelectasis",
        "urgency": 5,
        "default_zone": "mid-to-lower lung zone",
    },
    "Cardiomegaly": {
        "aliases": ["cardiomegaly"],
        "observation": "transverse enlargement of the cardiac silhouette with cardiothoracic ratio exceeding 0.50",
        "neg_visual": "a normal cardiac silhouette with cardiothoracic ratio within normal limits",
        "clinical_name": "cardiomegaly",
        "urgency": 4,
        "default_zone": "central mediastinum and cardiac silhouette",
    },
    "Enlarged Cardiomediastinum": {
        "aliases": ["enlarged cardiomediastinum", "mediastinal widening", "widened mediastinum"],
        "observation": "widened superior and central cardiomediastinal contour",
        "neg_visual": "a normal superior and central cardiomediastinal contour without widening",
        "clinical_name": "enlarged cardiomediastinal silhouette",
        "urgency": 4,
        "default_zone": "central mediastinum",
    },
    "Aortic Enlargement": {
        "aliases": ["aortic enlargement", "aortic tortuosity", "aortic unfolding", "aortic elongation", "prominent aorta"],
        "observation": "prominence and tortuous unfolding of the thoracic aortic knob contour",
        "neg_visual": "an unremarkable thoracic aortic knob contour without abnormal unfolding",
        "clinical_name": "aortic enlargement / tortuosity",
        "urgency": 6,
        "default_zone": "superior mediastinum / aortic arch",
    },
    "Mediastinal Shift": {
        "aliases": ["mediastinal shift", "tracheal deviation"],
        "observation": "lateral displacement of the trachea and central mediastinal structures",
        "neg_visual": "a midline trachea and normally positioned central mediastinal structures",
        "clinical_name": "mediastinal shift",
        "urgency": 1,
        "default_zone": "central mediastinum",
    },
    "Lung Opacity": {
        "aliases": ["lung opacity", "airspace opacity", "pulmonary opacity"],
        "observation": "focal or patchy parenchymal opacity partially obscuring vascular markings",
        "neg_visual": "uniformly aerated lung fields without focal parenchymal opacity",
        "clinical_name": "pulmonary airspace opacity",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "Infiltration": {
        "aliases": ["infiltration", "infiltrate"],
        "observation": "ill-defined patchy interstitial and alveolar parenchymal opacification",
        "neg_visual": "clear lung fields without patchy alveolar or interstitial infiltrate",
        "clinical_name": "pulmonary infiltrate",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "ILD": {
        "aliases": ["ild", "interstitial lung disease", "reticular opacity", "interstitial pattern"],
        "observation": "diffuse reticular and reticulonodular interstitial markings throughout the parenchyma",
        "neg_visual": "normal bronchovascular markings without diffuse reticular interstitial thickening",
        "clinical_name": "interstitial lung abnormality",
        "urgency": 5,
        "default_zone": "bilateral lung fields",
    },
    "Mass": {
        "aliases": ["mass", "pulmonary mass", "lung mass"],
        "observation": "discrete rounded or lobulated focal opacity measuring greater than 3 cm",
        "neg_visual": "no discrete rounded or lobulated pulmonary mass lesion",
        "clinical_name": "pulmonary mass",
        "urgency": 2,
        "default_zone": "pulmonary parenchyma",
    },
    "Nodule": {
        "aliases": ["nodule", "nodule/mass", "diffuse nodule", "pulmonary nodule"],
        "observation": "circumscribed focal nodular radio-opacity with distinct parenchymal borders",
        "neg_visual": "no discrete focal pulmonary nodule identified",
        "clinical_name": "pulmonary nodule",
        "urgency": 3,
        "default_zone": "pulmonary parenchyma",
    },
    "Lung Cavity": {
        "aliases": ["lung cavity", "cavitation", "cavitary lesion"],
        "observation": "thick-walled parenchymal opacity containing a central radiolucent air-filled cavity",
        "neg_visual": "no thick-walled cavitary parenchymal lesion",
        "clinical_name": "cavitary pulmonary lesion",
        "urgency": 2,
        "default_zone": "upper-to-mid lung zone",
    },
    "Lung Cyst": {
        "aliases": ["lung cyst", "bulla", "bleb"],
        "observation": "thin-walled circumscribed radiolucent air space within the parenchyma",
        "neg_visual": "no circumscribed thin-walled pulmonary cyst or bulla",
        "clinical_name": "pulmonary cyst / bulla",
        "urgency": 6,
        "default_zone": "lung parenchyma",
    },
    "Lung Lesion": {
        "aliases": ["lung lesion", "other lesion", "lesion"],
        "observation": "focal parenchymal lesion distorting local pulmonary architecture",
        "neg_visual": "no discrete focal parenchymal lesion",
        "clinical_name": "focal pulmonary lesion",
        "urgency": 4,
        "default_zone": "lung parenchyma",
    },
    "Emphysema": {
        "aliases": ["emphysema", "hyperinflation", "copd"],
        "observation": "hyperinflated bilateral lung fields with increased radiolucency and flattened hemidiaphragms",
        "neg_visual": "normal lung volumes and smoothly domed hemidiaphragms without hyperinflation",
        "clinical_name": "pulmonary emphysema / hyperinflation",
        "urgency": 5,
        "default_zone": "bilateral lung fields",
    },
    "Fibrosis": {
        "aliases": ["fibrosis", "pulmonary fibrosis"],
        "observation": "coarse linear reticular bands with parenchymal architectural distortion and traction",
        "neg_visual": "no coarse fibrotic reticular scarring or architectural distortion",
        "clinical_name": "pulmonary fibrosis",
        "urgency": 5,
        "default_zone": "lung parenchyma",
    },
    "Pleural Thickening": {
        "aliases": ["pleural thickening", "pleural_thickening", "apical cap", "apical pleural thickening"],
        "observation": "focal linear thickening along the peripheral or apical pleural margin",
        "neg_visual": "smooth peripheral and apical pleural margins without focal thickening",
        "clinical_name": "pleural thickening",
        "urgency": 6,
        "default_zone": "peripheral pleural margin",
    },
    "Calcification": {
        "aliases": ["calcification", "calcified granuloma", "aortic calcification", "granuloma"],
        "observation": "well-defined high-attenuation punctate or curvilinear calcific density",
        "neg_visual": "no suspicious focal parenchymal calcification",
        "clinical_name": "thoracic calcification / calcified granuloma",
        "urgency": 7,
        "default_zone": "lung parenchyma or aortic arch",
    },
    "Fracture": {
        "aliases": ["fracture", "rib fracture", "clavicle fracture"],
        "observation": "focal cortical step-off and lucency disrupting the osseous contour",
        "neg_visual": "intact cortical margins across the visualized ribs and clavicles without acute fracture",
        "clinical_name": "rib or clavicular fracture",
        "urgency": 3,
        "default_zone": "osseous thoracic cage",
    },
    "Hernia": {
        "aliases": ["hernia", "hiatal hernia"],
        "observation": "retrocardiac soft-tissue/gas-containing density projecting above the diaphragm",
        "neg_visual": "an unremarkable retrocardiac region without hiatal herniation",
        "clinical_name": "hiatal hernia",
        "urgency": 6,
        "default_zone": "retrocardiac lower mediastinum",
    },
    "Support Devices": {
        "aliases": [
            "support devices", "pacemaker", "endotracheal tube", "central line",
            "chest tube", "catheter", "sternotomy", "prosthesis",
        ],
        "observation": "radiopaque medical hardware / support device projecting over the thorax",
        "neg_visual": "no radiopaque intrathoracic tubes, lines, or implanted hardware",
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

# Longest aliases first + strict word boundaries so "ild" never matches inside "mild"
_SORTED_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)"), canon)
    for alias, canon in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda kv: len(kv[0]), reverse=True)
]

_NEGATED_RAW_PREFIXES = ("no ", "without ", "negative for ", "normal ", "unchanged normal")

# High-value distractors used for Mode 3 false-positive CAD alerts
_DISTRACTOR_POOL: list[str] = [
    "Pneumothorax",
    "Effusion",
    "Consolidation",
    "Edema",
    "Cardiomegaly",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Fracture",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
]

_DISTRACTOR_ZONES: dict[str, list[str]] = {
    "Pneumothorax": ["right apical pleural space", "left apical pleural space", "right apicolateral pleural space"],
    "Effusion": ["right costophrenic angle and lower pleural space", "left costophrenic angle and lower pleural space"],
    "Consolidation": ["right lower lung zone / lung base", "left lower lung zone / lung base", "right mid lung zone"],
    "Edema": ["bilateral perihilar and basilar lung fields"],
    "Cardiomegaly": ["central mediastinum and cardiac silhouette"],
    "Mass": ["right upper lung zone (apical)", "left mid lung zone"],
    "Nodule": ["right mid lung zone", "left upper lung zone (apical)"],
    "Atelectasis": ["left lower lung zone / lung base", "right lower lung zone / lung base"],
    "Fracture": ["right osseous thoracic cage", "left osseous thoracic cage"],
    "Emphysema": ["bilateral lung fields"],
    "Fibrosis": ["bilateral lung fields"],
    "Pleural Thickening": ["right apical pleural margin", "left apical pleural margin"],
}


def canonicalize_label(raw_label: str) -> str | None:
    """Maps raw dataset label names into the unified 25-pathology ontology safely."""
    cleaned = raw_label.strip().lower().replace("_", " ")
    if not cleaned or cleaned in ("no finding", "normal", "none", "unchanged", "suboptimal study"):
        return None
    if cleaned.startswith(_NEGATED_RAW_PREFIXES):
        return None
    if cleaned in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[cleaned]
    for pattern, canon in _SORTED_ALIAS_PATTERNS:
        if pattern.search(cleaned):
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


def _flip_laterality(zone: str) -> str | None:
    """Flips right <-> left in an anatomical zone string for Mode 3c laterality hard negatives."""
    has_right = bool(re.search(r"\bright\b", zone, flags=re.IGNORECASE))
    has_left = bool(re.search(r"\bleft\b", zone, flags=re.IGNORECASE))
    if has_right == has_left:
        return None
    tmp = re.sub(r"\bright\b", "__SIDE_SWAP__", zone, flags=re.IGNORECASE)
    tmp = re.sub(r"\bleft\b", "right", tmp, flags=re.IGNORECASE)
    return tmp.replace("__SIDE_SWAP__", "left")


# =====================================================================
# 3. STRICT 5-RULE REPORT SYNTHESIZER & VALIDATOR
# =====================================================================
def build_rulebook_target(
    findings: list[dict[str, Any]],
    raw_report_note: str | None = None,
    rejected_cad_notes: list[str] | None = None,
) -> str:
    """
    Synthesizes a strictly rule-compliant 3-section radiological report from
    grounded dataset annotations (`findings`: [{label, zone}]).
    Optional `rejected_cad_notes` adds visual refutation sentences in Section 1
    for Mode 3 (CAD-Authenticator Rejection) without contaminating Sections 2-3.
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
    if explicit_negatives:
        neg_sentence = f"No radiographic evidence of {', '.join(explicit_negatives)}."
    else:
        neg_sentence = "No radiographic evidence of free subdiaphragmatic air."

    refutation_block = " ".join(note.strip() for note in (rejected_cad_notes or []) if note.strip())

    if not ordered:
        sec1_parts = [
            "The trachea and central airways are midline and patent.",
            "Bilateral lung fields are well-aerated and clear without focal airspace opacity or pleural abnormality.",
            "The cardiomediastinal silhouette is normal in size and contour, and the thoracic skeleton is intact.",
        ]
        if refutation_block:
            sec1_parts.append(refutation_block)
        sec1_parts.append(neg_sentence)
        sec1 = " ".join(sec1_parts)
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
        if refutation_block:
            obs_sentences.append(refutation_block)
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
# 4. DISK-SAFE STREAMING DATASET LOADERS
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


def _save_resized_image(pil_img: Any, out_path: Path, image_size: int = 512) -> str:
    """Saves a compact 512x512 RGB JPEG (~40 KB), safely normalizing 12/16-bit medical images."""
    import numpy as np
    from PIL import Image

    if not out_path.exists():
        arr = np.asarray(pil_img)
        if arr.dtype != np.uint8 or str(getattr(pil_img, "mode", "")).startswith(("I", "F")):
            arr_f = arr.astype(np.float32)
            if arr_f.ndim == 3:
                arr_f = arr_f[..., 0]
            p1, p99 = np.percentile(arr_f, [1.0, 99.0])
            if p99 > p1:
                arr_f = np.clip((arr_f - p1) / (p99 - p1), 0.0, 1.0) * 255.0
            elif float(arr_f.max()) > float(arr_f.min()):
                arr_f = (arr_f - float(arr_f.min())) / (float(arr_f.max()) - float(arr_f.min())) * 255.0
            else:
                arr_f = np.zeros_like(arr_f)
            rgb = Image.fromarray(arr_f.astype(np.uint8)).convert("RGB")
        else:
            rgb = pil_img.convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), resample)
        rgb.save(out_path, format="JPEG", quality=95)
    return str(out_path)


def _parse_padchest_locations(locs: Any) -> str:
    """Safely parses PadChest-GR locations whether stored as a list or stringified list."""
    import ast

    if isinstance(locs, str):
        s_loc = locs.strip()
        if s_loc.startswith("["):
            try:
                locs = ast.literal_eval(s_loc)
            except Exception:
                locs = []
        else:
            locs = [s_loc] if s_loc else []
    elif not isinstance(locs, list):
        locs = []

    flat: list[str] = []
    for x in locs:
        if isinstance(x, (list, tuple)):
            flat.extend(str(y).strip() for y in x if str(y).strip())
        elif str(x).strip():
            flat.append(str(x).strip())
    return ", ".join(dict.fromkeys(flat))


def load_chestx_det_hf(
    max_samples: int | None = None,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    split: str = "train",
    streaming: bool = True,
) -> list[dict[str, Any]]:
    import numpy as np
    from datasets import load_dataset

    print(f"[Loader] Streaming ChestX-Det ({split})...", flush=True)
    ds = load_dataset("natealberti/ChestX-Det", split=split, streaming=streaming)
    records: list[dict[str, Any]] = []
    out_img_dir = Path(img_cache_dir) / f"chestx_det_{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        pil_img = row.get("image")
        lbl_img = row.get("label")
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size
        img_file = _save_resized_image(pil_img, out_img_dir / f"chestx_det_{split}_{idx:06d}.jpg", image_size)

        findings: list[dict[str, Any]] = []
        if lbl_img is not None:
            lbl_arr = np.asarray(lbl_img)
            if lbl_arr.ndim == 3:
                lbl_arr = lbl_arr[..., 0]
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
            "source": f"ChestX-Det-{split}",
            "group_id": f"chestx_det_{split}_{idx:06d}",
            "image_path": img_file,
            "findings": findings,
        })
        if len(records) % 250 == 0:
            print(f"  [ChestX-Det-{split}] {len(records)} cached...", flush=True)
    print(f"[Loader] ChestX-Det ({split}): loaded {len(records)} studies", flush=True)
    return records


def load_chexpert_hf(
    max_samples: int | None = None,
    frontal_only: bool = True,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Streams danjacobellis/chexpert (Frontal == 0, positive == 3; skips ambiguous uncertain-only rows)."""
    from datasets import load_dataset

    print("[Loader] Streaming CheXpert (train, frontal only)...", flush=True)
    ds = load_dataset("danjacobellis/chexpert", split="train", streaming=streaming)
    chex_cols = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    ]
    out_img_dir = Path(img_cache_dir) / "chexpert"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        fl_val = str(row.get("Frontal/Lateral", "")).strip().lower()
        if frontal_only and fl_val not in ("0", "frontal"):
            continue
        pil_img = row.get("image")
        if pil_img is None:
            continue

        findings = []
        has_uncertain = False
        for col in chex_cols:
            val = row.get(col)
            if val == 3 or str(val).strip() == "3":
                canon = canonicalize_label(col)
                if canon:
                    findings.append({
                        "label": canon,
                        "zone": PATHOLOGY_ONTOLOGY[canon]["default_zone"],
                    })
            elif val == 2 or str(val).strip() == "2":
                has_uncertain = True

        no_finding = row.get("No Finding") == 3 or str(row.get("No Finding")).strip() == "3"
        if not findings and (not no_finding or has_uncertain):
            continue

        img_file = _save_resized_image(pil_img, out_img_dir / f"chexpert_{idx:06d}.jpg", image_size)
        raw_path = str(row.get("Path") or row.get("path") or "")
        pat_match = re.search(r"patient\d+", raw_path, flags=re.IGNORECASE)
        group_id = f"chexpert_{pat_match.group(0).lower()}" if pat_match else f"chexpert_{idx:06d}"

        records.append({
            "source": "CheXpert",
            "group_id": group_id,
            "image_path": img_file,
            "findings": findings,
        })
        if len(records) % 500 == 0:
            print(f"  [CheXpert] {len(records)} cached...", flush=True)
    print(f"[Loader] CheXpert (train): loaded {len(records)} frontal studies", flush=True)
    return records


def load_padchest_gr_hf(
    max_samples: int | None = None,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    split: str = "train",
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Streams JasonZZ0601/PadChest-GR-Xray with 16-bit contrast normalization & safe location parsing."""
    from datasets import load_dataset

    print(f"[Loader] Streaming PadChest-GR ({split}, 16-bit normalized)...", flush=True)
    ds = load_dataset("JasonZZ0601/PadChest-GR-Xray", split=split, streaming=streaming)
    out_img_dir = Path(img_cache_dir) / f"padchest_gr_{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        pil_img = row.get("image")
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size
        img_file = _save_resized_image(pil_img, out_img_dir / f"padchest_{split}_{idx:06d}.jpg", image_size)

        raw_lbl = row.get("label") if row.get("label") is not None else row.get("labels", "")
        parts = [str(x) for x in raw_lbl] if isinstance(raw_lbl, list) else [p for p in str(raw_lbl).split(",")]
        loc_str = _parse_padchest_locations(row.get("locations"))

        boxes = row.get("boxes") or row.get("bbox") or None
        first_box = None
        if isinstance(boxes, list) and boxes:
            if isinstance(boxes[0], (list, tuple)) and len(boxes[0]) == 4:
                first_box = boxes[0]
            elif len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
                first_box = boxes

        findings = []
        for part in parts:
            canon = canonicalize_label(part)
            if canon:
                if first_box is not None:
                    zone = bbox_to_anatomical_zone(first_box, canon, img_w=img_w, img_h=img_h)
                else:
                    zone = loc_str or PATHOLOGY_ONTOLOGY[canon]["default_zone"]
                findings.append({"label": canon, "zone": zone})

        note = str(row.get("sentence_en") or "").strip()
        group_raw = row.get("PatientID") or row.get("patient_id") or row.get("ImageID") or f"{idx:06d}"

        records.append({
            "source": f"PadChest-GR-{split}",
            "group_id": f"padchest_{split}_{group_raw}",
            "image_path": img_file,
            "findings": findings,
            "note": None if (not note or note.lower() == "none") else note,
        })
        if len(records) % 100 == 0:
            print(f"  [PadChest-GR-{split}] {len(records)} cached...", flush=True)
    print(f"[Loader] PadChest-GR ({split}): loaded {len(records)} studies", flush=True)
    return records


def load_siim_pneumo_hf(
    max_samples: int | None = None,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Streams nwirandx/siim-pneumothorax-256-2400 for high-recall Pneumothorax bbox grounding."""
    import numpy as np
    from datasets import load_dataset

    print("[Loader] Streaming SIIM-Pneumothorax (train)...", flush=True)
    ds = load_dataset("nwirandx/siim-pneumothorax-256-2400", split="train", streaming=streaming)
    out_img_dir = Path(img_cache_dir) / "siim_pneumo"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        pil_img = row.get("image")
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size
        img_file = _save_resized_image(pil_img, out_img_dir / f"siim_pneumo_{idx:06d}.jpg", image_size)

        has_pneumo = int(row.get("has_pneumo", 0) or 0) == 1
        findings: list[dict[str, Any]] = []
        if has_pneumo:
            mask_img = row.get("mask")
            box = None
            if mask_img is not None:
                m_arr = np.asarray(mask_img)
                if m_arr.ndim == 3:
                    m_arr = m_arr[..., 0]
                ys, xs = np.where(m_arr > 0)
                if len(xs) > 0:
                    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            findings.append({
                "label": "Pneumothorax",
                "zone": bbox_to_anatomical_zone(box, "Pneumothorax", img_w=img_w, img_h=img_h),
            })

        records.append({
            "source": "SIIM-Pneumothorax",
            "group_id": f"siim_pneumo_{idx:06d}",
            "image_path": img_file,
            "findings": findings,
        })
        if len(records) % 500 == 0:
            print(f"  [SIIM-Pneumothorax] {len(records)} cached...", flush=True)
    print(f"[Loader] SIIM-Pneumothorax: loaded {len(records)} studies", flush=True)
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
            "group_id": f"vindr_{img_id}",
            "image_path": str(img_path),
            "findings": findings,
        })
    return records


def load_generic_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Loads custom JSONL annotations: {image_path, findings: [{label, bbox?, zone?}]}."""
    records: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
                "group_id": item.get("group_id") or f"custom_{idx:06d}",
                "image_path": item["image_path"],
                "findings": norm_findings,
                "note": item.get("note"),
            })
    return records


# =====================================================================
# 5. LEAK-FREE SPLITTING, REBALANCING & 3-MODE AUTHENTICATOR FORMATTING
# =====================================================================
def split_records_by_group(
    records: list[dict[str, Any]],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits studies by patient/image group_id BEFORE oversampling so zero images leak into val."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        gid = rec.get("group_id") or rec.get("image_path") or f"rec_{idx}"
        grouped[gid].append(rec)

    group_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    if len(group_ids) <= 1:
        return list(records), list(records)

    val_n_groups = max(1, int(round(len(group_ids) * val_ratio)))
    val_gids = set(group_ids[:val_n_groups])

    train_recs: list[dict[str, Any]] = []
    val_recs: list[dict[str, Any]] = []
    for gid, recs in grouped.items():
        if gid in val_gids:
            val_recs.extend(recs)
        else:
            train_recs.extend(recs)
    return train_recs, val_recs


def balance_records_against_normal_bias(
    records: list[dict[str, Any]],
    max_normal_ratio: float = 0.20,
    min_class_count: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Rebalances training dataset to eradicate "Normal-biased" hallucination:
    - Caps Normal-only studies to at most `max_normal_ratio` of final training set.
    - Oversamples underrepresented etiologies up to `min_class_count` occurrences.
    """
    rng = random.Random(seed)
    normal_recs = [r for r in records if not r["findings"]]
    abnormal_recs = [r for r in records if r["findings"]]

    if not abnormal_recs:
        return records

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in abnormal_recs:
        seen_in_rec = set()
        for f_item in rec["findings"]:
            lbl = f_item["label"]
            if lbl not in seen_in_rec:
                by_class[lbl].append(rec)
                seen_in_rec.add(lbl)

    balanced = list(abnormal_recs)
    for canon, class_recs in by_class.items():
        if 0 < len(class_recs) < min_class_count:
            needed = min_class_count - len(class_recs)
            balanced.extend(rng.choices(class_recs, k=needed))

    max_normals = max(1, int(len(balanced) * (max_normal_ratio / max(1e-5, 1.0 - max_normal_ratio))))
    if len(normal_recs) > max_normals:
        normal_recs = rng.sample(normal_recs, max_normals)

    combined = balanced + normal_recs
    rng.shuffle(combined)
    return combined


def _build_cad_prompt(cad_summary: str) -> str:
    """Shared CAD prompt template used identically in Mode 2 (confirm) and Mode 3 (reject)."""
    return (
        f"Candidate Ensemble CAD findings: {cad_summary}.\n"
        "Authenticate these CAD alerts against the radiograph and analyze this chest X-ray "
        "step-by-step according to the 5-Rule Clinical Rulebook."
    )


def _build_mode3_rejection_payload(
    record: dict[str, Any],
    rng: random.Random,
) -> tuple[str, list[str], str]:
    """
    Constructs a perturbed CAD alert + visual counter-evidence refutation notes.
    Returns: (cad_summary, rejected_notes, rejection_subtype)
      - normal_fp: Normal study + 1-2 fake CAD flags -> reject & conclude Normal.
      - laterality_flip: Unilateral study + opposite-side CAD flag -> reject wrong side & correct.
      - abnormal_distractor: Abnormal study + true flag(s) + 1 fake CAD flag -> confirm true, reject fake.
    """
    true_findings = record["findings"]
    present_labels = {f["label"] for f in true_findings}
    absent_pool = [c for c in _DISTRACTOR_POOL if c not in present_labels] or ["Pneumothorax"]

    # Subtype 3a: Normal False-Positive Rejection
    if not true_findings:
        k = 2 if (len(absent_pool) >= 2 and rng.random() < 0.25) else 1
        fake_classes = rng.sample(absent_pool, k=k)
        cad_items: list[str] = []
        ref_notes: list[str] = []
        for fc in fake_classes:
            meta = PATHOLOGY_ONTOLOGY[fc]
            fake_zone = rng.choice(_DISTRACTOR_ZONES.get(fc, [meta["default_zone"]]))
            cad_items.append(f"{meta['clinical_name']} at {fake_zone}")
            ref_notes.append(
                f"CAD flagged suspected {meta['clinical_name']} at the {fake_zone}; "
                f"however, direct visual evaluation of the radiograph does not substantiate this alert, "
                f"demonstrating {meta['neg_visual']}."
            )
        return "; ".join(cad_items), ref_notes, "normal_fp"

    # Subtype 3c: Laterality Flip Rejection (when a unilateral finding is present)
    flippable = [(i, f, _flip_laterality(f["zone"])) for i, f in enumerate(true_findings)]
    flippable = [(i, f, flipped) for i, f, flipped in flippable if flipped is not None]
    if flippable and rng.random() < 0.35:
        flip_idx, target_f, wrong_zone = rng.choice(flippable)
        meta = PATHOLOGY_ONTOLOGY[target_f["label"]]
        cad_items = []
        for i, f in enumerate(true_findings[:4]):
            f_meta = PATHOLOGY_ONTOLOGY[f["label"]]
            z = wrong_zone if i == flip_idx else f["zone"]
            cad_items.append(f"{f_meta['clinical_name']} at {z}")
        ref_note = (
            f"CAD flagged suspected {meta['clinical_name']} at the {wrong_zone}; "
            f"however, direct visual inspection rejects this laterality and confirms the abnormality "
            f"is actually located within the {target_f['zone']}."
        )
        return "; ".join(cad_items), [ref_note], "laterality_flip"

    # Subtype 3b: Abnormal Distractor Rejection (true finding + injected fake finding)
    fake_cls = rng.choice(absent_pool)
    fake_meta = PATHOLOGY_ONTOLOGY[fake_cls]
    fake_zone = rng.choice(_DISTRACTOR_ZONES.get(fake_cls, [fake_meta["default_zone"]]))

    cad_items = [
        f"{PATHOLOGY_ONTOLOGY[f['label']]['clinical_name']} at {f['zone']}"
        for f in true_findings[:3]
    ]
    cad_items.append(f"{fake_meta['clinical_name']} at {fake_zone}")
    rng.shuffle(cad_items)

    ref_note = (
        f"CAD additionally flagged suspected {fake_meta['clinical_name']} at the {fake_zone}; "
        f"however, direct visual evaluation rejects this false alert, showing {fake_meta['neg_visual']}."
    )
    return "; ".join(cad_items), [ref_note], "abnormal_distractor"


def format_vlm_chat_sample(
    record: dict[str, Any],
    include_cad_context: bool = False,
    mode: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    Formats a record into SmolVLM chat messages + validates 5-Rule compliance.
    Backward-compatible with `include_cad_context=bool`; pass `mode="pure" | "cad_aligned" | "cad_reject"`
    for full 3-Mode Rulebook + Authenticator synthesis.
    """
    effective_mode = mode or ("cad_aligned" if include_cad_context else "pure")
    local_rng = rng or random.Random(42)

    rejected_notes: list[str] | None = None
    rejection_subtype: str | None = None

    if effective_mode == "pure":
        user_text = "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
    elif effective_mode == "cad_aligned":
        if not record["findings"]:
            cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold"
        else:
            cad_summary = "; ".join(
                f"{PATHOLOGY_ONTOLOGY[f['label']]['clinical_name']} at {f['zone']}"
                for f in record["findings"][:4]
            )
        user_text = _build_cad_prompt(cad_summary)
    elif effective_mode == "cad_reject":
        cad_summary, rejected_notes, rejection_subtype = _build_mode3_rejection_payload(record, local_rng)
        user_text = _build_cad_prompt(cad_summary)
    else:
        raise ValueError(f"Unsupported mode: {effective_mode}")

    target_report = build_rulebook_target(
        record["findings"],
        raw_report_note=record.get("note"),
        rejected_cad_notes=rejected_notes,
    )
    ok, errs = validate_rulebook_compliance(target_report)
    if not ok:
        raise ValueError(f"Generated target failed rulebook validation: {errs}")

    return {
        "source": record.get("source", "unknown"),
        "group_id": record.get("group_id", ""),
        "image_path": record["image_path"],
        "labels": [f["label"] for f in record["findings"]],
        "mode": effective_mode,
        "rejection_subtype": rejection_subtype,
        "cad_conditioned": effective_mode != "pure",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_report}]},
        ],
    }


def _emit_three_mode_samples(
    records: list[dict[str, Any]],
    seed: int = 42,
    pure_ratio: float = 0.40,
    cad_aligned_ratio: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Assigns 40% Pure-Image, 35% CAD-Aligned, and 25% CAD-Authenticator Rejection
    while ensuring every study is represented cleanly.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for rec in records:
        r = rng.random()
        if r < pure_ratio:
            mode = "pure"
        elif r < pure_ratio + cad_aligned_ratio:
            mode = "cad_aligned"
        else:
            mode = "cad_reject"
        samples.append(format_vlm_chat_sample(rec, mode=mode, rng=rng))
    rng.shuffle(samples)
    return samples


def export_holdout_benchmarks(holdout_dir: str, img_cache_dir: str, image_size: int = 512, max_padchest: int = 120) -> None:
    """Exports completely unseen ChestX-Det (test) and PadChest-GR (test) manifest for Gate 1-4 validation."""
    out = Path(holdout_dir)
    out.mkdir(parents=True, exist_ok=True)
    holdout_records: list[dict[str, Any]] = []
    holdout_records.extend(
        load_chestx_det_hf(split="test", img_cache_dir=img_cache_dir, image_size=image_size, streaming=True)
    )
    try:
        holdout_records.extend(
            load_padchest_gr_hf(
                max_samples=max_padchest,
                split="test",
                img_cache_dir=img_cache_dir,
                image_size=image_size,
                streaming=True,
            )
        )
    except Exception as exc:
        print(f"[Holdout] Skipping PadChest-GR test split ({exc})")

    manifest_path = out / "holdout_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(holdout_records, mf, indent=2)
    print(f"[Holdout] Exported {len(holdout_records)} unseen test studies -> {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced 3-Mode 5-Rule CoT training JSONL for SmolVLM.")
    parser.add_argument("--chestx-det", action="store_true", help="Include HuggingFace natealberti/ChestX-Det.")
    parser.add_argument("--chexpert-hf", action="store_true", help="Include HuggingFace danjacobellis/chexpert.")
    parser.add_argument("--padchest-gr", action="store_true", help="Include HuggingFace JasonZZ0601/PadChest-GR-Xray.")
    parser.add_argument("--siim-pneumo", action="store_true", help="Include HuggingFace nwirandx/siim-pneumothorax-256-2400.")
    parser.add_argument("--max-per-source", type=int, default=None, help="Cap samples per dataset source.")
    parser.add_argument("--max-padchest", type=int, default=None, help="Optional separate cap for heavy 16-bit PadChest-GR.")
    parser.add_argument("--vindr-csv", type=str, default=None, help="Path to VinDr-CXR train.csv.")
    parser.add_argument("--vindr-images", type=str, default=None, help="Path to VinDr-CXR images folder.")
    parser.add_argument("--custom-jsonl", type=str, default=None, help="Path to custom JSONL manifest.")
    parser.add_argument("--max-normal-ratio", type=float, default=0.20, help="Max fraction of Normal studies.")
    parser.add_argument("--min-class-count", type=int, default=300, help="Oversample target per etiology.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split fraction (leak-free).")
    parser.add_argument("--image-size", type=int, default=512, help="Cached RGB JPEG size.")
    parser.add_argument("--img-cache-dir", type=str, default="./.tmp/training_images", help="Image cache folder (use /kaggle/tmp on Kaggle).")
    parser.add_argument("--out-dir", type=str, default="./.tmp/rulebook_dataset", help="Output JSONL directory.")
    parser.add_argument("--export-holdout-dir", type=str, default=None, help="Optionally export unseen ChestX-Det/PadChest-GR test holdout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-sample", action="store_true", help="Generate a tiny demo dataset from local image.")
    args = parser.parse_args()

    if args.export_holdout_dir:
        export_holdout_benchmarks(
            holdout_dir=args.export_holdout_dir,
            img_cache_dir=args.img_cache_dir,
            image_size=args.image_size,
        )

    records: list[dict[str, Any]] = []
    if args.smoke_sample:
        sample_img = "./1002_IM-0004-1001.dcm.png"
        if not os.path.exists(sample_img):
            raise FileNotFoundError(f"Smoke test image missing: {sample_img}")
        records = [
            {"source": "smoke_normal", "group_id": "smoke_1", "image_path": sample_img, "findings": []},
            {"source": "smoke_aortic", "group_id": "smoke_2", "image_path": sample_img, "findings": [
                {"label": "Aortic Enlargement", "zone": "superior mediastinum / left aortic arch"}
            ]},
            {"source": "smoke_ptx", "group_id": "smoke_3", "image_path": sample_img, "findings": [
                {"label": "Pneumothorax", "zone": "right apical pleural space"}
            ]},
        ]
    else:
        if args.chestx_det:
            records.extend(load_chestx_det_hf(
                max_samples=args.max_per_source,
                img_cache_dir=args.img_cache_dir,
                image_size=args.image_size,
            ))
        if args.chexpert_hf:
            records.extend(load_chexpert_hf(
                max_samples=args.max_per_source,
                img_cache_dir=args.img_cache_dir,
                image_size=args.image_size,
            ))
        if args.padchest_gr:
            records.extend(load_padchest_gr_hf(
                max_samples=args.max_padchest or args.max_per_source,
                img_cache_dir=args.img_cache_dir,
                image_size=args.image_size,
            ))
        if args.siim_pneumo:
            records.extend(load_siim_pneumo_hf(
                max_samples=args.max_per_source,
                img_cache_dir=args.img_cache_dir,
                image_size=args.image_size,
            ))
        if args.vindr_csv and args.vindr_images:
            records.extend(load_vindr_csv(args.vindr_csv, args.vindr_images, max_samples=args.max_per_source))
        if args.custom_jsonl:
            records.extend(load_generic_jsonl(args.custom_jsonl))

    if not records:
        print("No dataset inputs provided. Pass --chestx-det, --padchest-gr, --chexpert-hf, --siim-pneumo, or --smoke-sample.")
        return

    # 1. Leak-free group split BEFORE oversampling
    if args.smoke_sample:
        train_recs = records
        val_recs = records[:2]
    else:
        train_recs, val_recs = split_records_by_group(records, val_ratio=args.val_ratio, seed=args.seed)
        train_recs = balance_records_against_normal_bias(
            train_recs,
            max_normal_ratio=args.max_normal_ratio,
            min_class_count=args.min_class_count,
            seed=args.seed,
        )

    # 2. Emit 3-Mode training mix (40% Pure / 35% CAD-Aligned / 25% CAD-Rejection)
    if args.smoke_sample:
        rng = random.Random(args.seed)
        train_samples = [
            format_vlm_chat_sample(train_recs[0], mode="pure", rng=rng),
            format_vlm_chat_sample(train_recs[1], mode="cad_aligned", rng=rng),
            format_vlm_chat_sample(train_recs[0], mode="cad_reject", rng=rng),
            format_vlm_chat_sample(train_recs[2], mode="cad_reject", rng=rng),
        ]
        val_samples = [
            format_vlm_chat_sample(val_recs[0], mode="pure", rng=rng),
            format_vlm_chat_sample(val_recs[1], mode="cad_reject", rng=rng),
        ]
    else:
        train_samples = _emit_three_mode_samples(train_recs, seed=args.seed, pure_ratio=0.40, cad_aligned_ratio=0.35)
        val_samples = _emit_three_mode_samples(val_recs, seed=args.seed + 1, pure_ratio=0.40, cad_aligned_ratio=0.35)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, subset in (("train.jsonl", train_samples), ("val.jsonl", val_samples)):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for sample in subset:
                f.write(json.dumps(sample) + "\n")

    class_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    rej_counts: Counter[str] = Counter()
    for s in train_samples:
        mode_counts[s["mode"]] += 1
        if s.get("rejection_subtype"):
            rej_counts[s["rejection_subtype"]] += 1
        if not s["labels"]:
            class_counts["Normal"] += 1
        else:
            for lbl in s["labels"]:
                class_counts[lbl] += 1

    total_train = max(1, len(train_samples))
    normal_pct = 100.0 * class_counts["Normal"] / total_train
    print(f"[Dataset Builder] Wrote {len(train_samples)} train and {len(val_samples)} unseen val samples -> {out_dir}")
    print(f"[Dataset Builder] Mode split: {dict(mode_counts)} | Rejection subtypes: {dict(rej_counts)}")
    print(f"[Dataset Builder] Normal-only share: {normal_pct:.1f}% | Top classes: {dict(class_counts.most_common(15))}")


if __name__ == "__main__":
    main()