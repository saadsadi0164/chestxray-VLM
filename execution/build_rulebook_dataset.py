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
        obs_sentences: list[str] = ["The trachea and central airways are midline and patent."]
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
# Verified 0-indexed pixel IDs for natealberti/ChestX-Det masks (10=Nodule missing & merged into 12 upstream;
# ID 12 is intentionally excluded from mask fallback so Nodules are NEVER mislabeled as Pneumothorax).
CHESTX_DET_ID2LABEL: dict[int, str] = {
    0: "Atelectasis",
    1: "Calcification",
    2: "Cardiomegaly",
    3: "Consolidation",
    4: "Diffuse Nodule",
    5: "Effusion",
    6: "Emphysema",
    7: "Fibrosis",
    8: "Fracture",
    9: "Mass",
    11: "Pleural Thickening",
}


def normalize_cxr_to_rgb(pil_img: Any, image_size: int = 512) -> Any:
    """
    Uniform 1st-99th percentile grayscale contrast normalization across ALL datasets
    (8-bit, 16-bit, RGBA, L) and local v2.2 inference so the VLM cannot exploit
    dataset-specific intensity histograms as shortcuts. Pillow 13 safe.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(pil_img)
    if arr.ndim == 3:
        arr_f = arr[..., :3].astype(np.float32).mean(axis=-1)
    else:
        arr_f = arr.astype(np.float32)

    p1, p99 = np.percentile(arr_f, [1.0, 99.0])
    if p99 > p1:
        arr_f = np.clip((arr_f - p1) / (p99 - p1), 0.0, 1.0) * 255.0
    elif float(arr_f.max()) > float(arr_f.min()):
        arr_f = (arr_f - float(arr_f.min())) / (float(arr_f.max()) - float(arr_f.min())) * 255.0
    else:
        arr_f = np.zeros_like(arr_f)

    rgb = Image.fromarray(arr_f.astype(np.uint8)).convert("RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), resample)
    return rgb


def _save_resized_image(pil_img: Any, out_path: Path, image_size: int = 512) -> str:
    """Saves a uniformly normalized 512x512 RGB JPEG (~40 KB)."""
    if not out_path.exists():
        rgb = normalize_cxr_to_rgb(pil_img, image_size=image_size)
        rgb.save(out_path, format="JPEG", quality=95)
    return str(out_path)


def _fast_image_fingerprint(pil_img: Any) -> str:
    """Computes a fast 10-char hash to group multi-sentence PadChest-GR rows belonging to one radiograph."""
    import hashlib
    import numpy as np

    arr = np.asarray(pil_img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    h, w = arr.shape[:2]
    thumb = arr[:: max(1, h // 32), :: max(1, w // 32)]
    return hashlib.md5(thumb.tobytes()).hexdigest()[:10]


def _extract_mask_boxes(binary_mask: Any, min_pixels: int = 16) -> list[list[float]]:
    """Extracts per-instance connected-component bounding boxes so multi-focal lesions stay spatially accurate."""
    import numpy as np

    labeled = None
    n_comp = 0
    try:
        from scipy.ndimage import label as cc_label
        labeled, n_comp = cc_label(binary_mask)
    except Exception:
        try:
            from skimage.measure import label as sk_label
            labeled = sk_label(binary_mask, connectivity=2)
            n_comp = int(labeled.max())
        except Exception:
            labeled = None

    boxes: list[list[float]] = []
    if labeled is not None and n_comp > 0:
        for comp_id in range(1, n_comp + 1):
            ys, xs = np.where(labeled == comp_id)
            if len(xs) < min_pixels:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])

    if not boxes:
        ys, xs = np.where(binary_mask)
        if len(xs) > 0:
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return boxes


def _merge_component_zones(zones: list[str], canon: str) -> list[str]:
    """Deduplicates per-component zones and cleanly collapses matched left+right pairs into bilateral."""
    uniq = list(dict.fromkeys(zones))
    if len(uniq) <= 1:
        return uniq

    has_right = any(z.startswith("right ") for z in uniq)
    has_left = any(z.startswith("left ") for z in uniq)
    if has_right and has_left:
        suffixes = set()
        for z in uniq:
            if z.startswith("right "):
                suffixes.add(z[len("right "):])
            elif z.startswith("left "):
                suffixes.add(z[len("left "):])
        if len(suffixes) == 1:
            return [f"bilateral {next(iter(suffixes))}"]
        default_z = PATHOLOGY_ONTOLOGY[canon]["default_zone"]
        return [default_z if "bilateral" in default_z else "bilateral lung fields"]
    return uniq[:2]


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


def _normalize_padchest_zone(loc_str: str, canon: str) -> str:
    """Maps PadChest-GR free-text location tokens into our exact canonical anatomical zone grammar."""
    default_z = PATHOLOGY_ONTOLOGY[canon]["default_zone"]
    if canon in ("Cardiomegaly", "Enlarged Cardiomediastinum", "Aortic Enlargement", "Hernia", "Support Devices"):
        return default_z

    s = (loc_str or "").lower()
    if not s:
        return default_z

    has_r = bool(re.search(r"\bright\b", s))
    has_l = bool(re.search(r"\bleft\b", s))
    if "bilateral" in s or (has_r and has_l):
        side = "bilateral"
    elif has_r:
        side = "right"
    elif has_l:
        side = "left"
    else:
        side = ""

    if canon == "Effusion":
        return f"{side} costophrenic angle and lower pleural space" if side else default_z
    if canon == "Pneumothorax":
        return f"{side} apical pleural space" if side else default_z
    if canon == "Pleural Thickening":
        return f"{side} pleural space" if side else default_z
    if canon == "Fracture":
        return f"{side} osseous thoracic cage" if side else default_z

    if any(k in s for k in ("apical", "apex", "upper", "superior")):
        level = "upper lung zone (apical)"
    elif any(k in s for k in ("basal", "base", "lower", "inferior", "costophrenic")):
        level = "lower lung zone / lung base"
    elif any(k in s for k in ("mid", "middle", "hilar", "perihilar")):
        level = "mid lung zone"
    else:
        level = ""

    if side and level:
        return f"{side} {level}"
    if side:
        return f"{side} lung fields" if side == "bilateral" else f"{side} lung field"
    return default_z


def _fetch_official_chestx_det_json(split: str) -> list[dict[str, Any]] | None:
    """Fetches official Deepwise ChestX-Det multi-label box annotations to recover uncorrupted Nodule/Pneumothorax/Consolidation."""
    import urllib.request

    fname = "ChestX_Det_train.json" if split == "train" else "ChestX_Det_test.json"
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/{branch}/{fname}"
        try:
            raw = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data
        except Exception:
            continue
    return None


def _findings_from_deepwise_item(item: dict[str, Any], img_w: int, img_h: int) -> list[dict[str, Any]]:
    """Converts one official Deepwise ChestX-Det JSON record (syms + boxes) into canonical grounded findings."""
    syms = item.get("syms") or []
    boxes = item.get("boxes") or []
    by_canon: dict[str, list[str]] = defaultdict(list)

    for i, sym in enumerate(syms):
        canon = canonicalize_label(str(sym))
        if not canon:
            continue
        box = boxes[i] if i < len(boxes) and isinstance(boxes[i], (list, tuple)) and len(boxes[i]) == 4 else None
        if box is not None:
            eff_w = max(float(img_w), float(box[2]), 1.0)
            eff_h = max(float(img_h), float(box[3]), 1.0)
            zone = bbox_to_anatomical_zone(box, canon, img_w=eff_w, img_h=eff_h)
        else:
            zone = PATHOLOGY_ONTOLOGY[canon]["default_zone"]
        by_canon[canon].append(zone)

    findings: list[dict[str, Any]] = []
    for canon, zones in by_canon.items():
        for z in _merge_component_zones(zones, canon):
            findings.append({"label": canon, "zone": z})
    return findings


def load_chestx_det_hf(
    max_samples: int | None = None,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    split: str = "train",
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """
    Streams natealberti/ChestX-Det with uniform normalization and mathematically verified
    1-to-1 official Deepwise JSON recovery (recovers uncorrupted Nodule, Pneumothorax,
    and occluded Consolidations; falls back safely to verified 0-indexed mask IDs 0..9, 11).
    """
    import numpy as np
    from datasets import load_dataset

    print(f"[Loader] Streaming ChestX-Det ({split}, uniform norm + verified 0-indexed labels)...", flush=True)
    ds = load_dataset("natealberti/ChestX-Det", split=split, streaming=streaming)
    records: list[dict[str, Any]] = []
    mask_sigs: list[tuple[bool, bool, bool, int, int]] = []
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
                comp_zones = [bbox_to_anatomical_zone(b, canon, img_w=img_w, img_h=img_h) for b in comp_boxes]
                for z in _merge_component_zones(comp_zones, canon):
                    findings.append({"label": canon, "zone": z})

        mask_sigs.append((len(uniq_set) == 0, 2 in uniq_set, 5 in uniq_set, img_w, img_h))
        records.append({
            "source": f"ChestX-Det-{split}",
            "group_id": f"chestx_det_{split}_{idx:06d}",
            "image_path": img_file,
            "findings": findings,
        })
        if len(records) % 250 == 0:
            print(f"  [ChestX-Det-{split}] {len(records)} cached...", flush=True)

    # Attempt mathematically verified 1-to-1 upgrade to official Deepwise multi-label boxes
    official = _fetch_official_chestx_det_json(split)
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
                for rec, sig, it in zip(records, mask_sigs, cand):
                    rec["findings"] = _findings_from_deepwise_item(it, sig[3], sig[4])
                print(
                    f"[Loader] ChestX-Det ({split}): VERIFIED 1-to-1 official Deepwise JSON ({order_name}: "
                    f"empty={empty_ok:.1%}, cardio={cardio_ok:.1%}, eff={eff_ok:.1%}) -> uncorrupted 13-class boxes active!",
                    flush=True,
                )
                break
        else:
            print(
                f"[Loader] ChestX-Det ({split}): official JSON order differed; using verified 0-indexed mask fallback "
                "(IDs 0..9, 11; corrupted ID 12 skipped).",
                flush=True,
            )

    print(f"[Loader] ChestX-Det ({split}): loaded {len(records)} studies", flush=True)
    return records


def load_chexpert_hf(
    max_samples: int | None = None,
    frontal_only: bool = True,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """
    Streams danjacobellis/chexpert with verified HuggingFace ClassLabel mapping:
      0 = unlabeled, 1 = uncertain, 2 = absent, 3 = present (Frontal/Lateral: 0 = Frontal).
    """
    from datasets import load_dataset

    print("[Loader] Streaming CheXpert (train, frontal only, verified 1=uncertain/2=absent/3=present)...", flush=True)
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
            elif val == 1 or str(val).strip() == "1":
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
    """
    Streams JasonZZ0601/PadChest-GR-Xray with:
      1. Study-level grouping across multi-sentence rows for the same radiograph.
      2. Strict Normal validation (unmapped abnormalities are NEVER labeled Normal).
      3. Uniform contrast normalization + standardized zone vocabulary (zero dataset fingerprints).
    """
    from datasets import load_dataset

    print(f"[Loader] Streaming PadChest-GR ({split}, grouped studies + uniform norm)...", flush=True)
    ds = load_dataset("JasonZZ0601/PadChest-GR-Xray", split=split, streaming=streaming)
    out_img_dir = Path(img_cache_dir) / f"padchest_gr_{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    studies: dict[str, dict[str, Any]] = {}
    scanned_rows = 0
    max_scan_rows = (max_samples * 4) if max_samples else None

    for idx, row in enumerate(ds):
        scanned_rows += 1
        pil_img = row.get("image")
        if pil_img is None:
            continue

        group_raw = str(row.get("PatientID") or row.get("patient_id") or row.get("ImageID") or f"{idx:06d}")
        img_fp = _fast_image_fingerprint(pil_img)
        study_key = f"{group_raw}_{img_fp}"

        if max_samples and study_key not in studies and len(studies) >= max_samples:
            break
        if max_scan_rows and scanned_rows > max_scan_rows and study_key not in studies:
            break

        if study_key not in studies:
            img_file = _save_resized_image(
                pil_img,
                out_img_dir / f"padchest_{split}_{len(studies):06d}_{img_fp}.jpg",
                image_size,
            )
            studies[study_key] = {
                "source": f"PadChest-GR-{split}",
                "group_id": f"padchest_{split}_{group_raw}",
                "image_path": img_file,
                "findings": [],
                "_seen_pairs": set(),
                "_explicit_normal": False,
                "_has_unmapped_abn": False,
            }
            if len(studies) % 100 == 0:
                print(f"  [PadChest-GR-{split}] {len(studies)} unique studies ({scanned_rows} rows)...", flush=True)

        entry = studies[study_key]
        img_w, img_h = pil_img.size

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
                if first_box is not None:
                    zone = bbox_to_anatomical_zone(first_box, canon, img_w=img_w, img_h=img_h)
                else:
                    zone = _normalize_padchest_zone(loc_str, canon)
                pair = (canon, zone)
                if pair not in entry["_seen_pairs"]:
                    entry["_seen_pairs"].add(pair)
                    entry["findings"].append({"label": canon, "zone": zone})
            else:
                entry["_has_unmapped_abn"] = True

        if not mapped_any and (parts or has_note):
            entry["_has_unmapped_abn"] = True

    records: list[dict[str, Any]] = []
    skipped_unmapped = 0
    for entry in studies.values():
        if entry["findings"]:
            records.append({
                "source": entry["source"],
                "group_id": entry["group_id"],
                "image_path": entry["image_path"],
                "findings": entry["findings"],
                "note": None,
            })
        elif entry["_explicit_normal"] and not entry["_has_unmapped_abn"]:
            records.append({
                "source": entry["source"],
                "group_id": entry["group_id"],
                "image_path": entry["image_path"],
                "findings": [],
                "note": None,
            })
        else:
            skipped_unmapped += 1

    print(
        f"[Loader] PadChest-GR ({split}): kept {len(records)} clean studies "
        f"from {scanned_rows} rows (dropped {skipped_unmapped} unmapped-only studies)",
        flush=True,
    )
    return records


def load_siim_pneumo_hf(
    max_samples: int | None = None,
    img_cache_dir: str = "./.tmp/training_images",
    image_size: int = 512,
    split: str = "train",
    streaming: bool = True,
) -> list[dict[str, Any]]:
    """Streams nwirandx/siim-pneumothorax-256-2400 with uniform contrast normalization & CC boxes."""
    import numpy as np
    from datasets import load_dataset

    print(f"[Loader] Streaming SIIM-Pneumothorax ({split}, uniform norm)...", flush=True)
    ds = load_dataset("nwirandx/siim-pneumothorax-256-2400", split=split, streaming=streaming)
    out_img_dir = Path(img_cache_dir) / f"siim_pneumo_{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples and len(records) >= max_samples:
            break
        pil_img = row.get("image")
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size
        img_file = _save_resized_image(pil_img, out_img_dir / f"siim_pneumo_{split}_{idx:06d}.jpg", image_size)

        has_pneumo = int(row.get("has_pneumo", 0) or 0) == 1
        findings: list[dict[str, Any]] = []
        if has_pneumo:
            mask_img = row.get("mask")
            comp_boxes: list[list[float]] = []
            if mask_img is not None:
                m_arr = np.asarray(mask_img)
                if m_arr.ndim == 3:
                    m_arr = m_arr[..., 0]
                comp_boxes = _extract_mask_boxes(m_arr > 0, min_pixels=8)
            if comp_boxes:
                zones = [bbox_to_anatomical_zone(b, "Pneumothorax", img_w=img_w, img_h=img_h) for b in comp_boxes]
                for z in _merge_component_zones(zones, "Pneumothorax"):
                    findings.append({"label": "Pneumothorax", "zone": z})
            else:
                findings.append({
                    "label": "Pneumothorax",
                    "zone": PATHOLOGY_ONTOLOGY["Pneumothorax"]["default_zone"],
                })

        records.append({
            "source": f"SIIM-Pneumothorax-{split}",
            "group_id": f"siim_pneumo_{split}_{idx:06d}",
            "image_path": img_file,
            "findings": findings,
        })
        if len(records) % 500 == 0:
            print(f"  [SIIM-Pneumothorax-{split}] {len(records)} cached...", flush=True)
    print(f"[Loader] SIIM-Pneumothorax ({split}): loaded {len(records)} studies", flush=True)
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


def _sample_weighted_distractors(
    absent_pool: list[str],
    rng: random.Random,
    k: int = 1,
    class_weights: dict[str, float] | None = None,
) -> list[str]:
    """Samples fake CAD alerts frequency-matched to real positive classes so the VLM cannot exploit class priors."""
    if not absent_pool:
        return ["Pneumothorax"]
    k = min(k, len(absent_pool))
    if not class_weights:
        return rng.sample(absent_pool, k=k)

    pool = list(absent_pool)
    chosen: list[str] = []
    for _ in range(k):
        wts = [max(0.25, float(class_weights.get(c, 1.0))) for c in pool]
        pick = rng.choices(pool, weights=wts, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def _build_mode3_rejection_payload(
    record: dict[str, Any],
    rng: random.Random,
    class_weights: dict[str, float] | None = None,
) -> tuple[str, list[str], str]:
    """
    Constructs a frequency-matched false CAD alert + visual counter-evidence refutation notes (v2.2):
      - normal_fp: Normal study + 1-2 fake CAD flags -> reject & conclude Normal.
      - abnormal_full_fp: Abnormal study + ONLY 1 fake CAD flag -> reject fake AND preserve true visual findings.
      - abnormal_distractor: Abnormal study + true flag(s) + 1 fake CAD flag -> confirm true, reject fake.
    """
    true_findings = record["findings"]
    present_labels = {f["label"] for f in true_findings}
    absent_pool = [c for c in _DISTRACTOR_POOL if c not in present_labels] or ["Pneumothorax"]

    if not true_findings:
        k = 2 if (len(absent_pool) >= 2 and rng.random() < 0.25) else 1
        fake_classes = _sample_weighted_distractors(absent_pool, rng, k=k, class_weights=class_weights)
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

    fake_cls = _sample_weighted_distractors(absent_pool, rng, k=1, class_weights=class_weights)[0]
    fake_meta = PATHOLOGY_ONTOLOGY[fake_cls]
    fake_zone = rng.choice(_DISTRACTOR_ZONES.get(fake_cls, [fake_meta["default_zone"]]))

    if rng.random() < 0.50:
        cad_summary = f"{fake_meta['clinical_name']} at {fake_zone}"
        ref_note = (
            f"CAD flagged suspected {fake_meta['clinical_name']} at the {fake_zone}; "
            f"however, direct visual evaluation of the radiograph does not substantiate this alert, "
            f"demonstrating {fake_meta['neg_visual']}."
        )
        return cad_summary, [ref_note], "abnormal_full_fp"

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
    class_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Formats a record into SmolVLM chat messages + validates 5-Rule compliance."""
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
        cad_summary, rejected_notes, rejection_subtype = _build_mode3_rejection_payload(
            record, local_rng, class_weights=class_weights
        )
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
    pure_ratio: float = 0.50,
    cad_aligned_ratio: float = 0.20,
) -> list[dict[str, Any]]:
    """
    Balanced 3-Mode v2.2 mixture with frequency-matched CAD distractors:
      - Abnormal studies: 50% Pure-Image, 20% CAD-Aligned, 30% CAD-Reject (half full-FP, half distractor)
      - Normal studies:   40% Pure-Image, 10% CAD-Aligned, 50% CAD-Reject (normal_fp)
    """
    rng = random.Random(seed)
    pos_counts: Counter[str] = Counter()
    for rec in records:
        for f in rec.get("findings") or []:
            pos_counts[f["label"]] += 1
    avg_c = (sum(pos_counts.values()) / max(1, len(pos_counts))) if pos_counts else 1.0
    class_weights = {
        c: max(0.35, float(pos_counts.get(c, 0)) / max(1.0, avg_c))
        for c in _DISTRACTOR_POOL
    }

    samples: list[dict[str, Any]] = []
    for rec in records:
        r = rng.random()
        if not rec["findings"]:
            if r < 0.40:
                mode = "pure"
            elif r < 0.50:
                mode = "cad_aligned"
            else:
                mode = "cad_reject"
        else:
            if r < pure_ratio:
                mode = "pure"
            elif r < pure_ratio + cad_aligned_ratio:
                mode = "cad_aligned"
            else:
                mode = "cad_reject"
        samples.append(format_vlm_chat_sample(rec, mode=mode, rng=rng, class_weights=class_weights))
    rng.shuffle(samples)
    return samples


def export_holdout_benchmarks(
    holdout_dir: str,
    img_cache_dir: str,
    image_size: int = 512,
    max_padchest: int = 150,
    max_siim: int = 120,
) -> None:
    """Exports unseen ChestX-Det (test), PadChest-GR (test), and SIIM-Pneumothorax (test) for Gate 1-4 validation."""
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
    try:
        holdout_records.extend(
            load_siim_pneumo_hf(
                max_samples=max_siim,
                split="test",
                img_cache_dir=img_cache_dir,
                image_size=image_size,
                streaming=True,
            )
        )
    except Exception as exc:
        print(f"[Holdout] Skipping SIIM-Pneumothorax test split ({exc})")

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
        train_samples = _emit_three_mode_samples(train_recs, seed=args.seed, pure_ratio=0.50, cad_aligned_ratio=0.20)
        val_samples = _emit_three_mode_samples(val_recs, seed=args.seed + 1, pure_ratio=0.50, cad_aligned_ratio=0.20)

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