"""
ChestXStudio: Standalone 4-Model Multi-Resolution Ensemble Evaluator
====================================================================
Runs the 4-Model (224x224 DenseNet121 x3 + 512x512 ResNet50 x1) Dual-Gate + Quorum
pipeline over a folder of chest X-rays or a single image file, printing per-image
diagnostic tables, near-miss watchlists, and saving a CSV audit log.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv
from PIL import Image
from skimage import exposure

# =====================================================================
# 1. DUAL-GATE CLINICAL CALIBRATION & TERMINOLOGY TABLE
# =====================================================================
CLASS_CALIBRATION_GATES = {
    "Pneumothorax": {
        "min_cal": 0.54, "min_raw": 0.14,
        "observation": "visceral pleural line with peripheral pleural lucency and absent lung markings",
        "clinical_name": "pneumothorax"
    },
    "Effusion": {
        "min_cal": 0.55, "min_raw": 0.25,
        "observation": "blunting of the costophrenic angle with meniscus-shaped fluid density",
        "clinical_name": "pleural effusion"
    },
    "Consolidation": {
        "min_cal": 0.55, "min_raw": 0.22,
        "observation": "confluent homogeneous airspace opacification",
        "clinical_name": "focal lobar/segmental consolidation"
    },
    "Pneumonia": {
        "min_cal": 0.57, "min_raw": 0.18,
        "observation": "patchy airspace consolidation concerning for infectious process",
        "clinical_name": "pneumonia"
    },
    "Edema": {
        "min_cal": 0.56, "min_raw": 0.22,
        "observation": "perihilar vascular haziness, interstitial prominence, and vascular redistribution",
        "clinical_name": "pulmonary edema / vascular congestion"
    },
    "Atelectasis": {
        "min_cal": 0.55, "min_raw": 0.28,
        "observation": "linear or subsegmental basilar opacity with associated volume loss",
        "clinical_name": "atelectasis"
    },
    "Cardiomegaly": {
        "min_cal": 0.55, "min_raw": 0.28,
        "observation": "enlarged cardiac silhouette with increased cardiothoracic ratio",
        "clinical_name": "cardiomegaly"
    },
    "Enlarged Cardiomediastinum": {
        "min_cal": 0.58, "min_raw": 0.30,
        "observation": "widened cardiomediastinal contour",
        "clinical_name": "enlarged cardiomediastinal silhouette"
    },
    "Lung Opacity": {
        "min_cal": 0.56, "min_raw": 0.38,
        "observation": "increased parenchymal airspace opacity",
        "clinical_name": "pulmonary airspace opacity"
    },
    "Infiltration": {
        "min_cal": 0.60, "min_raw": 0.32,
        "observation": "patchy parenchymal infiltrate",
        "clinical_name": "pulmonary infiltrate"
    },
    "Mass": {
        "min_cal": 0.57, "min_raw": 0.18,
        "observation": "discrete focal mass-like radio-opacity",
        "clinical_name": "pulmonary mass"
    },
    "Nodule": {
        "min_cal": 0.58, "min_raw": 0.20,
        "observation": "focal rounded nodular radio-opacity",
        "clinical_name": "pulmonary nodule"
    },
    "Lung Lesion": {
        "min_cal": 0.58, "min_raw": 0.20,
        "observation": "focal parenchymal lesion",
        "clinical_name": "pulmonary lesion"
    },
    "Emphysema": {
        "min_cal": 0.64, "min_raw": 0.20,
        "observation": "hyperinflated lung fields with flattened hemidiaphragms",
        "clinical_name": "emphysema / hyperinflation"
    },
    "Fibrosis": {
        "min_cal": 0.65, "min_raw": 0.22,
        "observation": "coarse reticular interstitial markings and architectural distortion",
        "clinical_name": "pulmonary fibrosis"
    },
    "Pleural_Thickening": {
        "min_cal": 0.65, "min_raw": 0.22,
        "observation": "pleural margin thickening along the lateral thoracic wall",
        "clinical_name": "pleural thickening"
    },
    "Fracture": {
        "min_cal": 0.62, "min_raw": 0.22,
        "observation": "cortical discontinuity along the osseous thoracic cage",
        "clinical_name": "rib or clavicular fracture"
    },
    "Hernia": {
        "min_cal": 0.70, "min_raw": 0.25,
        "observation": "retrocardiac density consistent with hiatal hernia",
        "clinical_name": "hiatal hernia"
    }
}

MEDIASTINAL_LABELS = {"Cardiomegaly", "Enlarged Cardiomediastinum", "Hernia"}
DIFFUSE_BILATERAL_LABELS = {"Edema", "Emphysema"}
LOCALIZED_FINE_LABELS = {"Pneumothorax", "Nodule", "Mass", "Fracture", "Lung Lesion"}
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# =====================================================================
# 2. MODEL LOADER & VECTORIZED OP_NORM CALIBRATOR
# =====================================================================
def load_multi_resolution_ensemble(device_str: str = "cpu") -> tuple[list[dict], dict | None]:
    print("=" * 78)
    print(f"[ChestXStudio Evaluator] Loading 4-Model Ensemble on {device_str.upper()}...")
    densenet_specs = [
        ("all_224", "densenet121-res224-all", 1.00),
        ("chex_224", "densenet121-res224-chex", 1.15),
        ("mimic_224", "densenet121-res224-mimic_ch", 1.15),
    ]

    ensemble_224 = []
    for short_name, weight_id, vote_weight in densenet_specs:
        try:
            print(f" -> Loading 224x224 specialist: {weight_id}")
            m = xrv.models.DenseNet(weights=weight_id).to(device=device_str, dtype=torch.float32)
            m.eval()
            op_t = m.op_threshs.detach().cpu().numpy().copy()
            ensemble_224.append({
                "name": short_name,
                "weight_id": weight_id,
                "model": m,
                "vote_weight": vote_weight,
                "op_threshs": op_t,
                "pathologies": list(m.pathologies)
            })
        except Exception as e:
            print(f"    [Warning] Could not load {weight_id}: {e}")

    resnet_512 = None
    try:
        print(" -> Loading 512x512 specialist: resnet50-res512-all")
        r512 = xrv.models.ResNet(weights="resnet50-res512-all").to(device=device_str, dtype=torch.float32)
        r512.eval()
        r512_op_t = r512.op_threshs.detach().cpu().numpy().copy()
        resnet_512 = {
            "name": "resnet50_512",
            "weight_id": "resnet50-res512-all",
            "model": r512,
            "vote_weight": 1.25,
            "op_threshs": r512_op_t,
            "pathologies": list(r512.pathologies)
        }
    except Exception as e:
        print(f"    [Warning] Could not load resnet50-res512-all: {e}")

    if not ensemble_224:
        raise RuntimeError("Fatal: No 224x224 DenseNet models could be loaded.")

    total = len(ensemble_224) + (1 if resnet_512 is not None else 0)
    print(f"[ChestXStudio Evaluator] {total} clinical models initialized and ready.")
    print("=" * 78 + "\n")
    return ensemble_224, resnet_512


def compute_op_norm(raw_probs: np.ndarray, op_threshs: np.ndarray) -> np.ndarray:
    cal = np.full_like(raw_probs, np.nan, dtype=np.float32)
    for c_idx in range(raw_probs.shape[1]):
        t = float(op_threshs[c_idx])
        if np.isnan(t) or t <= 0.0 or t >= 1.0:
            continue
        p = raw_probs[:, c_idx]
        low_mask = p < t
        high_mask = ~low_mask
        cal[low_mask, c_idx] = (p[low_mask] / t) * 0.5
        cal[high_mask, c_idx] = ((p[high_mask] - t) / (1.0 - t)) * 0.5 + 0.5
    return cal


# =====================================================================
# 3. 16-BIT-SAFE PREPROCESSING & ASPECT-PRESERVED MULTI-SCALE TTA
# =====================================================================
def extract_grayscale_float32(image: Image.Image) -> np.ndarray:
    """
    Extracts a 2D float32 grayscale array while preserving 12-bit/16-bit PNG
    dynamic range (avoids PIL .convert('L') clipping values > 255).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            return 0.299 * r + 0.587 * g + 0.114 * b
        return arr[:, :, 0].astype(np.float32)
    return np.asarray(image.convert("L"), dtype=np.float32)


def prepare_multiscale_tta(
    image: Image.Image,
    device_str: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    raw_np = extract_grayscale_float32(image)
    h, w = raw_np.shape

    p1, p99 = np.percentile(raw_np, (1.0, 99.0))
    if (p99 - p1) > 1e-3:
        linear_0_255 = np.clip((raw_np - p1) * (255.0 / (p99 - p1)), 0.0, 255.0).astype(np.float32)
    else:
        linear_0_255 = np.clip(raw_np, 0.0, 255.0).astype(np.float32)

    unit_linear = np.clip(linear_0_255 / 255.0, 0.0, 1.0)
    clahe_0_255 = (exposure.equalize_adapthist(unit_linear, clip_limit=0.012).astype(np.float32)) * 255.0
    edge_0_255 = (0.80 * linear_0_255 + 0.20 * clahe_0_255).astype(np.float32)

    norm_linear = xrv.datasets.normalize(linear_0_255, 255.0).astype(np.float32)
    norm_edge = xrv.datasets.normalize(edge_0_255, 255.0).astype(np.float32)

    # View 0: Full-Anatomy Letterbox Square
    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    view0_square = np.pad(
        norm_linear,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=-1024.0
    )

    # View 1: 90% Central Thoracic Square Crop
    crop_side = max(16, int(min(h, w) * 0.90))
    cy0 = max(0, (h - crop_side) // 2)
    cx0 = max(0, (w - crop_side) // 2)
    view1_square = norm_linear[cy0:cy0 + crop_side, cx0:cx0 + crop_side]

    # View 2: 82% Upper-Thoracic Aspect-Preserved Square Crop (Edge-Enhanced)
    apical_side = max(16, int(min(h, w) * 0.82))
    ay0 = min(max(0, int(h * 0.03)), max(0, h - apical_side))
    ax0 = max(0, (w - apical_side) // 2)
    view2_square = norm_edge[ay0:ay0 + apical_side, ax0:ax0 + apical_side]

    t_v0 = torch.from_numpy(view0_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t_v1 = torch.from_numpy(view1_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t_v2 = torch.from_numpy(view2_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    tta_224 = torch.cat([
        F.interpolate(t_v0, size=(224, 224), mode="bilinear", align_corners=False),
        F.interpolate(t_v1, size=(224, 224), mode="bilinear", align_corners=False),
        F.interpolate(t_v2, size=(224, 224), mode="bilinear", align_corners=False),
    ], dim=0).to(device=device_str, dtype=torch.float32)

    tta_512 = torch.cat([
        F.interpolate(t_v0, size=(512, 512), mode="bilinear", align_corners=False),
        F.interpolate(t_v2, size=(512, 512), mode="bilinear", align_corners=False),
    ], dim=0).to(device=device_str, dtype=torch.float32)

    pad_meta = {
        "h": h, "w": w, "max_dim": max_dim,
        "pad_top": pad_top, "pad_left": pad_left,
        "display_uint8": linear_0_255.astype(np.uint8)
    }
    return tta_224, tta_512, pad_meta


# =====================================================================
# 4. SPATIAL CAM LOCALIZATION & HEATMAP RENDERER
# =====================================================================
def localize_from_cam_grid(cam_7x7: np.ndarray, label_name: str) -> str:
    if label_name in MEDIASTINAL_LABELS:
        return "Central mediastinum and cardiac silhouette"
    if label_name in DIFFUSE_BILATERAL_LABELS:
        return "Bilateral lung fields and perihilar regions"

    grid = np.maximum(cam_7x7.copy(), 0.0)
    grid[0, 0] = grid[0, 6] = grid[6, 0] = grid[6, 6] = 0.0

    right_roi = grid[1:6, 0:3]  # Image-left = Patient's Right Hemithorax
    left_roi = grid[1:6, 4:7]   # Image-right = Patient's Left Hemithorax

    right_peak = float(np.max(right_roi)) if right_roi.size else 0.0
    left_peak = float(np.max(left_roi)) if left_roi.size else 0.0
    max_peak = max(right_peak, left_peak)

    if max_peak <= 1e-6:
        return "Bilateral lung parenchyma and thoracic cavity"

    is_bilateral = (min(right_peak, left_peak) >= 0.72 * max_peak)

    if is_bilateral:
        side_str = "Bilateral"
        active_slice = (right_roi + left_roi) / 2.0
    elif right_peak >= left_peak:
        side_str = "Right"
        active_slice = right_roi
    else:
        side_str = "Left"
        active_slice = left_roi

    row_weights = np.sum(active_slice, axis=1)
    if float(np.sum(row_weights)) > 1e-6:
        mean_row = float(np.sum(np.arange(1, 6) * row_weights) / np.sum(row_weights))
    else:
        mean_row = 3.0

    if label_name == "Effusion" and mean_row >= 2.7:
        if is_bilateral:
            return "Bilateral costophrenic angles and lower pleural spaces"
        return f"{side_str} costophrenic angle and lower pleural space"

    if label_name == "Pneumothorax":
        if mean_row <= 2.4:
            return f"{side_str} apical pleural space"
        elif mean_row <= 3.6:
            return f"{side_str} apicolateral pleural space"
        return f"{side_str} lateral and basilar pleural space"

    if mean_row <= 2.2:
        zone_str = "upper lung zone (apical)"
    elif mean_row <= 3.5:
        zone_str = "mid lung zone"
    else:
        if is_bilateral and label_name in ("Atelectasis", "Consolidation", "Infiltration", "Lung Opacity"):
            return "Bibasilar lower lung zones"
        zone_str = "lower lung zone / lung base"

    return f"{side_str} {zone_str}"


def render_cam_heatmap_overlay(cam_7x7: np.ndarray, pad_meta: dict) -> Image.Image:
    gray_u8 = pad_meta["display_uint8"]
    base_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    h, w = gray_u8.shape
    max_dim = pad_meta["max_dim"]
    pad_top = pad_meta["pad_top"]
    pad_left = pad_meta["pad_left"]

    grid = np.maximum(cam_7x7.copy(), 0.0)
    grid[0, 0] = grid[0, 6] = grid[6, 0] = grid[6, 6] = 0.0
    g_min, g_max = float(np.min(grid)), float(np.max(grid))
    if (g_max - g_min) > 1e-6:
        norm_grid = (grid - g_min) / (g_max - g_min)
    else:
        norm_grid = np.zeros_like(grid)

    cam_uint8 = (norm_grid * 255.0).astype(np.uint8)
    cam_square = Image.fromarray(cam_uint8, mode="L").resize((max_dim, max_dim), resample=Image.BICUBIC)
    cam_square_np = np.asarray(cam_square, dtype=np.float32) / 255.0
    v = cam_square_np[pad_top:pad_top + h, pad_left:pad_left + w]

    if v.shape != (h, w):
        v = np.asarray(
            Image.fromarray((v * 255.0).astype(np.uint8), mode="L").resize((w, h), resample=Image.BICUBIC),
            dtype=np.float32
        ) / 255.0

    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0) * 255.0
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0) * 255.0
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0) * 255.0
    heatmap_rgb = np.stack([r, g, b], axis=-1)

    alpha = np.clip((v ** 1.6) * 0.48, 0.0, 0.48)[..., None]
    blended = (1.0 - alpha) * base_rgb + alpha * heatmap_rgb
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))


# =====================================================================
# 5. ENSEMBLE INFERENCE & DUAL-GATE + QUORUM EVALUATOR
# =====================================================================
def pool_tta_views(
    cal_vec: np.ndarray,
    raw_vec: np.ndarray,
    is_localized: bool
) -> tuple[float, float]:
    if len(cal_vec) == 3:
        mean_c = float(0.50 * cal_vec[0] + 0.35 * cal_vec[1] + 0.15 * cal_vec[2])
        mean_r = float(0.50 * raw_vec[0] + 0.35 * raw_vec[1] + 0.15 * raw_vec[2])
    else:
        mean_c = float(0.65 * cal_vec[0] + 0.35 * cal_vec[1])
        mean_r = float(0.65 * raw_vec[0] + 0.35 * raw_vec[1])

    if is_localized and float(cal_vec[0]) >= 0.36:
        max_c = float(np.max(cal_vec))
        max_r = float(np.max(raw_vec))
        return (0.55 * max_c + 0.45 * mean_c), (0.55 * max_r + 0.45 * mean_r)

    return mean_c, mean_r


def evaluate_single_image(
    image: Image.Image,
    ensemble_224: list[dict],
    resnet_512: dict | None,
    sensitivity_offset: float = 0.0,
    device_str: str = "cpu"
) -> dict:
    tta_224, tta_512, pad_meta = prepare_multiscale_tta(image, device_str=device_str)

    agg_cal = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_raw = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_weights = {label: [] for label in CLASS_CALIBRATION_GATES}
    cam_accum = {label: np.zeros((7, 7), dtype=np.float32) for label in CLASS_CALIBRATION_GATES}

    with torch.no_grad():
        for entry in ensemble_224:
            m = entry["model"]
            vote_w = entry["vote_weight"]
            path_list = entry["pathologies"]
            op_t = entry["op_threshs"]

            raw_feats = m.features(tta_224)
            relu_feats = F.relu(raw_feats, inplace=False)
            pooled = F.adaptive_avg_pool2d(relu_feats, (1, 1)).flatten(1)
            logits = m.classifier(pooled)

            raw_probs_batch = torch.sigmoid(logits).cpu().numpy()
            cal_probs_batch = compute_op_norm(raw_probs_batch, op_t)
            full_coord_feat_7x7 = 0.65 * relu_feats[0] + 0.35 * relu_feats[1]

            for idx, label in enumerate(path_list):
                if not label or label not in CLASS_CALIBRATION_GATES:
                    continue
                if np.isnan(cal_probs_batch[0, idx]) or np.isnan(raw_probs_batch[0, idx]):
                    continue

                is_loc = label in LOCALIZED_FINE_LABELS
                c_val, r_val = pool_tta_views(cal_probs_batch[:, idx], raw_probs_batch[:, idx], is_loc)

                agg_cal[label].append(c_val)
                agg_raw[label].append(r_val)
                agg_weights[label].append(vote_w)

                cls_w = m.classifier.weight[idx]
                cam_2d = torch.matmul(cls_w, full_coord_feat_7x7.reshape(1024, -1)).reshape(7, 7).cpu().numpy()
                cam_accum[label] += cam_2d * vote_w

        if resnet_512 is not None:
            r_model = resnet_512["model"]
            r_op_t = resnet_512["op_threshs"]
            r_paths = resnet_512["pathologies"]
            base_w = resnet_512["vote_weight"]

            # Call underlying torchvision ResNet directly to guarantee unscaled raw logits
            logits_512 = r_model.model(tta_512)
            raw_512 = torch.sigmoid(logits_512).cpu().numpy()
            cal_512 = compute_op_norm(raw_512, r_op_t)

            for idx, label in enumerate(r_paths):
                if not label or label not in CLASS_CALIBRATION_GATES:
                    continue
                if np.isnan(cal_512[0, idx]) or np.isnan(raw_512[0, idx]):
                    continue

                is_loc = label in LOCALIZED_FINE_LABELS
                c_val, r_val = pool_tta_views(cal_512[:, idx], raw_512[:, idx], is_loc)
                eff_w = base_w * 1.25 if is_loc else base_w

                agg_cal[label].append(c_val)
                agg_raw[label].append(r_val)
                agg_weights[label].append(eff_w)

    detected_findings = []
    subthreshold_findings = []
    composite_cam = np.zeros((7, 7), dtype=np.float32)

    for label, gate in CLASS_CALIBRATION_GATES.items():
        weights = agg_weights[label]
        if not weights or sum(weights) <= 0:
            continue

        c_list = agg_cal[label]
        r_list = agg_raw[label]
        mean_cal = float(np.average(c_list, weights=weights))
        mean_raw = float(np.average(r_list, weights=weights))

        cam_w_sum = sum(w for w in weights[:len(ensemble_224)]) or 1.0
        mean_cam = cam_accum[label] / cam_w_sum

        effective_min_cal = float(np.clip(gate["min_cal"] - sensitivity_offset, 0.48, 0.78))
        effective_min_raw = float(np.clip(gate["min_raw"] - (sensitivity_offset * 0.4), 0.08, 0.50))

        num_models = len(c_list)
        supporting_models = sum(
            1 for c_m, r_m in zip(c_list, r_list)
            if c_m >= (effective_min_cal - 0.05) and r_m >= (effective_min_raw * 0.75)
        )
        if num_models >= 3:
            quorum_ok = (supporting_models >= 2) or (mean_cal >= effective_min_cal + 0.08)
        else:
            quorum_ok = True

        passed_gate = (mean_cal >= effective_min_cal) and (mean_raw >= effective_min_raw) and quorum_ok
        location = localize_from_cam_grid(mean_cam, label)

        record = {
            "raw_label": label,
            "clinical_name": gate["clinical_name"],
            "observation": gate["observation"],
            "location": location,
            "cal_prob": mean_cal,
            "raw_prob": mean_raw,
            "min_cal": effective_min_cal,
            "min_raw": effective_min_raw,
            "quorum_str": f"{supporting_models}/{num_models}",
            "passed_gate": passed_gate
        }

        if passed_gate:
            detected_findings.append(record)
            composite_cam += np.maximum(mean_cam, 0.0) * mean_cal
        else:
            subthreshold_findings.append(record)

    detected_findings.sort(key=lambda d: d["cal_prob"], reverse=True)
    subthreshold_findings.sort(key=lambda d: d["cal_prob"], reverse=True)

    specific_airspace = {"Consolidation", "Pneumonia", "Atelectasis", "Edema"}
    if any(d["raw_label"] in specific_airspace for d in detected_findings):
        detected_findings = [
            d for d in detected_findings
            if d["raw_label"] not in ("Lung Opacity", "Infiltration")
        ]

    if any(d["raw_label"] == "Cardiomegaly" for d in detected_findings):
        detected_findings = [
            d for d in detected_findings
            if d["raw_label"] != "Enlarged Cardiomediastinum"
        ]

    top_findings = detected_findings[:4]
    return {
        "top_findings": top_findings,
        "near_misses": subthreshold_findings[:3],
        "composite_cam": composite_cam,
        "pad_meta": pad_meta
    }


# =====================================================================
# 6. DIRECTORY DISCOVERY & CLI RUNNER
# =====================================================================
def discover_xray_images(target_path: str | None) -> list[Path]:
    if target_path:
        p = Path(target_path).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            return [p]
        if p.is_dir():
            files = [
                f for f in sorted(p.rglob("*"))
                if f.is_file()
                and f.suffix.lower() in VALID_EXTENSIONS
                and "ensemble_audit_outputs" not in str(f)
                and ".venv" not in str(f)
            ]
            return files
        raise FileNotFoundError(f"No valid images found at specified path: {target_path}")

    # Auto-discover images in standard project folders if no path was provided
    cwd = Path.cwd()
    candidate_dirs = [
        cwd / "test_images",
        cwd / "images",
        cwd / "samples",
        cwd
    ]
    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            found = [
                f for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            if found:
                return found
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-Model (224+512) Dual-Gate Chest X-Ray Ensemble on local images."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a folder of X-ray images or a single X-ray file (auto-discovers if omitted)."
    )
    parser.add_argument(
        "--mode",
        choices=["balanced", "sensitive", "specific"],
        default="balanced",
        help="Operating threshold mode: balanced (0.0), sensitive (+0.04), or specific (-0.05)."
    )
    parser.add_argument(
        "--save-cam",
        action="store_true",
        help="Save padding-corrected CAM heatmap overlay images to ./ensemble_audit_outputs/."
    )
    args = parser.parse_args()

    sens_map = {"balanced": 0.0, "sensitive": 0.04, "specific": -0.05}
    sens_offset = sens_map[args.mode]

    image_paths = discover_xray_images(args.path)
    if not image_paths:
        print(
            "[ChestXStudio Evaluator] No image files (.png/.jpg/.jpeg) found in current folder.\n"
            "Usage: python3 evaluate_ensemble.py /path/to/xray_folder_or_file.png [--mode sensitive] [--save-cam]"
        )
        sys.exit(1)

    ensemble_224, resnet_512 = load_multi_resolution_ensemble(device_str="cpu")

    out_dir = Path("./ensemble_audit_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ensemble_audit_results.csv"

    print(f"Evaluating {len(image_paths)} radiograph(s) | Mode: {args.mode.upper()} (offset={sens_offset:+.2f})\n")

    csv_rows = []
    abnormal_count = 0
    normal_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        t0 = time.perf_counter()
        try:
            pil_img = Image.open(img_path)
            pil_img.load()
        except Exception as e:
            print(f"[{idx}/{len(image_paths)}] SKIPPED {img_path.name} (Unreadable image: {e})")
            continue

        res = evaluate_single_image(
            pil_img,
            ensemble_224=ensemble_224,
            resnet_512=resnet_512,
            sensitivity_offset=sens_offset,
            device_str="cpu"
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
            impression_summary = "; ".join(
                f"{d['clinical_name']} ({d['location']}, {d['cal_prob']:.0%} cal)"
                for d in findings
            )
            if args.save_cam:
                cam_img = render_cam_heatmap_overlay(res["composite_cam"], meta)
                cam_file = out_dir / f"cam_{img_path.stem}.png"
                cam_img.save(cam_file)
                print(f" -> Saved CAM Overlay: {cam_file}")
        else:
            normal_count += 1
            impression_summary = "No acute cardiopulmonary abnormality (All 18 classes below Dual-Gate)"
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
            "top_subthreshold": f"{near_misses[0]['clinical_name']} ({near_misses[0]['cal_prob']:.1%})" if near_misses else "-"
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