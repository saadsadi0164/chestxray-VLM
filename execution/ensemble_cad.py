"""
ChestXStudio-VLM: Multi-Resolution Ensemble CAD & Spatial CAM Engine
Path: execution/ensemble_cad.py

Single source of truth for:
  - 4-Model Multi-Resolution Ensemble (224x224 DenseNet121 x3 + 512x512 ResNet50 x1)
  - 16-bit-safe grayscale extraction & aspect-preserved 3-view TTA preprocessing
  - Vectorized piecewise op_norm calibration & Dual-Gate + Quorum consensus
  - Resolution-agnostic (16x16 fused DenseNet + ResNet50) Class Activation Mapping (CAM)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from skimage import exposure

# =====================================================================
# 1. DUAL-GATE CLINICAL CALIBRATION & TERMINOLOGY TABLE
# =====================================================================
CLASS_CALIBRATION_GATES: dict[str, dict[str, Any]] = {
    "Pneumothorax": {
        "min_cal": 0.54, "min_raw": 0.14,
        "observation": "visceral pleural line with peripheral pleural lucency and absent lung markings",
        "clinical_name": "pneumothorax",
    },
    "Effusion": {
        "min_cal": 0.55, "min_raw": 0.25,
        "observation": "blunting of the costophrenic angle with meniscus-shaped fluid density",
        "clinical_name": "pleural effusion",
    },
    "Consolidation": {
        "min_cal": 0.55, "min_raw": 0.22,
        "observation": "confluent homogeneous airspace opacification",
        "clinical_name": "focal lobar/segmental consolidation",
    },
    "Pneumonia": {
        "min_cal": 0.57, "min_raw": 0.18,
        "observation": "patchy airspace consolidation concerning for infectious process",
        "clinical_name": "pneumonia",
    },
    "Edema": {
        "min_cal": 0.56, "min_raw": 0.22,
        "observation": "perihilar vascular haziness, interstitial prominence, and vascular redistribution",
        "clinical_name": "pulmonary edema / vascular congestion",
    },
    "Atelectasis": {
        "min_cal": 0.55, "min_raw": 0.28,
        "observation": "linear or subsegmental basilar opacity with associated volume loss",
        "clinical_name": "atelectasis",
    },
    "Cardiomegaly": {
        "min_cal": 0.55, "min_raw": 0.28,
        "observation": "enlarged cardiac silhouette with increased cardiothoracic ratio",
        "clinical_name": "cardiomegaly",
    },
    "Enlarged Cardiomediastinum": {
        "min_cal": 0.58, "min_raw": 0.30,
        "observation": "widened cardiomediastinal contour",
        "clinical_name": "enlarged cardiomediastinal silhouette",
    },
    "Lung Opacity": {
        "min_cal": 0.56, "min_raw": 0.38,
        "observation": "increased parenchymal airspace opacity",
        "clinical_name": "pulmonary airspace opacity",
    },
    "Infiltration": {
        "min_cal": 0.60, "min_raw": 0.32,
        "observation": "patchy parenchymal infiltrate",
        "clinical_name": "pulmonary infiltrate",
    },
    "Mass": {
        "min_cal": 0.57, "min_raw": 0.18,
        "observation": "discrete focal mass-like radio-opacity",
        "clinical_name": "pulmonary mass",
    },
    "Nodule": {
        "min_cal": 0.58, "min_raw": 0.20,
        "observation": "focal rounded nodular radio-opacity",
        "clinical_name": "pulmonary nodule",
    },
    "Lung Lesion": {
        "min_cal": 0.58, "min_raw": 0.20,
        "observation": "focal parenchymal lesion",
        "clinical_name": "pulmonary lesion",
    },
    "Emphysema": {
        "min_cal": 0.64, "min_raw": 0.20,
        "observation": "hyperinflated lung fields with flattened hemidiaphragms",
        "clinical_name": "emphysema / hyperinflation",
    },
    "Fibrosis": {
        "min_cal": 0.65, "min_raw": 0.22,
        "observation": "coarse reticular interstitial markings and architectural distortion",
        "clinical_name": "pulmonary fibrosis",
    },
    "Pleural_Thickening": {
        "min_cal": 0.65, "min_raw": 0.22,
        "observation": "pleural margin thickening along the lateral thoracic wall",
        "clinical_name": "pleural thickening",
    },
    "Fracture": {
        "min_cal": 0.62, "min_raw": 0.22,
        "observation": "cortical discontinuity along the osseous thoracic cage",
        "clinical_name": "rib or clavicular fracture",
    },
    "Hernia": {
        "min_cal": 0.70, "min_raw": 0.25,
        "observation": "retrocardiac density consistent with hiatal hernia",
        "clinical_name": "hiatal hernia",
    },
}

MEDIASTINAL_LABELS = {"Cardiomegaly", "Enlarged Cardiomediastinum", "Hernia"}
DIFFUSE_BILATERAL_LABELS = {"Edema", "Emphysema"}
LOCALIZED_FINE_LABELS = {"Pneumothorax", "Nodule", "Mass", "Fracture", "Lung Lesion"}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DENSENET_SPECS = [
    ("all_224", "densenet121-res224-all", 1.00),
    ("chex_224", "densenet121-res224-chex", 1.15),
    ("mimic_224", "densenet121-res224-mimic_ch", 1.15),
]


# =====================================================================
# 2. CACHE RESOLUTION & ENSEMBLE LOADER
# =====================================================================
def resolve_cnn_device(preferred: str = "auto") -> str:
    """Resolves optimal PyTorch device for float32 CNN inference."""
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_xrv_cache_dir(weight_id: str) -> str | None:
    """
    Returns workspace-local cache_dir if the weight file exists there and default
    ~/.torchxrayvision/models_data is inaccessible (e.g., sandboxed execution).
    """
    url = xrv.models.model_urls.get(weight_id, {}).get("url", "")
    filename = os.path.basename(url) if url else ""
    local_dir = Path("./.tmp/torchxrayvision/models_data")
    default_dir = Path.home() / ".torchxrayvision" / "models_data"

    if filename:
        try:
            if (default_dir / filename).is_file():
                return None
        except OSError:
            pass
        if (local_dir / filename).is_file():
            return str(local_dir)
    return None


def load_multi_resolution_ensemble(
    device_str: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Loads the 4-model multi-resolution vision ensemble in float32:
      - 3x DenseNet121 (224x224): all, chex, mimic_ch
      - 1x ResNet50 (512x512): resnet50-res512-all
    """
    device_str = resolve_cnn_device(device_str)
    print(f"[CAD Engine] Loading 4-Model Multi-Resolution Ensemble on {device_str.upper()}...")

    ensemble_224: list[dict[str, Any]] = []
    for short_name, weight_id, vote_weight in DENSENET_SPECS:
        try:
            cache_dir = _resolve_xrv_cache_dir(weight_id)
            print(f" -> Loading 224x224 DenseNet specialist: {weight_id}")
            m = xrv.models.DenseNet(weights=weight_id, cache_dir=cache_dir)
            m = m.to(device=device_str, dtype=torch.float32)
            m.eval()
            op_t = m.op_threshs.detach().cpu().numpy().copy()
            ensemble_224.append({
                "name": short_name,
                "weight_id": weight_id,
                "model": m,
                "vote_weight": vote_weight,
                "op_threshs": op_t,
                "pathologies": list(m.pathologies),
            })
        except Exception as exc:
            print(f"    [Warning] Could not load {weight_id}: {exc}")

    resnet_512: dict[str, Any] | None = None
    try:
        cache_dir_r = _resolve_xrv_cache_dir("resnet50-res512-all")
        print(" -> Loading 512x512 ResNet50 specialist: resnet50-res512-all")
        r512 = xrv.models.ResNet(weights="resnet50-res512-all", cache_dir=cache_dir_r)
        r512 = r512.to(device=device_str, dtype=torch.float32)
        r512.eval()
        r512_op_t = r512.op_threshs.detach().cpu().numpy().copy()
        resnet_512 = {
            "name": "resnet50_512",
            "weight_id": "resnet50-res512-all",
            "model": r512,
            "vote_weight": 1.25,
            "op_threshs": r512_op_t,
            "pathologies": list(r512.pathologies),
        }
    except Exception as exc:
        print(f"    [Warning] Could not load resnet50-res512-all: {exc}")

    if not ensemble_224:
        raise RuntimeError("Fatal: No 224x224 DenseNet models could be loaded.")

    total = len(ensemble_224) + (1 if resnet_512 is not None else 0)
    print(f"[CAD Engine] {total} clinical models initialized on {device_str.upper()}.")
    return ensemble_224, resnet_512


# =====================================================================
# 3. CALIBRATION & 16-BIT SAFE PREPROCESSING
# =====================================================================
def compute_op_norm(raw_probs: np.ndarray, op_threshs: np.ndarray) -> np.ndarray:
    """
    Exact vectorized implementation of TorchXRayVision's piecewise op_norm
    calibration, avoiding redundant forward passes.
    """
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
    device_str: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Builds aspect-ratio-preserved multi-scale TTA batches:
      - View 0: Full-Anatomy Letterbox Square (linear 1st-99th percentile stretch)
      - View 1: 90% Central Thoracic Square Crop
      - View 2: 82% Upper-Thoracic Aspect-Preserved Crop (80% Linear + 20% CLAHE)
    """
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
        constant_values=-1024.0,
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
        "h": h,
        "w": w,
        "max_dim": max_dim,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "display_uint8": linear_0_255.astype(np.uint8),
    }
    return tta_224, tta_512, pad_meta


# =====================================================================
# 4. RESOLUTION-AGNOSTIC SPATIAL CAM LOCALIZATION & HEATMAP OVERLAY
# =====================================================================
def _mask_corner_borders(grid: np.ndarray) -> np.ndarray:
    """Suppresses extreme outer corners (image markers / shoulders) safely for any NxN grid."""
    n = grid.shape[0]
    corner = max(1, round(n * 0.14))
    cleaned = np.maximum(grid.copy(), 0.0)
    cleaned[:corner, :corner] = 0.0
    cleaned[:corner, -corner:] = 0.0
    cleaned[-corner:, :corner] = 0.0
    cleaned[-corner:, -corner:] = 0.0
    return cleaned


def localize_from_cam_grid(cam_grid: np.ndarray, label_name: str) -> str:
    """
    Computes anatomical localization from a full-coordinate NxN CAM (7x7 or 16x16).
    Note: Image-left corresponds to Patient's Right Hemithorax;
          Image-right corresponds to Patient's Left Hemithorax.
    """
    if label_name in MEDIASTINAL_LABELS:
        return "Central mediastinum and cardiac silhouette"
    if label_name in DIFFUSE_BILATERAL_LABELS:
        return "Bilateral lung fields and perihilar regions"

    grid = _mask_corner_borders(cam_grid)
    n = grid.shape[0]

    r_start = max(1, int(round(n * 0.14)))
    r_end = min(n, int(round(n * 0.86)))
    right_col_end = max(1, int(round(n * 0.43)))
    left_col_start = min(n - 1, max(right_col_end + 1, int(round(n * 0.57))))

    right_roi = grid[r_start:r_end, 0:right_col_end]
    left_roi = grid[r_start:r_end, left_col_start:n]

    right_peak = float(np.max(right_roi)) if right_roi.size else 0.0
    left_peak = float(np.max(left_roi)) if left_roi.size else 0.0
    max_peak = max(right_peak, left_peak)

    if max_peak <= 1e-6:
        return "Bilateral lung parenchyma and thoracic cavity"

    is_bilateral = min(right_peak, left_peak) >= 0.72 * max_peak
    if is_bilateral:
        side_str = "Bilateral"
        active_slice = 0.5 * (
            right_roi[:, :min(right_roi.shape[1], left_roi.shape[1])]
            + left_roi[:, :min(right_roi.shape[1], left_roi.shape[1])]
        )
    elif right_peak >= left_peak:
        side_str = "Right"
        active_slice = right_roi
    else:
        side_str = "Left"
        active_slice = left_roi

    row_weights = np.sum(active_slice, axis=1)
    num_rows = active_slice.shape[0]
    if float(np.sum(row_weights)) > 1e-6 and num_rows > 1:
        # Normalized vertical position within lung field: 0.0 (apex) -> 1.0 (base)
        norm_coords = np.linspace(0.0, 1.0, num_rows)
        norm_y = float(np.sum(norm_coords * row_weights) / np.sum(row_weights))
    else:
        norm_y = 0.5

    if label_name == "Effusion" and norm_y >= 0.42:
        if is_bilateral:
            return "Bilateral costophrenic angles and lower pleural spaces"
        return f"{side_str} costophrenic angle and lower pleural space"

    if label_name == "Pneumothorax":
        if norm_y <= 0.35:
            return f"{side_str} apical pleural space"
        if norm_y <= 0.65:
            return f"{side_str} apicolateral pleural space"
        return f"{side_str} lateral and basilar pleural space"

    if norm_y <= 0.30:
        zone_str = "upper lung zone (apical)"
    elif norm_y <= 0.62:
        zone_str = "mid lung zone"
    else:
        if is_bilateral and label_name in ("Atelectasis", "Consolidation", "Infiltration", "Lung Opacity"):
            return "Bibasilar lower lung zones"
        zone_str = "lower lung zone / lung base"

    return f"{side_str} {zone_str}"


def render_cam_heatmap_overlay(
    cam_grid: np.ndarray,
    pad_meta: dict[str, Any],
) -> Image.Image:
    """
    Upsamples the NxN CAM to the letterboxed square (max_dim, max_dim), slices out
    letterbox padding for zero spatial shift, and blends a thermal colormap over the
    16-bit-normalized radiograph display image.
    """
    gray_u8 = pad_meta["display_uint8"]
    base_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    h, w = gray_u8.shape
    max_dim = pad_meta["max_dim"]
    pad_top = pad_meta["pad_top"]
    pad_left = pad_meta["pad_left"]

    grid = _mask_corner_borders(cam_grid)
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
            dtype=np.float32,
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
    is_localized: bool,
) -> tuple[float, float]:
    """Baseline-verified soft-max pooling across TTA views."""
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


def _fuse_multi_res_cam(
    cam_224: np.ndarray,
    weight_224: float,
    cam_512: np.ndarray,
    weight_512: float,
    is_localized: bool,
) -> np.ndarray:
    """
    Fuses DenseNet 7x7 CAM and ResNet50 16x16 CAM into a unified 16x16 normalized map
    using explicitly tracked per-backbone weights (fixes cam_w_sum slice bug).
    """
    components: list[np.ndarray] = []
    comp_weights: list[float] = []

    if weight_224 > 0.0:
        avg_224 = np.maximum(cam_224 / weight_224, 0.0)
        peak_224 = float(avg_224.max())
        if peak_224 > 1e-6:
            avg_224 = avg_224 / peak_224
        t_224 = torch.from_numpy(avg_224).unsqueeze(0).unsqueeze(0).to(torch.float32)
        up_16 = F.interpolate(t_224, size=(16, 16), mode="bilinear", align_corners=False)[0, 0].numpy()
        components.append(up_16)
        comp_weights.append(0.45 if is_localized else 0.65)

    if weight_512 > 0.0:
        avg_512 = np.maximum(cam_512 / weight_512, 0.0)
        peak_512 = float(avg_512.max())
        if peak_512 > 1e-6:
            avg_512 = avg_512 / peak_512
        components.append(avg_512)
        comp_weights.append(0.55 if is_localized else 0.35)

    if not components:
        return np.zeros((16, 16), dtype=np.float32)

    total_cw = sum(comp_weights)
    fused = np.zeros((16, 16), dtype=np.float32)
    for comp, cw in zip(components, comp_weights):
        fused += comp * (cw / total_cw)
    return fused


def evaluate_single_image(
    image: Image.Image,
    ensemble_224: list[dict[str, Any]],
    resnet_512: dict[str, Any] | None,
    sensitivity_offset: float = 0.0,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """
    Runs the 4-model ensemble + multi-resolution 16x16 CAM + Dual-Gate + Quorum audit
    on a single PIL image.
    """
    tta_224, tta_512, pad_meta = prepare_multiscale_tta(image, device_str=device_str)

    agg_cal: dict[str, list[float]] = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_raw: dict[str, list[float]] = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_weights: dict[str, list[float]] = {label: [] for label in CLASS_CALIBRATION_GATES}

    cam_accum_224 = {label: np.zeros((7, 7), dtype=np.float32) for label in CLASS_CALIBRATION_GATES}
    cam_weights_224 = {label: 0.0 for label in CLASS_CALIBRATION_GATES}
    cam_accum_512 = {label: np.zeros((16, 16), dtype=np.float32) for label in CLASS_CALIBRATION_GATES}
    cam_weights_512 = {label: 0.0 for label in CLASS_CALIBRATION_GATES}

    with torch.no_grad():
        # 1. 224x224 DenseNet121 specialists
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
                if np.isnan(cal_probs_batch[:, idx]).any() or np.isnan(raw_probs_batch[:, idx]).any():
                    continue

                is_loc = label in LOCALIZED_FINE_LABELS
                c_val, r_val = pool_tta_views(cal_probs_batch[:, idx], raw_probs_batch[:, idx], is_loc)

                agg_cal[label].append(c_val)
                agg_raw[label].append(r_val)
                agg_weights[label].append(vote_w)

                cls_w = m.classifier.weight[idx]
                cam_2d = torch.matmul(cls_w, full_coord_feat_7x7.reshape(1024, -1)).reshape(7, 7).cpu().numpy()
                cam_accum_224[label] += cam_2d * vote_w
                cam_weights_224[label] += vote_w

        # 2. 512x512 ResNet50 specialist (single pass: raw logits + 16x16 layer4 CAM)
        if resnet_512 is not None:
            r_model = resnet_512["model"].model
            r_op_t = resnet_512["op_threshs"]
            r_paths = resnet_512["pathologies"]
            base_w = resnet_512["vote_weight"]

            f = r_model.conv1(tta_512)
            f = r_model.bn1(f)
            f = r_model.relu(f)
            f = r_model.maxpool(f)
            f = r_model.layer1(f)
            f = r_model.layer2(f)
            f = r_model.layer3(f)
            f_layer4 = r_model.layer4(f)  # [2, 2048, 16, 16]
            pooled_512 = torch.flatten(r_model.avgpool(f_layer4), 1)
            logits_512 = r_model.fc(pooled_512)

            raw_512 = torch.sigmoid(logits_512).cpu().numpy()
            cal_512 = compute_op_norm(raw_512, r_op_t)
            full_coord_feat_16x16 = f_layer4[0]  # View 0 full-anatomy letterbox coordinates

            for idx, label in enumerate(r_paths):
                if not label or label not in CLASS_CALIBRATION_GATES:
                    continue
                if np.isnan(cal_512[:, idx]).any() or np.isnan(raw_512[:, idx]).any():
                    continue

                is_loc = label in LOCALIZED_FINE_LABELS
                c_val, r_val = pool_tta_views(cal_512[:, idx], raw_512[:, idx], is_loc)
                eff_w = base_w * 1.25 if is_loc else base_w

                agg_cal[label].append(c_val)
                agg_raw[label].append(r_val)
                agg_weights[label].append(eff_w)

                r_cls_w = r_model.fc.weight[idx]
                cam_16 = torch.matmul(
                    r_cls_w, full_coord_feat_16x16.reshape(2048, -1)
                ).reshape(16, 16).cpu().numpy()
                cam_accum_512[label] += cam_16 * eff_w
                cam_weights_512[label] += eff_w

    detected_findings: list[dict[str, Any]] = []
    subthreshold_findings: list[dict[str, Any]] = []
    telemetry_rows: list[list[str]] = []
    composite_cam = np.zeros((16, 16), dtype=np.float32)

    for label, gate in CLASS_CALIBRATION_GATES.items():
        weights = agg_weights[label]
        if not weights or sum(weights) <= 0:
            continue

        c_list = agg_cal[label]
        r_list = agg_raw[label]
        mean_cal = float(np.average(c_list, weights=weights))
        mean_raw = float(np.average(r_list, weights=weights))

        is_loc = label in LOCALIZED_FINE_LABELS
        fused_cam = _fuse_multi_res_cam(
            cam_accum_224[label], cam_weights_224[label],
            cam_accum_512[label], cam_weights_512[label],
            is_localized=is_loc,
        )

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
        location = localize_from_cam_grid(fused_cam, label)

        telemetry_rows.append([
            gate["clinical_name"].title(),
            f"{mean_cal:.1%}",
            f"{mean_raw:.1%}",
            "DETECTED" if passed_gate else "Below Gate",
            location if passed_gate else "-",
        ])

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
            "passed_gate": passed_gate,
        }

        if passed_gate:
            detected_findings.append(record)
            composite_cam += fused_cam * mean_cal
        else:
            subthreshold_findings.append(record)

    telemetry_rows.sort(key=lambda r: float(r[1].rstrip("%")), reverse=True)
    detected_findings.sort(key=lambda d: d["cal_prob"], reverse=True)
    subthreshold_findings.sort(key=lambda d: d["cal_prob"], reverse=True)

    # Hierarchical redundancy suppression
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
    overlay_img = (
        render_cam_heatmap_overlay(composite_cam, pad_meta)
        if top_findings
        else Image.fromarray(pad_meta["display_uint8"], mode="L").convert("RGB")
    )

    return {
        "top_findings": top_findings,
        "near_misses": subthreshold_findings[:3],
        "telemetry_rows": telemetry_rows,
        "overlay_img": overlay_img,
        "composite_cam": composite_cam,
        "pad_meta": pad_meta,
    }
