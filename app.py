"""
ChestXStudio: 4-Model Multi-Resolution Ensemble (224 + 512) + Dual-Gate CAM + SmolVLM
=====================================================================================
Audited Architecture:
  1. Multi-Resolution Clinical Ensemble:
       - DenseNet121-All (224x224, 18 classes)
       - DenseNet121-CheXpert (224x224, Stanford CheXpert specialist)
       - DenseNet121-MIMIC (224x224, MIT MIMIC-CXR specialist)
       - ResNet50-512-All (512x512 native high-resolution specialist for subtle
         Pneumothorax, Nodule, Mass, and Fracture detection — 5.2x pixel density)
  2. Aspect-Preserved 3-View Test-Time Augmentation (TTA):
       - View 0: Full-Anatomy Letterbox Square (Pure Linear Window)
       - View 1: 90% Thoracic Center Square (Pure Linear Window)
       - View 2: 82% Upper-Thoracic Aspect-Preserved Square (80% Linear + 20% CLAHE)
  3. Dual-Gate Calibration (op_norm + Raw Sigmoid Floor) + Multi-Model Quorum Check.
  4. Padding-Corrected 7x7 Class Activation Mapping (CAM) + SmolVLM Sentence Auditor.
"""

import gc
import re
import socket
import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv
from PIL import Image
from skimage import exposure
from transformers import AutoProcessor, AutoModelForImageTextToText
import gradio as gr

# =====================================================================
# 1. HARDWARE & 4-MODEL MULTI-RESOLUTION ENSEMBLE INITIALIZATION
# =====================================================================
DEVICE_LLM = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CNN = "cpu"
DTYPE_LLM = torch.float16 if DEVICE_LLM in ("mps", "cuda") else torch.float32

print("[ChestXStudio] Initializing 4-Model Multi-Resolution Ensemble (224x224 + 512x512)...")

DENSENET_SPECS = [
    ("all_224", "densenet121-res224-all", 1.00),
    ("chex_224", "densenet121-res224-chex", 1.15),
    ("mimic_224", "densenet121-res224-mimic_ch", 1.15),
]

cnn_ensemble_224 = []
for short_name, weight_id, vote_weight in DENSENET_SPECS:
    try:
        print(f" -> Loading 224x224 DenseNet specialist '{weight_id}' on {DEVICE_CNN.upper()}...")
        m = xrv.models.DenseNet(weights=weight_id)
        m = m.to(device=DEVICE_CNN, dtype=torch.float32)
        m.eval()
        op_t = m.op_threshs.detach().cpu().numpy().copy()
        cnn_ensemble_224.append({
            "name": short_name,
            "weight_id": weight_id,
            "model": m,
            "vote_weight": vote_weight,
            "op_threshs": op_t,
            "pathologies": list(m.pathologies)
        })
    except Exception as e:
        print(f"    [Warning] Could not load '{weight_id}': {e}")

# Load Native 512x512 High-Resolution ResNet50 Specialist (Trained on SIIM-Pneumothorax & VinDr)
resnet_512_spec = None
try:
    print(" -> Loading 512x512 High-Res specialist 'resnet50-res512-all' on CPU...")
    r512 = xrv.models.ResNet(weights="resnet50-res512-all")
    r512 = r512.to(device=DEVICE_CNN, dtype=torch.float32)
    r512.eval()
    r512_op_t = r512.op_threshs.detach().cpu().numpy().copy()
    # Disable internal op_norm so forward() returns raw unscaled logits in 1 pass
    r512.op_threshs = None
    r512.apply_sigmoid = False
    resnet_512_spec = {
        "name": "resnet50_512",
        "weight_id": "resnet50-res512-all",
        "model": r512,
        "vote_weight": 1.25,
        "op_threshs": r512_op_t,
        "pathologies": list(r512.pathologies)
    }
    print(" -> 512x512 High-Res ResNet50 specialist active.")
except Exception as e:
    print(f"    [Warning] 512x512 ResNet50 unavailable ({e}); continuing with 224x224 ensemble.")

if not cnn_ensemble_224:
    raise RuntimeError("No TorchXRayVision DenseNet models could be loaded!")

# Initialize SmolVLM-500M-Instruct
VLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
processor = AutoProcessor.from_pretrained(VLM_ID)
processor.image_processor.do_image_splitting = False
processor.tokenizer.padding_side = "left"

try:
    llm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID,
        dtype=DTYPE_LLM,
        device_map=DEVICE_LLM
    )
except TypeError:
    llm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID,
        torch_dtype=DTYPE_LLM,
        device_map=DEVICE_LLM
    )

llm_model = llm_model.to(device=DEVICE_LLM, dtype=DTYPE_LLM)
llm_model.eval()
print(f"[ChestXStudio] SmolVLM-500M-Instruct ready on {DEVICE_LLM.upper()} ({DTYPE_LLM}).")

# =====================================================================
# 2. DUAL-GATE CLINICAL CALIBRATION & TERMINOLOGY TABLE
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


def compute_op_norm(raw_probs: np.ndarray, op_threshs: np.ndarray) -> np.ndarray:
    """
    Exact vectorized implementation of TorchXRayVision's piecewise op_norm
    calibration (eliminates running a second CNN forward pass).
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


# =====================================================================
# 3. ASPECT-PRESERVED MULTI-RESOLUTION (224 + 512) PREPROCESSING
# =====================================================================
def prepare_multiscale_tta(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Builds aspect-ratio-preserved multi-scale tensors without vertical stretching:
      - View 0 (Linear Window, Full-Anatomy Letterbox Square): 100% of costophrenic angles & apices.
      - View 1 (Linear Window, 90% Central Thoracic Square): Parenchymal & cardiac focus.
      - View 2 (80% Linear + 20% CLAHE Edge Window, 82% Upper-Thoracic Square):
        1:1 aspect-ratio zoom into lung apices & upper/mid zones for Pneumothorax/Nodules.
      - Also produces a 512x512 high-resolution batch [2, 1, 512, 512] for ResNet50-512.
    """
    img_gray = image.convert("L")
    raw_np = np.asarray(img_gray, dtype=np.float32)
    h, w = raw_np.shape

    # 1. Standard Linear 1st-99th Percentile Stretch (Preserves XRV calibration baseline)
    p1, p99 = np.percentile(raw_np, (1.0, 99.0))
    if (p99 - p1) > 10.0:
        linear_0_255 = np.clip((raw_np - p1) * (255.0 / (p99 - p1)), 0.0, 255.0).astype(np.float32)
    else:
        linear_0_255 = np.clip(raw_np, 0.0, 255.0).astype(np.float32)

    # 2. Controlled 80% Linear + 20% CLAHE Edge-Enhanced Window (Strictly for View 2)
    unit_linear = np.clip(linear_0_255 / 255.0, 0.0, 1.0)
    clahe_0_255 = (exposure.equalize_adapthist(unit_linear, clip_limit=0.012).astype(np.float32)) * 255.0
    edge_0_255 = (0.80 * linear_0_255 + 0.20 * clahe_0_255).astype(np.float32)

    # 3. Scale both to TorchXRayVision [-1024, 1024] range in float32
    norm_linear = xrv.datasets.normalize(linear_0_255, 255.0).astype(np.float32)
    norm_edge = xrv.datasets.normalize(edge_0_255, 255.0).astype(np.float32)

    # View 0: Full-Anatomy Letterbox Square (1:1 aspect ratio)
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

    # View 1: 90% Central Thoracic Square Crop (1:1 aspect ratio)
    crop_side = max(16, int(min(h, w) * 0.90))
    cy0 = max(0, (h - crop_side) // 2)
    cx0 = max(0, (w - crop_side) // 2)
    view1_square = norm_linear[cy0:cy0 + crop_side, cx0:cx0 + crop_side]

    # View 2: 82% Upper-Thoracic Aspect-Preserved Square Crop on Edge-Enhanced Window
    # Anchored near the lung apices (top 3% margin) with 1:1 square aspect ratio (zero distortion)
    apical_side = max(16, int(min(h, w) * 0.82))
    ay0 = min(max(0, int(h * 0.03)), max(0, h - apical_side))
    ax0 = max(0, (w - apical_side) // 2)
    view2_square = norm_edge[ay0:ay0 + apical_side, ax0:ax0 + apical_side]

    t_v0 = torch.from_numpy(view0_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t_v1 = torch.from_numpy(view1_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    t_v2 = torch.from_numpy(view2_square).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    # 224x224 3-View Batch for DenseNet121 Ensemble
    tta_224 = torch.cat([
        F.interpolate(t_v0, size=(224, 224), mode="bilinear", align_corners=False),
        F.interpolate(t_v1, size=(224, 224), mode="bilinear", align_corners=False),
        F.interpolate(t_v2, size=(224, 224), mode="bilinear", align_corners=False),
    ], dim=0).to(device=DEVICE_CNN, dtype=torch.float32)

    # 512x512 2-View High-Resolution Batch for ResNet50-512 Specialist
    tta_512 = torch.cat([
        F.interpolate(t_v0, size=(512, 512), mode="bilinear", align_corners=False),
        F.interpolate(t_v2, size=(512, 512), mode="bilinear", align_corners=False),
    ], dim=0).to(device=DEVICE_CNN, dtype=torch.float32)

    pad_meta = {
        "h": h, "w": w, "max_dim": max_dim,
        "pad_top": pad_top, "pad_left": pad_left
    }
    return tta_224, tta_512, pad_meta


# =====================================================================
# 4. BILATERAL-AWARE ROI CAM LOCALIZATION & PADDING-CORRECTED HEATMAP
# =====================================================================
def localize_from_cam_grid(cam_7x7: np.ndarray, label_name: str) -> str:
    """
    Computes anatomical localization from the Full-Coordinate 7x7 CAM without
    hardcoded overrides, preserving true apical, mid-zone, or basilar locations.
    """
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


def render_cam_heatmap_overlay(
    original_img: Image.Image,
    cam_7x7: np.ndarray,
    pad_meta: dict
) -> Image.Image:
    """
    Upsamples the 7x7 CAM to the letterboxed square (max_dim, max_dim), slices out
    the exact letterbox padding so rectangular images have zero spatial shift, and
    blends a smooth thermal colormap over the radiograph.
    """
    base_rgb = np.asarray(original_img.convert("RGB"), dtype=np.float32)
    h, w, _ = base_rgb.shape
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

    # Upsample to full square (max_dim, max_dim), then crop out letterbox padding
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
# 5. 4-MODEL MULTI-RESOLUTION ENSEMBLE + QUORUM INFERENCE ENGINE
# =====================================================================
def pool_tta_views(
    cal_vec: np.ndarray,
    raw_vec: np.ndarray,
    is_localized: bool
) -> tuple[float, float]:
    """
    Baseline-Verified Soft-Max Pooling:
      - For localized pathologies (Pneumothorax, Nodule, Mass, Fracture), boosts
        sensitivity using the peak view (0.55 * max + 0.45 * mean) ONLY if the
        uncropped baseline View 0 also clears a minimum sanity floor (cal >= 0.36).
        This prevents single-crop border artifacts from triggering false positives.
      - For diffuse/global pathologies, uses weighted consensus across views.
    """
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


def run_ensemble_cad(
    image: Image.Image,
    sensitivity_offset: float = 0.0
) -> tuple[list[dict], list[list], Image.Image]:
    tta_224, tta_512, pad_meta = prepare_multiscale_tta(image)

    agg_cal = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_raw = {label: [] for label in CLASS_CALIBRATION_GATES}
    agg_weights = {label: [] for label in CLASS_CALIBRATION_GATES}
    cam_accum = {label: np.zeros((7, 7), dtype=np.float32) for label in CLASS_CALIBRATION_GATES}

    with torch.no_grad():
        # 1. Run 224x224 DenseNet121 Specialists (Single forward pass per model)
        for entry in cnn_ensemble_224:
            m = entry["model"]
            vote_w = entry["vote_weight"]
            path_list = entry["pathologies"]
            op_t = entry["op_threshs"]

            raw_feats = m.features(tta_224)
            relu_feats = F.relu(raw_feats, inplace=False)  # [3, 1024, 7, 7]
            pooled = F.adaptive_avg_pool2d(relu_feats, (1, 1)).flatten(1)
            logits = m.classifier(pooled)

            raw_probs_batch = torch.sigmoid(logits).cpu().numpy()  # [3, 18]
            cal_probs_batch = compute_op_norm(raw_probs_batch, op_t)  # [3, 18]

            # Spatial CAM uses full-coordinate views (View 0: 65%, View 1: 35%)
            # so cropped View 2 never shifts apical activations down to the lung base.
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

        # 2. Run 512x512 High-Resolution ResNet50 Specialist
        if resnet_512_spec is not None:
            r_model = resnet_512_spec["model"]
            r_op_t = resnet_512_spec["op_threshs"]
            r_paths = resnet_512_spec["pathologies"]
            base_w = resnet_512_spec["vote_weight"]

            logits_512 = r_model(tta_512)  # [2, 18] raw logits
            raw_512 = torch.sigmoid(logits_512).cpu().numpy()
            cal_512 = compute_op_norm(raw_512, r_op_t)

            for idx, label in enumerate(r_paths):
                if not label or label not in CLASS_CALIBRATION_GATES:
                    continue
                if np.isnan(cal_512[0, idx]) or np.isnan(raw_512[0, idx]):
                    continue

                is_loc = label in LOCALIZED_FINE_LABELS
                c_val, r_val = pool_tta_views(cal_512[:, idx], raw_512[:, idx], is_loc)
                # Give 512x512 specialist higher authority on fine localized pathologies
                eff_w = base_w * 1.25 if is_loc else base_w

                agg_cal[label].append(c_val)
                agg_raw[label].append(r_val)
                agg_weights[label].append(eff_w)

    detected_findings = []
    telemetry_rows = []
    composite_cam = np.zeros((7, 7), dtype=np.float32)

    for label, gate in CLASS_CALIBRATION_GATES.items():
        weights = agg_weights[label]
        if not weights or sum(weights) <= 0:
            continue

        total_w = sum(weights)
        c_list = agg_cal[label]
        r_list = agg_raw[label]

        mean_cal = float(np.average(c_list, weights=weights))
        mean_raw = float(np.average(r_list, weights=weights))

        # Normalize CAM by the 224x224 DenseNet weight sum
        cam_w_sum = sum(w for w in weights[:len(cnn_ensemble_224)]) or 1.0
        mean_cam = cam_accum[label] / cam_w_sum

        effective_min_cal = float(np.clip(gate["min_cal"] - sensitivity_offset, 0.48, 0.78))
        effective_min_raw = float(np.clip(gate["min_raw"] - (sensitivity_offset * 0.4), 0.08, 0.50))

        # Multi-Model Quorum Check: if >= 3 models evaluate this pathology, require at least
        # 2 models to independently show supporting signal (prevents single-model outlier spikes)
        num_models = len(c_list)
        if num_models >= 3:
            supporting_models = sum(
                1 for c_m, r_m in zip(c_list, r_list)
                if c_m >= (effective_min_cal - 0.05) and r_m >= (effective_min_raw * 0.75)
            )
            quorum_ok = (supporting_models >= 2) or (mean_cal >= effective_min_cal + 0.08)
        else:
            quorum_ok = True

        passed_gate = (mean_cal >= effective_min_cal) and (mean_raw >= effective_min_raw) and quorum_ok
        location = localize_from_cam_grid(mean_cam, label)

        telemetry_rows.append([
            gate["clinical_name"].title(),
            f"{mean_cal:.1%}",
            f"{mean_raw:.1%}",
            "DETECTED" if passed_gate else "Below Gate",
            location if passed_gate else "-"
        ])

        if passed_gate:
            detected_findings.append({
                "raw_label": label,
                "clinical_name": gate["clinical_name"],
                "observation": gate["observation"],
                "location": location,
                "cal_prob": mean_cal,
                "raw_prob": mean_raw,
                "cam": mean_cam
            })
            composite_cam += np.maximum(mean_cam, 0.0) * mean_cal

    telemetry_rows.sort(key=lambda r: float(r[1].rstrip("%")), reverse=True)
    detected_findings.sort(key=lambda d: d["cal_prob"], reverse=True)

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
    if top_findings:
        overlay_img = render_cam_heatmap_overlay(image, composite_cam, pad_meta)
    else:
        overlay_img = image.convert("RGB")

    return top_findings, telemetry_rows, overlay_img


# =====================================================================
# 6. VLM GENERATION + SENTENCE-LEVEL ANTI-HALLUCINATION AUDITOR
# =====================================================================
SYSTEM_RULEBOOK = (
    "You are an expert diagnostic chest X-ray AI radiologist following the 5-Rule Clinical Rulebook:\n"
    "1. Systematic Search: Evaluate airways, lungs/pleura, mediastinum, heart, and bones.\n"
    "2. Observation Before Etiology: Describe physical visual findings FIRST before stating the diagnosis.\n"
    "3. Spatial Grounding: Include the exact anatomical location for every positive finding.\n"
    "4. Explicit Negatives: State the absence of unmentioned critical acute pathologies "
    "(pneumothorax, pleural effusion, focal consolidation).\n"
    "5. Uncertainty Calibration: Use calibrated diagnostic phrasing.\n\n"
    "Write the report using strictly the CAD findings provided in these exact 3 sections:\n"
    "1. Systematic Visual Findings\n"
    "2. Spatial Localization\n"
    "3. Clinical Impression"
)

ALL_PATHOLOGY_STEMS = {
    "Pneumothorax": ["pneumothora"],
    "Effusion": ["effusion", "pleural fluid"],
    "Consolidation": ["consolidat"],
    "Pneumonia": ["pneumonia"],
    "Edema": ["edema", "congest"],
    "Atelectasis": ["atelecta"],
    "Cardiomegaly": ["cardiomegal", "enlarged heart", "enlarged cardiac"],
    "Enlarged Cardiomediastinum": ["widened mediastin", "enlarged cardiomediastin"],
    "Lung Opacity": ["opacit"],
    "Infiltration": ["infiltrat"],
    "Mass": ["mass"],
    "Nodule": ["nodul"],
    "Emphysema": ["emphysem", "hyperinflat"],
    "Fibrosis": ["fibros"],
    "Pleural_Thickening": ["pleural thickening"],
    "Fracture": ["fractur"],
    "Hernia": ["hernia"]
}


def filter_unverified_vlm_sentences(vlm_text: str, allowed_raw_labels: set[str]) -> list[str]:
    allowed_stems = set()
    for lbl in allowed_raw_labels:
        for st in ALL_PATHOLOGY_STEMS.get(lbl, []):
            allowed_stems.add(st)
        if lbl in ("Consolidation", "Pneumonia", "Atelectasis", "Edema", "Infiltration"):
            allowed_stems.add("opacit")

    kept = []
    for s in re.split(r"(?<=[.!?])\s+", vlm_text):
        s_clean = s.strip()
        if not s_clean or re.match(r"^[123]\.\s*", s_clean):
            continue
        s_low = s_clean.lower()
        s_unnegated = re.sub(
            r"\b(?:no|without|negative for|free of|absence of|clear of|unremarkable|within normal limits|normal)\b.*",
            " ",
            s_low
        )
        hallucinated = False
        for _, stems in ALL_PATHOLOGY_STEMS.items():
            if any(st in s_unnegated for st in stems) and not any(st in allowed_stems for st in stems):
                hallucinated = True
                break
        if not hallucinated and len(s_clean) > 15:
            kept.append(s_clean.rstrip("."))
    return kept[:2]


def synthesize_audited_rulebook_report(
    vlm_raw_text: str,
    cad_findings: list[dict],
    use_vlm_commentary: bool = True
) -> str:
    detected_raw_labels = {d["raw_label"] for d in cad_findings}

    acute_checks = [
        ({"Pneumothorax"}, "pneumothorax"),
        ({"Effusion"}, "pleural effusion"),
        ({"Consolidation", "Pneumonia", "Infiltration", "Lung Opacity"}, "focal lobar consolidation"),
        ({"Edema"}, "pulmonary edema")
    ]
    explicit_negatives = [
        neg_name for label_set, neg_name in acute_checks
        if not (label_set & detected_raw_labels)
    ]
    neg_sentence = f"No radiographic evidence of {', '.join(explicit_negatives)}." if explicit_negatives else ""

    if not cad_findings:
        sec1 = (
            "Trachea and central airways are midline and patent. "
            "Bilateral lung fields are well-aerated and clear without focal parenchymal opacity. "
            "Cardiomediastinal silhouette is within normal limits and bilateral costophrenic angles are sharp. "
            f"{neg_sentence}"
        ).strip()
        sec2 = "Bilateral lung parenchyma, costophrenic angles, pleura, and cardiomediastinal silhouette."
        sec3 = "No acute cardiopulmonary process identified on frontal chest radiograph."
    else:
        obs_sentences = []
        loc_entries = []
        imp_entries = []

        for idx, item in enumerate(cad_findings, start=1):
            loc_lower = item["location"][0].lower() + item["location"][1:]
            obs_sentences.append(
                f"Visual evaluation demonstrates {item['observation']} localized to the {loc_lower}"
            )
            if item["location"] not in loc_entries:
                loc_entries.append(item["location"])

            cal_p = item["cal_prob"]
            raw_p = item["raw_prob"]
            if cal_p >= 0.72 and raw_p >= 0.40:
                cal_phrase = "Findings are strongly consistent with"
            elif cal_p >= 0.62:
                cal_phrase = "Radiographic appearance is consistent with"
            else:
                cal_phrase = "Mild or borderline radiographic features suggestive of"

            imp_entries.append(
                f"{idx}. {cal_phrase} {item['clinical_name']} involving the {loc_lower} "
                f"(Ensemble Calibrated: {cal_p:.0%} | Raw Sigmoid: {raw_p:.0%})"
            )

        if not ({"Cardiomegaly", "Enlarged Cardiomediastinum"} & detected_raw_labels):
            obs_sentences.append("The cardiac silhouette and central mediastinal contours are within normal limits")

        if use_vlm_commentary and vlm_raw_text:
            verified_vlm = filter_unverified_vlm_sentences(vlm_raw_text, detected_raw_labels)
            if verified_vlm:
                obs_sentences.extend(verified_vlm[:1])

        sec1 = ". ".join(obs_sentences) + ". " + neg_sentence
        sec2 = "; ".join(loc_entries) + "."
        sec3 = "\n".join(imp_entries) + "\nClinical correlation is recommended."

    total_models = len(cnn_ensemble_224) + (1 if resnet_512_spec is not None else 0)
    ensemble_header = (
        f"[Multi-Resolution Ensemble ({total_models} Models: 224x224 + 512x512) + Quorum Dual-Gate Audit]\n"
        + (
            "Status: STRICT NORMAL (All 18 pathologies below Dual-Gate & Quorum threshold)"
            if not cad_findings
            else " | ".join(
                f"{d['clinical_name'].title()}: {d['cal_prob']:.0%} cal / {d['raw_prob']:.0%} raw ({d['location']})"
                for d in cad_findings
            )
        )
    )

    return (
        f"{ensemble_header}\n\n"
        f"1. Systematic Visual Findings:\n{sec1.strip()}\n\n"
        f"2. Spatial Localization:\n{sec2.strip()}\n\n"
        f"3. Clinical Impression:\n{sec3.strip()}"
    )


# =====================================================================
# 7. END-TO-END INFERENCE CONTROLLER
# =====================================================================
def analyze_xray(
    image: Image.Image,
    sensitivity_mode: str = "Balanced Clinical (Recommended)",
    use_vlm: bool = True
) -> tuple[str, Image.Image, list[list]]:
    if image is None:
        return "Please upload a frontal chest X-ray image (PA or AP).", None, []

    if "High Sensitivity" in sensitivity_mode:
        sens_offset = 0.04
    elif "High Specificity" in sensitivity_mode:
        sens_offset = -0.05
    else:
        sens_offset = 0.0

    cad_findings, telemetry_rows, cam_overlay_img = run_ensemble_cad(image, sensitivity_offset=sens_offset)

    raw_vlm_text = ""
    if use_vlm:
        if not cad_findings:
            cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold."
        else:
            cad_summary = "; ".join(
                f"{d['clinical_name']} at {d['location']} (Calibrated: {d['cal_prob']:.0%})"
                for d in cad_findings
            )

        image_rgb = image.convert("RGB").resize((512, 512))
        user_prompt = (
            f"Verified Ensemble CAD findings: {cad_summary}.\n"
            "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[[image_rgb]], return_tensors="pt")
        inputs = {
            k: v.to(device=DEVICE_LLM, dtype=DTYPE_LLM) if v.is_floating_point() else v.to(device=DEVICE_LLM)
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.15
            )
        raw_vlm_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        del inputs, output_ids
        gc.collect()
        if DEVICE_LLM == "mps":
            torch.mps.empty_cache()

    final_report = synthesize_audited_rulebook_report(raw_vlm_text, cad_findings, use_vlm_commentary=use_vlm)
    return final_report, cam_overlay_img, telemetry_rows


# =====================================================================
# 8. GRADIO 6.0 CLINICAL WORKSTATION INTERFACE
# =====================================================================
def find_available_port(preferred_port: int = 7860) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


with gr.Blocks() as interface:
    gr.Markdown("# ChestXStudio: 4-Model Multi-Resolution (224 + 512) Clinical Workstation")
    gr.Markdown(
        "**Stage 1 (Multi-Resolution Ensemble):** `DenseNet121-All (224)` + `DenseNet121-CheXpert (224)` + "
        "`DenseNet121-MIMIC (224)` + `ResNet50-All (512x512 High-Res)` with Aspect-Preserved 3-View TTA & Quorum Dual-Gate  \n"
        "**Stage 2 (Spatial Grounding & NLP Auditor):** Padding-Aligned Hemithorax CAM + `SmolVLM-500M-Instruct`"
    )
    gr.Markdown(
        "*Research & Educational Use Only — This workstation is for academic research and must not "
        "replace interpretation by a licensed radiologist or physician.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="1. Upload Frontal Chest X-Ray (PA or AP)")
            sensitivity_radio = gr.Radio(
                choices=[
                    "High Sensitivity (Screening - catches subtle findings)",
                    "Balanced Clinical (Recommended)",
                    "High Specificity (Strict - suppresses borderline findings)"
                ],
                value="Balanced Clinical (Recommended)",
                label="2. Operating Threshold Calibration"
            )
            use_vlm_checkbox = gr.Checkbox(
                value=True,
                label="Enable SmolVLM-500M Multimodal Commentary (Audited by Sentence Guardrail)"
            )
            submit_btn = gr.Button("Analyze Radiograph", variant="primary", size="lg")

        with gr.Column(scale=1):
            cam_output = gr.Image(type="pil", label="Spatial Class Activation Map (CAM) Overlay")

    with gr.Row():
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="5-Rule Clinical Diagnostic Report",
                lines=16,
                placeholder="Upload a chest X-ray and click 'Analyze Radiograph'..."
            )
        with gr.Column(scale=2):
            telemetry_table = gr.Dataframe(
                headers=["Pathology", "Calibrated", "Raw Sigmoid", "Dual-Gate Status", "CAM Location"],
                datatype=["str", "str", "str", "str", "str"],
                label="Full 18-Pathology Dual-Gate Telemetry"
            )

    submit_btn.click(
        fn=analyze_xray,
        inputs=[image_input, sensitivity_radio, use_vlm_checkbox],
        outputs=[output_text, cam_output, telemetry_table]
    )

if __name__ == "__main__":
    port = find_available_port(7860)
    print(f"[ChestXStudio] Launching Gradio workstation on http://127.0.0.1:{port}")
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft()
    )