"""
ChestXStudio-VLM: Canonical Aspect-Preserving CXR Preprocessing
Path: execution/preprocess.py

Single source of truth for loading and normalizing chest radiographs across
training (`build_rulebook_dataset.py`, `train_rulebook_vlm.py`) and inference
(`analyze_xray.py`):
  - Supports PIL Image, 8-bit/16-bit PNG, JPEG, and DICOM (via optional `pydicom`
    with RescaleSlope/RescaleIntercept, MONOCHROME1 inversion, and windowing).
  - Applies uniform 1st-99th percentile grayscale contrast normalization on the
    native unpadded anatomy (Pillow 13 safe).
  - Preserves aspect ratio via centered black letterbox padding BEFORE bilinear
    resizing to `(image_size, image_size)` RGB so cardiothoracic ratio is never distorted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _first_numeric(val: Any) -> float | None:
    """Extracts a float from a scalar or pydicom MultiValue/sequence."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)) or hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
        try:
            if len(val) == 0:
                return None
            return float(val[0])
        except Exception:
            return None
    try:
        return float(val)
    except Exception:
        return None


def _is_dicom_file(path: Path) -> bool:
    """Checks extension or 128-byte preamble 'DICM' magic header."""
    if path.suffix.lower() in (".dcm", ".dicom"):
        return True
    try:
        with open(path, "rb") as f:
            header = f.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except OSError:
        return False


def _load_dicom_float32(path: Path) -> np.ndarray:
    """
    Loads a DICOM pixel array as float32 grayscale with optional Modality LUT
    (RescaleSlope/RescaleIntercept), WindowCenter/WindowWidth clipping, and
    MONOCHROME1 inversion.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM (.dcm/.dicom) files; "
            "install pydicom or convert the study to PNG/JPEG."
        ) from exc

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            arr = arr[0]

    slope = _first_numeric(getattr(ds, "RescaleSlope", None))
    intercept = _first_numeric(getattr(ds, "RescaleIntercept", None))
    if slope is not None or intercept is not None:
        arr = arr * (slope if slope is not None else 1.0) + (intercept if intercept is not None else 0.0)

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photometric == "MONOCHROME1":
        arr = float(arr.max()) + float(arr.min()) - arr

    wc = _first_numeric(getattr(ds, "WindowCenter", None))
    ww = _first_numeric(getattr(ds, "WindowWidth", None))
    if wc is not None and ww is not None and ww > 0:
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        arr = np.clip(arr, low, high)

    return arr.astype(np.float32)


def _extract_grayscale_2d(image_or_path: Image.Image | str | os.PathLike[str]) -> np.ndarray:
    """Loads any supported path or PIL Image into a 2D float32 grayscale array."""
    if isinstance(image_or_path, Image.Image):
        arr = np.asarray(image_or_path)
    elif isinstance(image_or_path, (str, os.PathLike)):
        p = Path(image_or_path)
        if _is_dicom_file(p):
            return _load_dicom_float32(p)
        try:
            with Image.open(p) as im:
                im.load()
                arr = np.asarray(im)
        except Exception:
            if p.suffix.lower() in (".dcm", ".dicom") or _is_dicom_file(p):
                return _load_dicom_float32(p)
            raise
    else:
        raise TypeError(f"Unsupported image input type: {type(image_or_path)!r}")

    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            r = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            b = arr[..., 2].astype(np.float32)
            return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        return arr[..., 0].astype(np.float32)
    raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")


def normalize_cxr_to_rgb(
    image_or_path: Image.Image | str | os.PathLike[str],
    image_size: int = 512,
) -> Image.Image:
    """
    Canonical ChestXStudio-VLM CXR preprocessor:
      1. Extracts native float32 2D grayscale (8-bit, 16-bit, RGB, or DICOM).
      2. Applies 1st-99th percentile contrast stretch BEFORE padding so black
         letterbox borders never skew percentile statistics.
      3. Centered letterbox-pads to a square canvas (black=0) preserving aspect ratio.
      4. Bilinear-resizes to `(image_size, image_size)` RGB (Pillow 13 safe).
    """
    arr_f = _extract_grayscale_2d(image_or_path)
    h, w = arr_f.shape[:2]

    p1, p99 = np.percentile(arr_f, [1.0, 99.0])
    if p99 > p1:
        arr_f = np.clip((arr_f - p1) / (p99 - p1), 0.0, 1.0) * 255.0
    else:
        vmin, vmax = float(arr_f.min()), float(arr_f.max())
        if vmax > vmin:
            arr_f = ((arr_f - vmin) / (vmax - vmin)) * 255.0
        else:
            arr_f = np.zeros_like(arr_f)

    norm_u8 = np.clip(np.round(arr_f), 0.0, 255.0).astype(np.uint8)

    if h != w:
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        square_u8 = np.pad(
            norm_u8,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        square_u8 = norm_u8

    rgb = Image.fromarray(square_u8, mode="L").convert("RGB")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), resample)
    return rgb
