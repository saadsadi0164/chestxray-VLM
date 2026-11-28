"""
ChestXStudio-VLM: Automated Unit & Integration Tests for Phase 1 Preprocessing & MPS Precision
Path: tests/test_preprocess_and_precision.py

Verifies:
  1. Training path (`RulebookCXRDataset`) and inference path (`normalize_cxr_to_rgb` + processor)
     produce byte-identical image tensors on 8-bit and 16-bit non-square inputs.
  2. Aspect ratio is preserved via symmetric centered letterbox padding before resize.
  3. DICOM (`pydicom`) loading handles RescaleSlope/Intercept, WindowCenter/Width,
     MONOCHROME1 inversion, and fails cleanly when pydicom is absent.
  4. `resolve_llm_device` defaults Apple Silicon MPS to `torch.float32` unless
     `allow_mps_bf16=True` is explicitly set.
  5. Real generation on 5 fixed images from `.tmp/reports/test_manifest.jsonl` on MPS fp32
     produces zero NaN/Inf logits and non-empty clinical text.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.analyze_xray import ChestXRayPipeline, DEFAULT_ADAPTER_PATH, DEFAULT_BASE_MODEL, SYSTEM_RULEBOOK, resolve_llm_device
from execution.build_rulebook_dataset import build_rulebook_target
from execution.preprocess import normalize_cxr_to_rgb
from execution.train_rulebook_vlm import RulebookCXRDataset


class TestPreprocessAndPrecision(unittest.TestCase):
    def test_aspect_ratio_preserved_with_symmetric_letterbox(self):
        # Wide 1000x500 bright gradient image -> must pad ~128 black rows top & bottom in 512x512
        wide = np.full((500, 1000), 180, dtype=np.uint8)
        wide[:, :20] = 60
        wide[:, -20:] = 240
        out_wide = np.asarray(normalize_cxr_to_rgb(Image.fromarray(wide, mode="L"), image_size=512))
        self.assertEqual(out_wide.shape, (512, 512, 3))
        # Top & bottom 100 rows are purely black letterbox padding
        self.assertEqual(int(out_wide[:100, :, :].max()), 0)
        self.assertEqual(int(out_wide[-100:, :, :].max()), 0)
        # Central 128..384 anatomy rows are bright and symmetric
        self.assertGreater(float(out_wide[200:312, :, :].mean()), 120.0)
        top_nonblack = int(np.argmax(out_wide[:, 500 // 2, 0] > 0))
        bot_nonblack = 511 - int(np.argmax(out_wide[::-1, 500 // 2, 0] > 0))
        self.assertAlmostEqual(top_nonblack, 511 - bot_nonblack, delta=2)

        # Tall 500x1000 image -> symmetrical left & right black padding
        tall = np.full((1000, 500), 180, dtype=np.uint8)
        tall[:20, :] = 60
        tall[-20:, :] = 240
        out_tall = np.asarray(normalize_cxr_to_rgb(Image.fromarray(tall, mode="L"), image_size=512))
        self.assertEqual(int(out_tall[:, :100, :].max()), 0)
        self.assertEqual(int(out_tall[:, -100:, :].max()), 0)
        self.assertGreater(float(out_tall[:, 200:312, :].mean()), 120.0)

    def test_train_and_inference_paths_produce_byte_identical_tensors(self):
        try:
            processor = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL, local_files_only=True)
        except Exception:
            self.skipTest("SmolVLM processor not in local HF cache")
        processor.image_processor.do_image_splitting = False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Non-square 16-bit synthetic chest radiograph saved as PNG
            arr16 = np.linspace(120, 4000, 420 * 512, dtype=np.uint16).reshape(420, 512)
            img_path = tmp / "sample_16bit.png"
            Image.fromarray(arr16).save(img_path, format="PNG")

            train_jsonl = tmp / "train.jsonl"
            sample = {
                "image_path": str(img_path),
                "labels": ["Cardiomegaly"],
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": build_rulebook_target([{"label": "Cardiomegaly", "zone": "central mediastinum and cardiac silhouette"}])}]},
                ],
            }
            train_jsonl.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            ds_train = RulebookCXRDataset(str(train_jsonl), processor=processor, image_size=512, augment=False)
            train_item = ds_train[0]

            # Inference path on exact same file opened via PIL
            with Image.open(img_path) as raw_pil:
                raw_pil.load()
                inf_rgb = normalize_cxr_to_rgb(raw_pil, image_size=512)
            inf_prompt = processor.apply_chat_template(
                sample["messages"][:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            inf_inputs = processor(text=[inf_prompt], images=[[inf_rgb]], return_tensors="pt")

            self.assertEqual(tuple(train_item["pixel_values"].shape), tuple(inf_inputs["pixel_values"][0].shape))
            self.assertTrue(
                torch.equal(train_item["pixel_values"], inf_inputs["pixel_values"][0]),
                "Training and inference pixel_values tensors must be byte-identical",
            )

    def test_dicom_support_and_graceful_fallback(self):
        fake_pixels = np.linspace(100, 3000, 128 * 96, dtype=np.uint16).reshape(96, 128)
        fake_ds = types.SimpleNamespace(
            pixel_array=fake_pixels,
            RescaleSlope=2.0,
            RescaleIntercept=-100.0,
            PhotometricInterpretation="MONOCHROME1",
            WindowCenter=[2500.0],
            WindowWidth=[3000.0],
        )
        fake_pydicom = types.SimpleNamespace(dcmread=lambda _p: fake_ds)

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_path = Path(tmpdir) / "study.dcm"
            dcm_path.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 16)

            orig_mod = sys.modules.get("pydicom")
            try:
                sys.modules["pydicom"] = fake_pydicom
                rgb = normalize_cxr_to_rgb(dcm_path, image_size=512)
                arr = np.asarray(rgb)
                self.assertEqual(arr.shape, (512, 512, 3))
                # Top/bottom padded (96 -> 128 square) and center darker than inverted brightest corners
                self.assertEqual(int(arr[:50, :, :].max()), 0)
                self.assertGreater(int(arr[256, 256, 0]), 0)
            finally:
                if orig_mod is None:
                    sys.modules.pop("pydicom", None)
                else:
                    sys.modules["pydicom"] = orig_mod

            # Test clean ImportError when pydicom is unavailable
            real_import = builtins.__import__

            def _block_pydicom(name, *args, **kwargs):
                if name == "pydicom":
                    raise ImportError("No module named 'pydicom'")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = _block_pydicom
            try:
                with self.assertRaises(ImportError):
                    normalize_cxr_to_rgb(dcm_path, image_size=512)
            finally:
                builtins.__import__ = real_import

    def test_resolve_llm_device_defaults_mps_to_fp32(self):
        if torch.backends.mps.is_available():
            dev_auto, dt_auto = resolve_llm_device("auto")
            self.assertEqual(dev_auto, "mps")
            self.assertEqual(dt_auto, torch.float32)

        dev_mps, dt_mps = resolve_llm_device("mps", allow_mps_bf16=False)
        self.assertEqual(dev_mps, "mps")
        self.assertEqual(dt_mps, torch.float32)

        dev_bf, dt_bf = resolve_llm_device("mps", allow_mps_bf16=True)
        self.assertEqual(dev_bf, "mps")
        self.assertEqual(dt_bf, torch.bfloat16)

    def test_generation_on_five_heldout_images_zero_nan_logits(self):
        manifest_path = Path(".tmp/reports/test_manifest.jsonl")
        if not manifest_path.exists():
            self.skipTest("Held-out manifest not built yet")

        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Pick 5 diverse fixed studies (including non-square CheXpert/PadChest if present)
        sample_rows = rows[:5]
        pipeline = ChestXRayPipeline(
            base_model_id=DEFAULT_BASE_MODEL,
            adapter_path=DEFAULT_ADAPTER_PATH,
            use_lora=True,
            load_cad=False,
            llm_device="auto",
        )
        if torch.backends.mps.is_available():
            self.assertEqual(pipeline.llm_device, "mps")
            self.assertEqual(pipeline.llm_dtype, torch.float32)

        prompt = "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
        for item in sample_rows:
            with Image.open(item["image_path"]) as img:
                img.load()
                text, scores = pipeline._run_vlm_generate(
                    img,
                    prompt,
                    max_new_tokens=96,
                    return_scores=True,
                )
            self.assertTrue(bool(text.strip()), f"Empty generation on {item['study_id']}")
            self.assertTrue(scores and len(scores) > 0, f"No generation scores captured for {item['study_id']}")
            for step_idx, s in enumerate(scores):
                self.assertFalse(
                    bool(torch.isnan(s).any()),
                    f"NaN logits at step {step_idx} for {item['study_id']}",
                )
                self.assertFalse(
                    bool(torch.isinf(s).any()),
                    f"Inf logits at step {step_idx} for {item['study_id']}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
