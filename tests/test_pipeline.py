"""
ChestXStudio-VLM: Automated Audit & Regression Test Suite
Path: tests/test_pipeline.py

Verifies:
  1. Clause-level anti-hallucination guardrail (catches post-normal clauses & Normal-bias).
  2. Multi-resolution CAM math (explicit cam_w_sum, 16x16 fusion, laterality mirroring).
  3. Exact op_norm parity against torchxrayvision.models.op_norm.
  4. 16-bit grayscale extraction without uint8 clipping.
  5. Multi-dataset rulebook synthesis, bounding-box grounding, class balancing & validation.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.analyze_xray import filter_unverified_vlm_sentences, synthesize_audited_rulebook_report
from execution.build_rulebook_dataset import (
    balance_records_against_normal_bias,
    bbox_to_anatomical_zone,
    build_rulebook_target,
    canonicalize_label,
    validate_rulebook_compliance,
)
from execution.ensemble_cad import (
    _fuse_multi_res_cam,
    compute_op_norm,
    extract_grayscale_float32,
    localize_from_cam_grid,
)


class TestAntiHallucinationGuardrail(unittest.TestCase):
    def test_catches_hallucination_after_normal_clause(self):
        # Previously, regex `normal.*` wiped the rest of the sentence and let "pneumothorax" pass!
        vlm = "The heart is normal, and there is a large right pneumothorax."
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels=set())
        self.assertEqual(kept, [], "Hallucinated pathology after normal clause must be blocked")

    def test_allows_verified_pathology_with_explicit_negative_in_other_clause(self):
        vlm = "There is cardiomegaly with enlarged cardiac silhouette, and no pleural effusion is present."
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels={"Cardiomegaly"})
        self.assertEqual(len(kept), 1)

    def test_blocks_normal_bias_claim_when_cad_detects_pathology(self):
        vlm = "No acute cardiopulmonary disease is seen."
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels={"Effusion"})
        self.assertEqual(kept, [], "Normal-bias sentence must be blocked when CAD verified pathology")

    def test_allows_safe_normal_description_when_cad_is_normal(self):
        vlm = "The aortic arch is mildly tortuous without acute cardiopulmonary abnormality."
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels=set())
        self.assertEqual(len(kept), 1)

    def test_blocks_prompt_echo_sentences(self):
        vlm = (
            "Systematic Search: Evaluate airways, lungs/pleura, mediastinum, heart, and bones. "
            "Step 1: Examine the heart for any focal areas of abnormality. "
            "The trachea is midline."
        )
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels=set())
        self.assertEqual(kept, ["The trachea is midline"])

    def test_blocks_vlm_negating_cad_verified_pathology(self):
        vlm = "The right lung is clear without evidence of pleural effusions."
        kept = filter_unverified_vlm_sentences(vlm, allowed_raw_labels={"Effusion"})
        self.assertEqual(kept, [], "VLM must not contradict Stage 1 CAD's verified Effusion")

    def test_synthesized_report_obeys_5_rule_compliance(self):
        cad = [{
            "raw_label": "Pneumothorax",
            "clinical_name": "pneumothorax",
            "observation": "visceral pleural line with peripheral lucency",
            "location": "Right apical pleural space",
            "cal_prob": 0.78,
            "raw_prob": 0.45,
        }]
        report = synthesize_audited_rulebook_report("", cad, total_cad_models=4, use_vlm_commentary=False)
        ok, errs = validate_rulebook_compliance(report)
        self.assertTrue(ok, f"Rulebook validation failed: {errs}")
        self.assertIn("No radiographic evidence of pleural effusion", report)


class TestEnsembleMathAndCAM(unittest.TestCase):
    def test_op_norm_matches_xrv_reference(self):
        raw = np.array([[0.01, 0.25, 0.80], [0.50, 0.10, 0.60]], dtype=np.float32)
        thresh = np.array([0.05, 0.25, 0.50], dtype=np.float32)
        ours = compute_op_norm(raw, thresh)
        ref = xrv.models.op_norm(torch.from_numpy(raw), torch.from_numpy(thresh)).numpy()
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)

    def test_extract_grayscale_preserves_16bit_range(self):
        arr16 = np.array([[100, 1024], [2048, 4095]], dtype=np.uint16)
        pil16 = Image.fromarray(arr16)
        out = extract_grayscale_float32(pil16)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.max()), 4095.0)

    def test_localize_laterality_and_vertical_zones(self):
        # Image-left (cols 0..6 in 16x16) = Patient's RIGHT hemithorax
        cam_right_apex = np.zeros((16, 16), dtype=np.float32)
        cam_right_apex[3, 3] = 10.0
        self.assertEqual(localize_from_cam_grid(cam_right_apex, "Pneumothorax"), "Right apical pleural space")

        # Image-right (cols 9..15 in 16x16) = Patient's LEFT hemithorax, lower zone
        cam_left_base = np.zeros((16, 16), dtype=np.float32)
        cam_left_base[12, 12] = 10.0
        self.assertEqual(
            localize_from_cam_grid(cam_left_base, "Effusion"),
            "Left costophrenic angle and lower pleural space",
        )

    def test_cam_fusion_handles_partial_backbone_weights(self):
        # Simulate class missing 224 models (only 1 DenseNet + 1 ResNet, or only 224)
        c224 = np.ones((7, 7), dtype=np.float32) * 2.0
        c512 = np.ones((16, 16), dtype=np.float32) * 4.0
        fused = _fuse_multi_res_cam(c224, weight_224=1.0, cam_512=c512, weight_512=1.25, is_localized=True)
        self.assertEqual(fused.shape, (16, 16))
        self.assertAlmostEqual(float(fused.max()), 1.0, places=5)


class TestDatasetBuilder(unittest.TestCase):
    def test_canonicalization_and_bbox_grounding(self):
        self.assertEqual(canonicalize_label("Pleural_Thickening"), "Pleural Thickening")
        self.assertEqual(canonicalize_label("Nodule/Mass"), "Nodule")
        self.assertIsNone(canonicalize_label("No finding"))
        zone = bbox_to_anatomical_zone([0.1, 0.1, 0.3, 0.3], "Pneumothorax")
        self.assertEqual(zone, "right apical pleural space")

    def test_class_balancing_reduces_normal_bias(self):
        recs = [{"image_path": "a.png", "findings": []} for _ in range(80)]
        recs += [{"image_path": "b.png", "findings": [{"label": "Pneumothorax", "zone": "right apical pleural space"}]} for _ in range(10)]
        recs += [{"image_path": "c.png", "findings": [{"label": "Effusion", "zone": "left costophrenic angle"}]} for _ in range(10)]
        balanced = balance_records_against_normal_bias(recs, max_normal_ratio=0.20, min_class_count=30)
        normals = sum(1 for r in balanced if not r["findings"])
        self.assertLess(normals / len(balanced), 0.25)

    def test_rulebook_target_validates(self):
        txt = build_rulebook_target([{"label": "Cardiomegaly", "zone": "central mediastinum"}])
        ok, errs = validate_rulebook_compliance(txt)
        self.assertTrue(ok, f"Errors: {errs}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
