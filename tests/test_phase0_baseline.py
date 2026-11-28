"""
ChestXStudio-VLM: Automated Unit Tests for Phase 0 Held-Out Manifest & Baseline Scorer
Path: tests/test_phase0_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.benchmark_baseline import (
    _fallback_compute_metrics,
    _fallback_parse_report_status,
    _score_record,
    load_manifest,
)
from execution.build_heldout_test_set import select_patient_grouped_holdout


class TestPhase0HoldoutAndBaseline(unittest.TestCase):
    def test_select_patient_grouped_enforces_zero_patient_overlap(self):
        cands = []
        # Create multi-view duplicates for same patients to prove grouping keeps 1 study/patient
        for p in range(60):
            cands.append({
                "study_id": f"n_a_{p}",
                "patient_id": f"pat_n_{p}",
                "source": "ChestX-Det-test" if p % 2 == 0 else "PadChest-GR-test",
                "image_path": f"/tmp/na_{p}.png",
                "findings": [],
                "image_fingerprint": f"fp_na_{p}",
            })
            cands.append({
                "study_id": f"n_b_{p}",
                "patient_id": f"pat_n_{p}",
                "source": "PadChest-GR-test",
                "image_path": f"/tmp/nb_{p}.png",
                "findings": [],
                "image_fingerprint": f"fp_nb_{p}",
            })

        classes = [
            "Pneumothorax", "Effusion", "Consolidation", "Cardiomegaly",
            "Edema", "Nodule", "Atelectasis", "Fracture",
        ]
        for p in range(100):
            cls = classes[p % len(classes)]
            cands.append({
                "study_id": f"ab_a_{p}",
                "patient_id": f"pat_ab_{p}",
                "source": "ChestX-Det-test",
                "image_path": f"/tmp/aba_{p}.png",
                "findings": [{"label": cls, "zone": "right lung field"}],
                "image_fingerprint": f"fp_aba_{p}",
            })
            cands.append({
                "study_id": f"ab_b_{p}",
                "patient_id": f"pat_ab_{p}",
                "source": "ChestX-Det-test",
                "image_path": f"/tmp/abb_{p}.png",
                "findings": [{"label": cls, "zone": "right lung field"}],
                "image_fingerprint": f"fp_abb_{p}",
            })

        selected = select_patient_grouped_holdout(cands, target_normal=30, target_abnormal=60, seed=42)
        self.assertEqual(len(selected), 90)
        pids = [r["patient_id"] for r in selected]
        self.assertEqual(len(set(pids)), 90)
        present_cls = {f["label"] for r in selected for f in r["findings"]}
        self.assertGreaterEqual(len(present_cls), 8)

    def test_negated_findings_never_counted_as_present(self):
        report = (
            "1. Systematic Visual Findings:\n"
            "No pneumothorax or pleural effusion is seen. Lungs are without consolidation, "
            "negative for cardiomegaly, and free of pulmonary edema. "
            "However, there is a right humeral fracture.\n"
            "2. Spatial Localization:\nRight humerus.\n"
            "3. Clinical Impression:\nAcute fracture."
        )
        parsed = _fallback_parse_report_status(report)
        self.assertEqual(parsed, {"Fracture": "present"})

    def test_normal_bias_false_alarm_and_manifest_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mpath = Path(tmpdir) / "test_manifest.jsonl"
            rows = [
                {"study_id": "s1", "patient_id": "p1", "source": "t", "image_path": "a.png", "gt_findings": ["Pneumothorax"]},
                {"study_id": "s2", "patient_id": "p2", "source": "t", "image_path": "b.png", "gt_findings": ["Effusion"]},
                {"study_id": "s3", "patient_id": "p3", "source": "t", "image_path": "c.png", "gt_findings": []},
            ]
            mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            loaded = load_manifest(mpath)
            self.assertEqual(len(loaded), 3)

            r1 = _score_record(loaded[0], "Lungs are clear without pneumothorax. Normal chest.", 0.5)
            r2 = _score_record(loaded[1], "There is a right pleural effusion.", 0.5)
            r3 = _score_record(loaded[2], "Left lower lobe consolidation is noted.", 0.5)

            self.assertTrue(r1["normal_bias"])
            self.assertFalse(r2["normal_bias"])
            self.assertTrue(r3["false_alarm"])

            summary = _fallback_compute_metrics([r1, r2, r3])
            self.assertAlmostEqual(summary["normal_bias_rate"], 0.5, places=4)
            self.assertAlmostEqual(summary["false_alarm_rate"], 1.0, places=4)
            self.assertEqual(summary["per_finding"]["Effusion"]["tp"], 1)
            self.assertEqual(summary["per_finding"]["Pneumothorax"]["fn"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
