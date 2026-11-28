"""
ChestXStudio-VLM: Automated Unit Tests for Phase 2 Negation-Aware Report Parser & Scoring
Path: tests/test_report_parser.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.build_rulebook_dataset import PATHOLOGY_ONTOLOGY, build_rulebook_target
from execution.compare_runs import format_comparison_markdown, load_benchmark_runs
from execution.report_parser import compute_clinical_metrics, parse_report_findings


class TestReportParser(unittest.TestCase):
    def test_returns_status_for_every_canonical_class(self):
        parsed = parse_report_findings("No acute cardiopulmonary disease.")
        self.assertEqual(set(parsed.keys()), set(PATHOLOGY_ONTOLOGY.keys()))
        self.assertTrue(all(v == "absent" for v in parsed.values()))

    def test_negated_comma_list_and_or_continuation(self):
        text = (
            "There is no radiographic evidence of pneumothorax, pleural effusion, "
            "focal lobar consolidation, pulmonary edema, or acute osseous fracture."
        )
        parsed = parse_report_findings(text)
        for cls in ("Pneumothorax", "Effusion", "Consolidation", "Edema", "Fracture"):
            self.assertEqual(parsed[cls], "absent")

    def test_prepositional_negations_without_free_of_negative_for(self):
        text = (
            "Visual inspection demonstrates right lower lobe consolidation without pleural effusion. "
            "Lungs are free of pneumothorax and negative for pulmonary edema."
        )
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Consolidation"], "present")
        self.assertEqual(parsed["Effusion"], "absent")
        self.assertEqual(parsed["Pneumothorax"], "absent")
        self.assertEqual(parsed["Edema"], "absent")

    def test_trailing_passive_negation(self):
        text = "A pleural effusion is not seen. Pneumothorax is ruled out. Fibrosis is absent."
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Effusion"], "absent")
        self.assertEqual(parsed["Pneumothorax"], "absent")
        self.assertEqual(parsed["Fibrosis"], "absent")

    def test_hedges_return_uncertain(self):
        text = (
            "Perihilar haziness is suggestive of pulmonary edema. "
            "Possible right apical pneumothorax. Left lower lobe consolidation cannot be excluded. "
            "Cannot exclude pulmonary nodule."
        )
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Edema"], "uncertain")
        self.assertEqual(parsed["Pneumothorax"], "uncertain")
        self.assertEqual(parsed["Consolidation"], "uncertain")
        self.assertEqual(parsed["Nodule"], "uncertain")

    def test_mixed_clause_heart_normal_and_pneumothorax(self):
        text = "The heart is normal, and there is pneumothorax."
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Cardiomegaly"], "absent")
        self.assertEqual(parsed["Pneumothorax"], "present")

    def test_contrast_conjunction_breaks_prior_negation(self):
        text = (
            "No pleural effusion or pneumothorax; however, bilateral alveolar consolidation is present. "
            "No rib fracture, but a right upper lobe pulmonary mass is identified."
        )
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Effusion"], "absent")
        self.assertEqual(parsed["Pneumothorax"], "absent")
        self.assertEqual(parsed["Fracture"], "absent")
        self.assertEqual(parsed["Consolidation"], "present")
        self.assertEqual(parsed["Mass"], "present")

    def test_structured_5_rule_multifinding_report(self):
        report = build_rulebook_target([
            {"label": "Pneumothorax", "zone": "right apical pleural space"},
            {"label": "Cardiomegaly", "zone": "central mediastinum and cardiac silhouette"},
            {"label": "Effusion", "zone": "left costophrenic angle and lower pleural space"},
        ])
        parsed = parse_report_findings(report)
        self.assertEqual(parsed["Pneumothorax"], "present")
        self.assertEqual(parsed["Cardiomegaly"], "present")
        self.assertEqual(parsed["Effusion"], "present")
        self.assertEqual(parsed["Edema"], "absent")
        self.assertEqual(parsed["Fracture"], "absent")

    def test_word_boundaries_and_cad_refutation_guards(self):
        text = (
            "Candidate Ensemble CAD findings: pneumothorax at right apical pleural space.\n"
            "1. Systematic Visual Findings:\n"
            "CAD flagged suspected pneumothorax at the right apical pleural space; "
            "however, direct visual evaluation of the radiograph does not substantiate this alert, "
            "demonstrating intact peripheral pulmonary vascular markings. "
            "No evidence of massive pleural effusion or mild opacity.\n"
            "2. Spatial Localization:\nBilateral lung fields (unremarkable).\n"
            "3. Clinical Impression:\nNo acute cardiopulmonary abnormality."
        )
        parsed = parse_report_findings(text)
        self.assertEqual(parsed["Pneumothorax"], "absent")
        self.assertEqual(parsed["Mass"], "absent")
        self.assertEqual(parsed["ILD"], "absent")
        self.assertEqual(parsed["Effusion"], "absent")

    def test_compute_clinical_metrics_and_combined_score_properties(self):
        good_recs = [
            {"gt_findings": ["Pneumothorax"], "report": build_rulebook_target([{"label": "Pneumothorax", "zone": "right apical pleural space"}])},
            {"gt_findings": ["Effusion"], "report": build_rulebook_target([{"label": "Effusion", "zone": "left costophrenic angle"}])},
            {"gt_findings": [], "report": build_rulebook_target([])},
            {"gt_findings": [], "report": build_rulebook_target([])},
        ]
        all_normal_recs = [
            {"gt_findings": ["Pneumothorax"], "report": build_rulebook_target([])},
            {"gt_findings": ["Effusion"], "report": build_rulebook_target([])},
            {"gt_findings": [], "report": build_rulebook_target([])},
            {"gt_findings": [], "report": build_rulebook_target([])},
        ]
        all_positive_recs = [
            {"gt_findings": ["Pneumothorax"], "report": "Pneumothorax and pleural effusion."},
            {"gt_findings": ["Effusion"], "report": "Pneumothorax and pleural effusion."},
            {"gt_findings": [], "report": "Pneumothorax and pleural effusion."},
            {"gt_findings": [], "report": "Pneumothorax and pleural effusion."},
        ]

        m_good = compute_clinical_metrics(good_recs)
        m_norm = compute_clinical_metrics(all_normal_recs)
        m_pos = compute_clinical_metrics(all_positive_recs)

        self.assertAlmostEqual(m_good["macro_sensitivity"], 1.0, places=4)
        self.assertAlmostEqual(m_good["normal_bias_rate"], 0.0, places=4)
        self.assertAlmostEqual(m_good["false_alarm_rate"], 0.0, places=4)
        self.assertAlmostEqual(m_good["rulebook_compliance"], 1.0, places=4)
        self.assertAlmostEqual(m_norm["normal_bias_rate"], 1.0, places=4)
        self.assertAlmostEqual(m_pos["false_alarm_rate"], 1.0, places=4)

        self.assertGreater(m_good["combined_score"], m_norm["combined_score"])
        self.assertGreater(m_good["combined_score"], m_pos["combined_score"])

        # Uncertain hedge in an abnormal study is not counted as blind normal-bias
        hedged_rec = [{
            "gt_findings": ["Edema"],
            "report": "Lungs are clear without pneumothorax. There is mild haziness that may represent pulmonary edema.",
        }]
        m_hedged = compute_clinical_metrics(hedged_rec)
        self.assertAlmostEqual(m_hedged["normal_bias_rate"], 0.0, places=4)

    def test_compare_runs_filters_non_report_jsons_and_renders_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "heldout_patient_ids.json").write_text('["p1", "p2"]', encoding="utf-8")
            (d / "run_a.json").write_text(json.dumps({
                "metadata": {"adapter": "v1", "llm_device": "mps", "llm_dtype": "torch.float32", "parser": "execution.report_parser"},
                "summary": compute_clinical_metrics([
                    {"gt_findings": ["Pneumothorax"], "report": build_rulebook_target([])},
                    {"gt_findings": [], "report": build_rulebook_target([])},
                ]),
                "records": [],
            }), encoding="utf-8")
            runs = load_benchmark_runs(d)
            self.assertEqual(len(runs), 1)
            md = format_comparison_markdown(runs)
            self.assertIn("run_a", md)
            self.assertIn("Pneumothorax", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
