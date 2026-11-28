"""
ChestXStudio-VLM: Automated Unit & Integration Tests for Phase 3 Upgrades
Path: tests/test_v3_phase3.py

Verifies:
  1. v3 concise target format, exact `findings_span` offsets & compliance validator.
  2. `report_parser.py` parsing v3 reports (`FINDINGS: NONE` and multi-finding) without
     cross-class ambiguity (`Lung Opacity` vs `Consolidation`, `Calcification` vs `Nodule`).
  3. Zero patient leakage across `train`, `val`, and `--exclude-patients-json`.
  4. Per-source Normal balancing (`~50/50` SIIM, balanced multi-label sources, `<= 20%` global)
     and loud failure when `--strict-min-class` is violated.
  5. Exact offset-based 5.0x loss weighting on abnormal samples & loud failure when span missing.
  6. Mean+Max pooled `aux_head` forward/backward, `aux_head.pt` + `rulebook_v3.json` save/reload,
     and `vis_hook.captured is None` VRAM release.
  7. Resolution/JPEG non-flip augmentation preserves 512x512 RGB and never flips left/right.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.analyze_xray import ChestXRayPipeline, DEFAULT_BASE_MODEL
from execution.build_rulebook_dataset import (
    PATHOLOGY_ONTOLOGY,
    assert_zero_patient_leakage,
    balance_records_against_normal_bias,
    build_v3_target,
    filter_excluded_patients,
    format_vlm_chat_sample,
    load_excluded_patient_ids,
    load_siim_pneumo_hf,
    split_records_by_group,
    validate_min_class_counts,
    validate_rulebook_compliance,
    validate_v3_compliance,
)
from execution.report_parser import parse_report_findings
from execution.train_rulebook_vlm import (
    AUX_CLASS_TO_IDX,
    AUX_CLASSES,
    RulebookCXRDataset,
    VisualConnectorHook,
    _augment_cxr_nonflip,
    _compute_offset_token_weights,
    _labels_to_multihot,
    _source_to_aux_mask,
    build_aux_visual_head,
    collate_single_or_padded,
    compute_multitask_loss,
    compute_validation_metrics,
    save_adapter_bundle,
    verify_loss_masking,
)


class _DummyPatchVLM(nn.Module):
    def __init__(self, hidden: int = 32, vocab: int = 128):
        super().__init__()
        self.connector = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, pixel_values=None, **_kwargs):
        bsz, seq = input_ids.shape
        # Spatially sparse signal: only patch 0 is hot (tests Max+Mean sensitivity!)
        vis_in = torch.zeros((bsz, 8, self.connector.in_features), dtype=torch.float32)
        vis_in[:, 0, :] = 3.0
        vis_out = self.connector(vis_in)
        seq_feat = vis_out.mean(dim=1, keepdim=True).expand(bsz, seq, -1)
        return type("Out", (), {"logits": self.lm_head(seq_feat)})()

    def save_pretrained(self, out_dir: Path) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")


class _DummyProc:
    def save_pretrained(self, out_dir: Path) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TestPhase3Upgrades(unittest.TestCase):
    def test_v3_target_format_offsets_and_compliance(self):
        rep_norm, (s0, e0) = build_v3_target([])
        self.assertEqual(rep_norm[s0:e0], "FINDINGS: NONE")
        ok, errs = validate_v3_compliance(rep_norm)
        self.assertTrue(ok, f"v3 normal failed compliance: {errs}")
        ok_auto, _ = validate_rulebook_compliance(rep_norm, version="auto")
        self.assertTrue(ok_auto)

        rep_abn, (s1, e1) = build_v3_target([
            {"label": "Pneumothorax", "zone": "right apical pleural space"},
            {"label": "Effusion", "zone": "left costophrenic angle and lower pleural space"},
        ])
        self.assertEqual(rep_abn[s1:e1], rep_abn.splitlines()[0])
        self.assertTrue(rep_abn[s1:e1].startswith("FINDINGS: Pneumothorax"))
        ok2, errs2 = validate_v3_compliance(rep_abn)
        self.assertTrue(ok2, f"v3 abnormal failed compliance: {errs2}")

        # Must reject Rule-4 boilerplate chant in v3
        bad = "FINDINGS: NONE\nIMPRESSION: No radiographic evidence of pneumothorax."
        ok_bad, _ = validate_v3_compliance(bad)
        self.assertFalse(ok_bad)

    def test_report_parser_v3_none_and_unambiguous_multiclass(self):
        rep_norm, _ = build_v3_target([])
        parsed_norm = parse_report_findings(rep_norm)
        self.assertTrue(all(v == "absent" for v in parsed_norm.values()))

        # Ensure Lung Opacity never falsely triggers Consolidation & Calcification never triggers Nodule
        rep_abn, _ = build_v3_target([
            {"label": "Lung Opacity", "zone": "right mid lung zone"},
            {"label": "Calcification", "zone": "left upper lung zone (apical)"},
            {"label": "Pneumothorax", "zone": "right apical pleural space"},
        ])
        parsed_abn = parse_report_findings(rep_abn)
        present = {k for k, v in parsed_abn.items() if v == "present"}
        self.assertEqual(present, {"Lung Opacity", "Calcification", "Pneumothorax"})

    def test_zero_patient_leakage_across_splits_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            excl_path = Path(tmpdir) / "heldout_patient_ids.json"
            excl_path.write_text(json.dumps(["pat_001", "pat_005"]), encoding="utf-8")
            excluded = load_excluded_patient_ids(excl_path)

            recs = [
                {"source": "CheXpert", "patient_id": "pat_001", "group_id": "pat_001", "image_path": "a.jpg", "findings": []},
                {"source": "CheXpert", "patient_id": "pat_002", "group_id": "pat_002_s1", "image_path": "b1.jpg", "findings": [{"label": "Edema", "zone": "bilateral"}]},
                {"source": "CheXpert", "patient_id": "pat_002", "group_id": "pat_002_s2", "image_path": "b2.jpg", "findings": [{"label": "Edema", "zone": "bilateral"}]},
                {"source": "ChestX-Det-train", "patient_id": "pat_003", "group_id": "pat_003", "image_path": "c.jpg", "findings": []},
                {"source": "ChestX-Det-train", "patient_id": "pat_004", "group_id": "pat_004", "image_path": "d.jpg", "findings": [{"label": "Effusion", "zone": "left"}]},
                {"source": "SIIM-Pneumothorax-train", "patient_id": "pat_005", "group_id": "pat_005", "image_path": "e.jpg", "findings": [{"label": "Pneumothorax", "zone": "right"}]},
                {"source": "SIIM-Pneumothorax-train", "patient_id": "pat_006", "group_id": "pat_006", "image_path": "f.jpg", "findings": [{"label": "Pneumothorax", "zone": "right"}]},
            ]
            filtered, removed = filter_excluded_patients(recs, excluded)
            self.assertEqual(removed, 2)
            train_r, val_r = split_records_by_group(filtered, val_ratio=0.25, seed=42)
            # Multi-study patient pat_002 must stay in a single split (0 leakage)
            assert_zero_patient_leakage(train_r, val_r, excluded_ids=excluded)

            # Inject leakage intentionally and assert failure
            with self.assertRaises(AssertionError):
                assert_zero_patient_leakage(train_r + [recs[0]], val_r, excluded_ids=excluded)

    def test_per_source_normal_balancing_and_strict_min_class(self):
        recs = []
        # SIIM: 200 Pneumothorax + 600 Normal (75% normal raw -> downsampled to ~50/50 within SIIM)
        for i in range(200):
            recs.append({"source": "SIIM-Pneumothorax-train", "patient_id": f"siim_p_{i}", "group_id": f"siim_p_{i}", "image_path": "x.jpg", "findings": [{"label": "Pneumothorax", "zone": "right apical pleural space"}]})
        for i in range(600):
            recs.append({"source": "SIIM-Pneumothorax-train", "patient_id": f"siim_n_{i}", "group_id": f"siim_n_{i}", "image_path": "x.jpg", "findings": []})
        # ChestX-Det: 1000 abnormals across priority classes + only 15 Normals (1.5% raw -> oversampled!)
        for i in range(400):
            recs.append({"source": "ChestX-Det-train", "patient_id": f"cx_e_{i}", "group_id": f"cx_e_{i}", "image_path": "x.jpg", "findings": [{"label": "Effusion", "zone": "left"}]})
        for i in range(400):
            recs.append({"source": "ChestX-Det-train", "patient_id": f"cx_c_{i}", "group_id": f"cx_c_{i}", "image_path": "x.jpg", "findings": [{"label": "Cardiomegaly", "zone": "central"}]})
        for i in range(200):
            recs.append({"source": "ChestX-Det-train", "patient_id": f"cx_p_{i}", "group_id": f"cx_p_{i}", "image_path": "x.jpg", "findings": [{"label": "Pneumothorax", "zone": "left apical pleural space"}]})
        for i in range(15):
            recs.append({"source": "ChestX-Det-train", "patient_id": f"cx_n_{i}", "group_id": f"cx_n_{i}", "image_path": "x.jpg", "findings": []})

        balanced = balance_records_against_normal_bias(recs, max_normal_ratio=0.20, min_class_count=400, seed=42)
        total_norm = sum(1 for r in balanced if not r["findings"])
        self.assertLessEqual(total_norm / len(balanced), 0.21)

        siim_recs = [r for r in balanced if r["source"] == "SIIM-Pneumothorax-train"]
        siim_norm_frac = sum(1 for r in siim_recs if not r["findings"]) / len(siim_recs)
        self.assertGreaterEqual(siim_norm_frac, 0.45)
        self.assertLessEqual(siim_norm_frac, 0.55)

        cx_norm_cnt = sum(1 for r in balanced if r["source"] == "ChestX-Det-train" and not r["findings"])
        self.assertGreater(cx_norm_cnt, 15)

        # Oversampled ChestX-Det normals carry distinct aug_variant > 0
        self.assertTrue(any(r.get("aug_variant", 0) > 0 for r in balanced))

        # Validation with min_class_count=400 passes on balanced
        counts = validate_min_class_counts(balanced, min_class_count=400, strict=True)
        self.assertGreaterEqual(counts["Pneumothorax"], 400)
        self.assertGreaterEqual(counts["Effusion"], 400)

        # Strict check must fail loudly when a loaded priority class is < min_class_count
        underfilled = [r for r in recs if r["source"] == "ChestX-Det-train"]
        with self.assertRaises(ValueError):
            validate_min_class_counts(underfilled, min_class_count=400, strict=True)

    def test_offset_5x_loss_weighting_and_missing_span_failure(self):
        try:
            processor = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL, local_files_only=True)
        except Exception:
            self.skipTest("SmolVLM processor not in local cache")
        processor.image_processor.do_image_splitting = False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img_path = tmp / "sample.jpg"
            Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).convert("RGB").save(img_path)

            s_abn = format_vlm_chat_sample(
                {"source": "t", "patient_id": "p1", "group_id": "p1", "image_path": str(img_path), "findings": [{"label": "Pneumothorax", "zone": "right apical pleural space"}]},
                mode="pure",
                format_version="v3",
            )
            s_norm = format_vlm_chat_sample(
                {"source": "t", "patient_id": "p2", "group_id": "p2", "image_path": str(img_path), "findings": []},
                mode="pure",
                format_version="v3",
            )
            jsonl_p = tmp / "train.jsonl"
            jsonl_p.write_text(json.dumps(s_abn) + "\n" + json.dumps(s_norm) + "\n", encoding="utf-8")

            ds = RulebookCXRDataset(str(jsonl_p), processor=processor, image_size=512, findings_weight=5.0, augment=False)
            stats = verify_loss_masking(ds, processor)
            self.assertEqual(stats["abnormal_scanned"], 1)
            self.assertGreater(stats["min_5x_tokens"], 0)

            item0 = ds[0]
            wts = item0["loss_weights"]
            lbls = item0["labels"]
            self.assertTrue(bool((wts[lbls == -100] == 0.0).all()))
            self.assertGreater(int(((lbls != -100) & (wts == 5.0)).sum().item()), 0)
            self.assertGreater(int(((lbls != -100) & (wts == 1.0)).sum().item()), 0)

            # Corrupt span on abnormal sample -> must fail loudly
            bad_abn = dict(s_abn)
            bad_abn["findings_span"] = [0, 0]
            bad_jsonl = tmp / "bad.jsonl"
            bad_jsonl.write_text(json.dumps(bad_abn) + "\n", encoding="utf-8")
            bad_ds = RulebookCXRDataset(str(bad_jsonl), processor=processor, image_size=512, findings_weight=5.0, augment=False)
            with self.assertRaises(RuntimeError):
                verify_loss_masking(bad_ds, processor)

    def test_mean_max_aux_head_backward_save_reload_and_hook_clear(self):
        hidden = 32
        model = _DummyPatchVLM(hidden=hidden, vocab=128)
        hook = VisualConnectorHook(model)
        aux_head = build_aux_visual_head(hidden, len(AUX_CLASSES))
        self.assertEqual(aux_head[-1].in_features, 2 * hidden)
        self.assertEqual(aux_head[-1].out_features, len(AUX_CLASSES))

        batch = {
            "input_ids": torch.randint(1, 100, (2, 12)),
            "labels": torch.randint(1, 100, (2, 12)),
            "loss_weights": torch.ones((2, 12), dtype=torch.float32),
            "aux_targets": torch.stack([_labels_to_multihot(["Pneumothorax"]), _labels_to_multihot([])]),
        }
        total_loss, _, aux_loss, aux_logits = compute_multitask_loss(
            model, batch, vis_hook=hook, aux_head=aux_head, aux_loss_weight=1.0
        )
        # Immediately cleared before backward() to free connector graph VRAM
        self.assertIsNone(hook.captured)
        self.assertIsNotNone(aux_logits)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(aux_loss))
        total_loss.backward()
        self.assertGreater(float(model.connector.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(aux_head[-1].weight.grad.abs().sum().item()), 0.0)
        hook.remove()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ckpt"
            save_adapter_bundle(
                model,
                _DummyProc(),
                out,
                adapt_vision=True,
                image_size=512,
                extra_meta={"best_combined_score": 0.55},
                aux_head=aux_head,
                format_version="v3",
            )
            self.assertTrue((out / "rulebook_v3.json").exists())
            self.assertTrue((out / "aux_head.pt").exists())
            ckpt = torch.load(out / "aux_head.pt", map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["pooling"], "mean_max")
            self.assertEqual(ckpt["in_dim"], 2 * hidden)
            self.assertEqual(ckpt["classes"], AUX_CLASSES)
            reloaded = build_aux_visual_head(hidden, len(AUX_CLASSES))
            reloaded.load_state_dict(ckpt["state_dict"])
            reloaded.eval()
            aux_head.eval()
            test_x = torch.randn(2, 2 * hidden)
            self.assertTrue(torch.allclose(aux_head(test_x), reloaded(test_x)))

    def test_nonflip_resolution_jpeg_jitter_never_flips_laterality(self):
        # Asymmetric marker strictly on left half of image (Cols 20..180)
        arr = np.full((512, 512, 3), 20, dtype=np.uint8)
        arr[120:390, 40:180, :] = 220
        base_img = Image.fromarray(arr, mode="RGB")

        random.seed(123)
        outputs = [_augment_cxr_nonflip(base_img.copy(), 512, aug_variant=(i % 3) + 1) for i in range(15)]
        for aug in outputs:
            self.assertEqual(aug.size, (512, 512))
            self.assertEqual(aug.mode, "RGB")
            aug_arr = np.asarray(aug)
            left_mean = float(aug_arr[:, :256, :].mean())
            right_mean = float(aug_arr[:, 256:, :].mean())
            self.assertGreater(left_mean, right_mean + 30.0, "Horizontal flip detected!")

        # Variant calls produce non-identical augmented views
        arr0 = np.asarray(outputs[0])
        self.assertTrue(any(not np.array_equal(arr0, np.asarray(o)) for o in outputs[1:]))

    def test_source_masked_aux_bce_zeroes_unannotated_gradients(self):
        c_mask = _source_to_aux_mask("ChestX-Det-train")
        self.assertEqual(float(c_mask[AUX_CLASS_TO_IDX["Nodule"]]), 1.0)
        self.assertEqual(float(c_mask[AUX_CLASS_TO_IDX["Normal"]]), 1.0)
        self.assertEqual(float(c_mask[AUX_CLASS_TO_IDX["Edema"]]), 0.0)
        self.assertEqual(float(c_mask[AUX_CLASS_TO_IDX["Support Devices"]]), 0.0)

        x_mask = _source_to_aux_mask("CheXpert")
        self.assertEqual(float(x_mask[AUX_CLASS_TO_IDX["Edema"]]), 1.0)
        self.assertEqual(float(x_mask[AUX_CLASS_TO_IDX["Normal"]]), 1.0)
        self.assertEqual(float(x_mask[AUX_CLASS_TO_IDX["Nodule"]]), 0.0)
        self.assertEqual(float(x_mask[AUX_CLASS_TO_IDX["Mass"]]), 0.0)

        s_mask = _source_to_aux_mask("SIIM-Pneumothorax-train")
        self.assertEqual(float(s_mask.sum().item()), 1.0)
        self.assertEqual(float(s_mask[AUX_CLASS_TO_IDX["Pneumothorax"]]), 1.0)
        self.assertEqual(float(s_mask[AUX_CLASS_TO_IDX["Normal"]]), 0.0)
        self.assertEqual(float(s_mask[AUX_CLASS_TO_IDX["Edema"]]), 0.0)

        p_mask = _source_to_aux_mask("PadChest-GR-train")
        self.assertEqual(int(p_mask.sum().item()), len(AUX_CLASSES))

        hidden = 16
        model = _DummyPatchVLM(hidden=hidden, vocab=64)
        hook = VisualConnectorHook(model)
        aux_head = nn.Linear(2 * hidden, len(AUX_CLASSES))

        # Keep LM labels all -100 so backward gradients isolate exact aux_head output classes
        ptx_idx = AUX_CLASS_TO_IDX["Pneumothorax"]
        batch = collate_single_or_padded([
            {
                "input_ids": torch.randint(1, 50, (10,)),
                "labels": torch.full((10,), -100, dtype=torch.long),
                "loss_weights": torch.zeros((10,), dtype=torch.float32),
                "aux_targets": _labels_to_multihot(["Pneumothorax"]),
                "aux_mask": s_mask,
            }
        ], pad_token_id=0)
        total_loss, _, aux_loss, _ = compute_multitask_loss(
            model, batch, vis_hook=hook, aux_head=aux_head, aux_loss_weight=1.0
        )
        self.assertTrue(torch.isfinite(aux_loss))
        total_loss.backward()
        row_grads = aux_head.weight.grad.abs().sum(dim=1)
        self.assertGreater(float(row_grads[ptx_idx].item()), 0.0)
        other_grads = torch.cat([row_grads[:ptx_idx], row_grads[ptx_idx + 1 :]])
        self.assertEqual(float(other_grads.sum().item()), 0.0)

        # Validation macro-F1 ignores false positives on unannotated classes when aux_mask == 0
        class _DummyValModel(nn.Module):
            def __init__(self, logits_val):
                super().__init__()
                self.connector = nn.Identity()
                self._logits_val = logits_val

            def forward(self, input_ids, **_kw):
                bsz, seq = input_ids.shape
                # Hook captures connector output; return zeros vocab logits
                _ = self.connector(torch.zeros((bsz, 4, 8), dtype=torch.float32))
                return type("Out", (), {"logits": torch.zeros((bsz, seq, 32), dtype=torch.float32)})()

        val_model = _DummyValModel(None)
        val_hook = VisualConnectorHook(val_model)
        val_head = nn.Linear(16, len(AUX_CLASSES))
        with torch.no_grad():
            val_head.weight.zero_()
            # Predict Pneumothorax (+5) AND unannotated Cardiomegaly (+5)
            val_head.bias.fill_(-5.0)
            val_head.bias[ptx_idx] = 5.0
            val_head.bias[AUX_CLASS_TO_IDX["Cardiomegaly"]] = 5.0
        val_m = compute_validation_metrics(
            val_model,
            [batch],
            device="cpu",
            amp_dtype=torch.float32,
            vis_hook=val_hook,
            aux_head=val_head,
            aux_loss_weight=1.0,
        )
        self.assertAlmostEqual(val_m["aux_macro_f1"], 1.0, places=5)
        hook.remove()
        val_hook.remove()

    def test_siim_positives_only_and_zero_normal_balancing(self):
        img_pos = Image.fromarray(np.full((128, 128), 160, dtype=np.uint8), mode="L")
        img_neg = Image.fromarray(np.full((128, 128), 140, dtype=np.uint8), mode="L")
        mask_pos = np.zeros((128, 128), dtype=np.uint8)
        mask_pos[10:40, 20:50] = 255
        mask_pil = Image.fromarray(mask_pos, mode="L")

        fake_rows = [
            {"image": img_pos, "has_pneumo": 1, "mask": mask_pil, "image_id": "p1"},
            {"image": img_neg, "has_pneumo": 0, "mask": None, "image_id": "n1"},
            {"image": img_pos, "has_pneumo": 1, "mask": mask_pil, "image_id": "p2"},
            {"image": img_neg, "has_pneumo": 0, "mask": None, "image_id": "n2"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("datasets.load_dataset", return_value=fake_rows):
                pos_recs = load_siim_pneumo_hf(
                    img_cache_dir=tmpdir, image_size=256, split="train", streaming=True, positives_only=True
                )
                all_recs = load_siim_pneumo_hf(
                    img_cache_dir=tmpdir, image_size=256, split="test", streaming=True, positives_only=False
                )

            self.assertEqual(len(pos_recs), 2)
            self.assertTrue(all(len(r["findings"]) >= 1 and r["findings"][0]["label"] == "Pneumothorax" for r in pos_recs))
            train_cached = list((Path(tmpdir) / "siim_pneumo_train").glob("*.jpg"))
            self.assertEqual(len(train_cached), 2)
            self.assertEqual(len(all_recs), 4)

        # Zero-Normal SIIM alongside mixed ChestX-Det balances cleanly without error
        synthetic_recs = list(pos_recs)
        for i in range(6):
            synthetic_recs.append({
                "source": "ChestX-Det-train",
                "patient_id": f"cxd_abn_{i}",
                "group_id": f"cxd_abn_{i}",
                "image_path": f"cxd_a_{i}.jpg",
                "findings": [{"label": "Effusion", "zone": "left costophrenic angle and lower pleural space"}],
            })
        for i in range(6):
            synthetic_recs.append({
                "source": "ChestX-Det-train",
                "patient_id": f"cxd_norm_{i}",
                "group_id": f"cxd_norm_{i}",
                "image_path": f"cxd_n_{i}.jpg",
                "findings": [],
            })

        bal = balance_records_against_normal_bias(synthetic_recs, max_normal_ratio=0.20, min_class_count=10, seed=7)
        siim_norms = [r for r in bal if "siim" in r["source"].lower() and not r["findings"]]
        all_norms = [r for r in bal if not r["findings"]]
        self.assertEqual(len(siim_norms), 0)
        self.assertGreater(len(all_norms), 0)
        self.assertLessEqual(len(all_norms) / len(bal), 0.25)
        counts = validate_min_class_counts(bal, min_class_count=10, strict=True)
        self.assertGreaterEqual(counts["Pneumothorax"], 10)
        self.assertGreaterEqual(counts["Effusion"], 10)

    def test_multitile_320_tokens_and_inference_image_splitting_config(self):
        class _DummyMultiTileVLM(nn.Module):
            def __init__(self, hidden: int = 32, num_tiles: int = 5, tokens_per_tile: int = 64, vocab: int = 128):
                super().__init__()
                self.num_tiles = num_tiles
                self.tokens_per_tile = tokens_per_tile
                self.connector = nn.Linear(hidden, hidden)
                self.lm_head = nn.Linear(hidden, vocab)

            def forward(self, input_ids, pixel_values=None, **_kwargs):
                bsz, seq = input_ids.shape
                n_tiles = pixel_values.shape[1] if (pixel_values is not None and pixel_values.ndim == 5) else self.num_tiles
                vis_in = torch.zeros((bsz * n_tiles, self.tokens_per_tile, self.connector.in_features), dtype=torch.float32)
                vis_in[:, 0, :] = 2.5
                vis_out = self.connector(vis_in)
                pooled = vis_out.reshape(bsz, n_tiles * self.tokens_per_tile, -1).mean(dim=1, keepdim=True)
                return type("Out", (), {"logits": self.lm_head(pooled.expand(bsz, seq, -1))})()

            def save_pretrained(self, out_dir: Path) -> None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")

        hidden = 32
        model = _DummyMultiTileVLM(hidden=hidden, num_tiles=5, tokens_per_tile=64, vocab=128)
        hook = VisualConnectorHook(model)
        aux_head = build_aux_visual_head(hidden, len(AUX_CLASSES))
        batch = {
            "input_ids": torch.randint(1, 100, (2, 14)),
            "labels": torch.randint(1, 100, (2, 14)),
            "loss_weights": torch.ones((2, 14), dtype=torch.float32),
            "pixel_values": torch.zeros((2, 5, 3, 64, 64), dtype=torch.float32),
            "aux_targets": torch.stack([_labels_to_multihot(["Pneumothorax"]), _labels_to_multihot([])]),
            "aux_mask": torch.stack([_source_to_aux_mask("SIIM"), _source_to_aux_mask("ChestX-Det")]),
        }
        total_loss, _, aux_loss, aux_logits = compute_multitask_loss(
            model, batch, vis_hook=hook, aux_head=aux_head, aux_loss_weight=1.0
        )
        self.assertIsNone(hook.captured)
        self.assertEqual(tuple(aux_logits.shape), (2, len(AUX_CLASSES)))
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(aux_loss))
        total_loss.backward()
        self.assertGreater(float(model.connector.weight.grad.abs().sum().item()), 0.0)
        hook.remove()

        # Verify save_adapter_bundle + ChestXRayPipeline._resolve_image_splitting_config
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "v3_split_ckpt"
            save_adapter_bundle(
                model,
                _DummyProc(),
                out,
                adapt_vision=True,
                image_size=512,
                aux_head=aux_head,
                format_version="v3",
                do_image_splitting=True,
            )
            meta = json.loads((out / "rulebook_v3.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["do_image_splitting"])
            self.assertEqual(meta["image_longest_edge"], 1024)

            pipe = ChestXRayPipeline.__new__(ChestXRayPipeline)
            pipe.use_lora = True
            pipe.adapter_path = str(out)
            self.assertEqual(pipe._resolve_image_splitting_config(), (True, 1024))

            # Missing rulebook JSON -> False (v1/v2 backward-compatible)
            empty_dir = Path(tmpdir) / "v1_ckpt"
            empty_dir.mkdir(parents=True, exist_ok=True)
            pipe.adapter_path = str(empty_dir)
            self.assertEqual(pipe._resolve_image_splitting_config(), (False, 512))

        # Verify AutoProcessor produces 5 tiles (320 visual tokens) when cached locally
        try:
            proc = AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL, local_files_only=True)
        except Exception:
            proc = None
        if proc is not None:
            proc.image_processor.do_image_splitting = True
            proc.image_processor.size = {"longest_edge": 1024}
            p_txt = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Inspect."}]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            enc = proc(text=[p_txt], images=[[Image.new("RGB", (512, 512), (120, 120, 120))]], return_tensors="pt")
            self.assertEqual(enc["pixel_values"].shape[1], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
