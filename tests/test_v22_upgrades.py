"""Automated regression tests for all ChestXStudio-VLM v2.2 expert fixes."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from execution.build_rulebook_dataset import (
    CHESTX_DET_ID2LABEL,
    _build_mode3_rejection_payload,
    _emit_three_mode_samples,
    _extract_mask_boxes,
    _findings_from_deepwise_item,
    _normalize_padchest_zone,
    _parse_padchest_locations,
    build_rulebook_target,
    normalize_cxr_to_rgb,
)
from execution.train_rulebook_vlm import (
    AUX_CLASSES,
    VisualConnectorHook,
    _compute_decision_token_weights,
    _labels_to_multihot,
    compute_multitask_loss,
)


class _DummyTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class _DummyVLM(nn.Module):
    def __init__(self, hidden: int = 32, vocab: int = 128):
        super().__init__()
        self.connector = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, pixel_values=None, **_kwargs):
        bsz, seq = input_ids.shape
        vis_in = torch.ones((bsz, 4, self.connector.in_features), dtype=torch.float32)
        vis_out = self.connector(vis_in)
        seq_feat = vis_out.mean(dim=1, keepdim=True).expand(bsz, seq, -1)
        logits = self.lm_head(seq_feat)
        return type("Out", (), {"logits": logits})()


class TestV22Upgrades(unittest.TestCase):
    def test_uniform_norm_8bit_and_16bit_match(self):
        arr8 = np.linspace(30, 180, 256 * 256, dtype=np.uint8).reshape(256, 256)
        arr16 = np.linspace(100, 4095, 256 * 256, dtype=np.uint16).reshape(256, 256)
        n8 = np.asarray(normalize_cxr_to_rgb(Image.fromarray(arr8), 512))
        n16 = np.asarray(normalize_cxr_to_rgb(Image.fromarray(arr16), 512))
        self.assertEqual(n8.shape, (512, 512, 3))
        self.assertLess(abs(float(n8.std()) - float(n16.std())), 3.0)

    def test_chestx_det_0_indexed_and_excludes_corrupted_12(self):
        self.assertEqual(CHESTX_DET_ID2LABEL[0], "Atelectasis")
        self.assertEqual(CHESTX_DET_ID2LABEL[2], "Cardiomegaly")
        self.assertEqual(CHESTX_DET_ID2LABEL[3], "Consolidation")
        self.assertEqual(CHESTX_DET_ID2LABEL[5], "Effusion")
        self.assertEqual(CHESTX_DET_ID2LABEL[11], "Pleural Thickening")
        self.assertNotIn(10, CHESTX_DET_ID2LABEL)
        self.assertNotIn(12, CHESTX_DET_ID2LABEL)

    def test_deepwise_json_recovers_nodule_and_pneumothorax(self):
        item = {
            "syms": ["Nodule", "Pneumothorax", "Consolidation"],
            "boxes": [[100, 100, 180, 180], [700, 50, 900, 300], [150, 600, 400, 850]],
        }
        findings = _findings_from_deepwise_item(item, 1024, 1024)
        labels = {f["label"] for f in findings}
        self.assertEqual(labels, {"Nodule", "Pneumothorax", "Consolidation"})

    def test_connected_components_split_multifocal_lesions(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[20:40, 20:40] = True
        mask[140:160, 140:160] = True
        boxes = _extract_mask_boxes(mask)
        self.assertEqual(len(boxes), 2)

    def test_padchest_location_and_zone_normalization(self):
        self.assertEqual(_parse_padchest_locations("[]"), "")
        self.assertEqual(_parse_padchest_locations("['right', 'basal']"), "right, basal")
        self.assertEqual(_normalize_padchest_zone("right, basal", "Atelectasis"), "right lower lung zone / lung base")
        self.assertEqual(_normalize_padchest_zone("cardiac", "Cardiomegaly"), "central mediastinum and cardiac silhouette")

    def test_unified_opening_kills_token1_tell(self):
        norm_rep = build_rulebook_target([])
        abn_rep = build_rulebook_target([{"label": "Effusion", "zone": "right costophrenic angle and lower pleural space"}])
        self.assertTrue(norm_rep.startswith("1. Systematic Visual Findings:\nThe trachea and central airways are midline and patent."))
        self.assertTrue(abn_rep.startswith("1. Systematic Visual Findings:\nThe trachea and central airways are midline and patent."))

    def test_mode3_balanced_and_no_laterality_flip(self):
        rng = random.Random(42)
        abn_rec = {"source": "t", "group_id": "g1", "image_path": "a.jpg", "findings": [{"label": "Cardiomegaly", "zone": "central mediastinum and cardiac silhouette"}]}
        norm_rec = {"source": "t", "group_id": "g2", "image_path": "b.jpg", "findings": []}
        subtypes = {_build_mode3_rejection_payload(abn_rec, rng)[2] for _ in range(50)}
        self.assertEqual(subtypes, {"abnormal_full_fp", "abnormal_distractor"})
        self.assertEqual(_build_mode3_rejection_payload(norm_rec, rng)[2], "normal_fp")

        norm_modes = [s["mode"] for s in _emit_three_mode_samples([norm_rec] * 200, seed=42)]
        self.assertGreater(norm_modes.count("cad_reject"), 80)

    def test_decision_token_weights_and_aux_visual_backward(self):
        report = build_rulebook_target([{"label": "Pneumothorax", "zone": "right apical pleural space"}])
        prefix = "PROMPT:"
        ids = torch.tensor([ord(c) for c in (prefix + report)], dtype=torch.long)
        wts = _compute_decision_token_weights(ids, len(prefix), _DummyTokenizer(), diag_weight=6.0, decision_weight=8.0)
        self.assertTrue(bool((wts[: len(prefix)] == 0.0).all()))
        self.assertGreater(int((wts == 8.0).sum().item()), 20)
        self.assertGreater(int((wts == 6.0).sum().item()), 20)
        self.assertGreater(int((wts == 0.4).sum().item()), 20)

        # Verify auxiliary visual hook captures connector output and backpropagates into connector weights
        model = _DummyVLM(hidden=32, vocab=128)
        hook = VisualConnectorHook(model)
        self.assertIsNotNone(hook.connector_module)
        aux_head = nn.Linear(32, len(AUX_CLASSES))
        batch = {
            "input_ids": torch.randint(1, 100, (2, 16)),
            "labels": torch.randint(1, 100, (2, 16)),
            "loss_weights": torch.ones((2, 16), dtype=torch.float32),
            "aux_targets": torch.stack([_labels_to_multihot(["Pneumothorax"]), _labels_to_multihot([])]),
        }
        total_loss, diag_loss, aux_loss, aux_logits = compute_multitask_loss(
            model, batch, vis_hook=hook, aux_head=aux_head, aux_loss_weight=0.5
        )
        self.assertIsNone(hook.captured)
        self.assertIsNotNone(aux_logits)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(aux_loss))
        total_loss.backward()
        self.assertIsNotNone(model.connector.weight.grad)
        self.assertGreater(float(model.connector.weight.grad.abs().sum().item()), 0.0)
        hook.remove()


if __name__ == "__main__":
    unittest.main()
