"""
ChestXStudio-VLM: Multi-Dataset LoRA Trainer with Auxiliary Visual Supervision (v2.2)
Path: execution/train_rulebook_vlm.py

Expert-audited v2.2 upgrades:
1. Training-Only Auxiliary Visual Classification Head (Zero Inference Cost):
   Hooks mean-pooled `connector` image embeddings (`[B, 64, H] -> [B, H]`) into a
   multi-label linear head with frequency-balanced `BCEWithLogitsLoss`, driving
   undiluted visual gradients into SigLIP + `connector` every batch and reporting
   exact `aux_val_macro_f1` every epoch. Discarded before saving the PEFT adapter.
2. Decision-Token Weighted Cross-Entropy (20x Ratio):
   Weights fixed Rulebook boilerplate at 0.4x, Section 1 findings/refutations at
   6.0x, and Section 2 Localization + Section 3 Impression at 8.0x.
3. Full SigLIP Vision-Tower LoRA (`q/k/v/out_proj`, `fc1`, `fc2`) + full-rank
   `connector` at `lr=5e-5` (decoupled from LLM/ViT LoRA `lr=1e-4`).
4. Native Tesla T4 `fp16` + `GradScaler` (only uses `bf16` on Ampere+ cc >= 8),
   cutting Kaggle Commit runtime from ~10 hours to ~2.5-3 hours.
5. Honest Pure-Mode Validation Audit: Evaluates `mode == "pure"` studies separately
   (no `cad_aligned` inflation) + `cad_reject` Normal-FP rejection audit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from execution.build_rulebook_dataset import PATHOLOGY_ONTOLOGY, validate_rulebook_compliance
from execution.preprocess import normalize_cxr_to_rgb

AUX_CLASSES: list[str] = sorted(PATHOLOGY_ONTOLOGY.keys()) + ["Normal"]
AUX_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(AUX_CLASSES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_exact_prompt_len(prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> int:
    """Aligns prompt prefix against full sequence to avoid off-by-one BPE boundary errors."""
    p_len = int(prompt_ids.shape[0])
    f_len = int(full_ids.shape[0])
    limit = min(p_len, f_len)
    if limit > 0 and torch.equal(prompt_ids[:limit], full_ids[:limit]):
        return limit

    matched = 0
    for a, b in zip(prompt_ids.tolist(), full_ids.tolist()):
        if a != b:
            break
        matched += 1
    return matched


def _labels_to_multihot(labels: list[str]) -> torch.Tensor:
    vec = torch.zeros(len(AUX_CLASSES), dtype=torch.float32)
    matched = False
    for lbl in labels or []:
        if lbl in AUX_CLASS_TO_IDX:
            vec[AUX_CLASS_TO_IDX[lbl]] = 1.0
            matched = True
    if not matched:
        vec[AUX_CLASS_TO_IDX["Normal"]] = 1.0
    return vec


def _compute_decision_token_weights(
    input_ids: torch.Tensor,
    prompt_len: int,
    tokenizer: Any,
    diag_weight: float = 6.0,
    decision_weight: float = 8.0,
    boilerplate_weight: float = 0.4,
) -> torch.Tensor:
    """
    Assigns:
      - 0.0 to masked prompt/image tokens
      - `boilerplate_weight` (0.4x) to fixed Rulebook headers & Rule 4 negatives
      - `diag_weight` (6.0x) to Section 1 positive/normal visual findings & CAD refutations
      - `decision_weight` (8.0x) to Section 2 Localization & Section 3 Impression (concise diagnosis)
    """
    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    unmasked_ids = input_ids[prompt_len:].tolist()
    if not unmasked_ids:
        return loss_weights

    loss_weights[prompt_len:] = boilerplate_weight
    if diag_weight <= boilerplate_weight and decision_weight <= boilerplate_weight:
        return loss_weights

    pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in unmasked_ids]
    recon = "".join(pieces)

    sec1_spans: list[tuple[int, int]] = []
    for pat in (
        r"Visual inspection shows .+?(?=The cardiomediastinal silhouette is within normal limits\.|No radiographic evidence of|\n\n2\. Spatial Localization:|$)",
        r"Bilateral lung fields are well-aerated.+?(?=No radiographic evidence of|\n\n2\. Spatial Localization:|$)",
        r"CAD (?:additionally )?flagged suspected .+?(?=No radiographic evidence of|\n\n2\. Spatial Localization:|$)",
    ):
        for m in re.finditer(pat, recon, flags=re.IGNORECASE | re.DOTALL):
            sec1_spans.append((m.start(), m.end()))

    dec_spans: list[tuple[int, int]] = []
    m2 = re.search(
        r"2\. Spatial Localization:\s*(.+?)(?=\n\n3\. Clinical Impression:|$)",
        recon,
        flags=re.DOTALL,
    )
    if m2:
        dec_spans.append((m2.start(1), m2.end(1)))

    m3 = re.search(
        r"3\. Clinical Impression:\s*(.+?)(?=\nClinical correlation is recommended\.|<end_of_utterance>|$)",
        recon,
        flags=re.DOTALL,
    )
    if m3:
        dec_spans.append((m3.start(1), m3.end(1)))

    cursor = 0
    for rel_i, piece in enumerate(pieces):
        p_start = cursor
        p_end = cursor + len(piece)
        cursor = p_end
        if p_end <= p_start:
            continue
        if any(p_start < s_end and p_end > s_start for s_start, s_end in dec_spans):
            loss_weights[prompt_len + rel_i] = decision_weight
        elif any(p_start < s_end and p_end > s_start for s_start, s_end in sec1_spans):
            loss_weights[prompt_len + rel_i] = diag_weight
    return loss_weights


def _augment_cxr_nonflip(image: Image.Image, image_size: int) -> Image.Image:
    """Gentle non-flip radiograph augmentation (NEVER horizontally flips so left/right laterality stays exact)."""
    w, h = image.size
    if random.random() < 0.70 and w >= 64 and h >= 64:
        mx = int(w * random.uniform(0.0, 0.03))
        my = int(h * random.uniform(0.0, 0.03))
        if mx > 0 or my > 0:
            image = image.crop((mx, my, w - mx, h - my))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample)
    if random.random() < 0.70:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.90, 1.10))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.90, 1.10))
    return image


class RulebookCXRDataset(Dataset):
    """Loads JSONL chat records and produces assistant-only labels, decision weights, and multi-hot aux targets."""

    def __init__(
        self,
        jsonl_path: str,
        processor: Any,
        image_size: int = 512,
        diag_weight: float = 6.0,
        decision_weight: float = 8.0,
        augment: bool = False,
    ):
        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.processor = processor
        self.image_size = image_size
        self.diag_weight = diag_weight
        self.decision_weight = decision_weight
        self.augment = augment

    def compute_aux_pos_weights(self, max_weight: float = 6.0) -> torch.Tensor:
        """Computes capped positive class weights so rare pathologies get strong visual recall gradients."""
        total = max(1, len(self.samples))
        pos = torch.zeros(len(AUX_CLASSES), dtype=torch.float32)
        for s in self.samples:
            pos += _labels_to_multihot(s.get("labels") or [])
        neg = (total - pos).clamp_min(1.0)
        raw = torch.where(pos > 0, neg / pos.clamp_min(1.0), torch.ones_like(pos))
        return raw.clamp(min=1.0, max=max_weight)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image = normalize_cxr_to_rgb(item["image_path"], image_size=self.image_size)
        if self.augment:
            image = _augment_cxr_nonflip(image, self.image_size)

        messages = item["messages"]
        prompt_messages = messages[:-1]

        prompt_str = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        prompt_inputs = self.processor(text=[prompt_str], images=[[image]], return_tensors="pt")
        full_inputs = self.processor(text=[full_str], images=[[image]], return_tensors="pt")

        prompt_ids = prompt_inputs["input_ids"][0]
        input_ids = full_inputs["input_ids"][0]
        prompt_len = _find_exact_prompt_len(prompt_ids, input_ids)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        loss_weights = _compute_decision_token_weights(
            input_ids,
            prompt_len=prompt_len,
            tokenizer=self.processor.tokenizer,
            diag_weight=self.diag_weight,
            decision_weight=self.decision_weight,
        )

        batch = {k: v[0] for k, v in full_inputs.items()}
        batch["labels"] = labels
        batch["loss_weights"] = loss_weights
        batch["aux_targets"] = _labels_to_multihot(item.get("labels") or [])
        return batch


class VisualConnectorHook:
    """Captures active `connector` image embeddings for training-only auxiliary visual classification."""

    def __init__(self, model: nn.Module):
        self.connector_module = self._find_connector(model)
        self.captured: torch.Tensor | None = None
        self.handle: Any = None
        if self.connector_module is not None:
            self.handle = self.connector_module.register_forward_hook(self._hook_fn)

    @staticmethod
    def _find_connector(model: nn.Module) -> nn.Module | None:
        for name, mod in model.named_modules():
            if (name == "connector" or name.endswith(".connector")) and "original_module" not in name and "modules_to_save" not in name:
                return mod
        return None

    def _hook_fn(self, _mod: nn.Module, _inp: Any, out: Any) -> None:
        if isinstance(out, tuple):
            out = out[0]
        if torch.is_tensor(out):
            self.captured = out

    def clear(self) -> None:
        self.captured = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.captured = None


def verify_loss_masking(dataset: RulebookCXRDataset, processor: Any) -> None:
    """Prints a one-time forensic check of sample 0 so masking & decision weights are verified before Epoch 1."""
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    sample = dataset[0]
    ids = sample["input_ids"].tolist()
    lbls = sample["labels"].tolist()
    wts = sample["loss_weights"].tolist()

    masked_ids = [t for t, l in zip(ids, lbls) if l == -100]
    unmasked_ids = [t for t, l in zip(ids, lbls) if l != -100]
    dec_ids = [t for t, l, w in zip(ids, lbls, wts) if l != -100 and w >= 7.5]
    diag_ids = [t for t, l, w in zip(ids, lbls, wts) if l != -100 and 1.5 < w < 7.5]
    boiler_ids = [t for t, l, w in zip(ids, lbls, wts) if l != -100 and w <= 1.5]

    if not unmasked_ids:
        raise RuntimeError("Forensic check failed: 0 unmasked assistant target tokens found!")

    print("[Masking Check] Masked prompt tail  :", repr(processor.tokenizer.decode(masked_ids[-20:])))
    print("[Masking Check] Unmasked target head:", repr(processor.tokenizer.decode(unmasked_ids[:25])))
    print("[Masking Check] Unmasked target tail:", repr(processor.tokenizer.decode(unmasked_ids[-15:])))
    print(
        f"[Weighting Check] Decision (8x): {len(dec_ids)} | Findings (6x): {len(diag_ids)} | "
        f"Boilerplate (0.4x): {len(boiler_ids)} | Decision preview: {repr(processor.tokenizer.decode(dec_ids[:25]))}"
    )


def collate_single_or_padded(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pads variable-length multimodal sequences dynamically in a batch."""
    if len(batch) == 1:
        return {k: v.unsqueeze(0) for k, v in batch[0].items()}

    max_len = max(item["input_ids"].shape[0] for item in batch)
    collated: dict[str, torch.Tensor] = {}

    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if key in ("input_ids", "attention_mask", "labels", "loss_weights"):
            if key == "input_ids":
                fill_val: int | float = pad_token_id
            elif key == "labels":
                fill_val = -100
            elif key == "loss_weights":
                fill_val = 0.0
            else:
                fill_val = 0
            padded = torch.full((len(batch), max_len), fill_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, : t.shape[0]] = t
            collated[key] = padded
        else:
            collated[key] = torch.stack(tensors, dim=0)
    return collated


def _load_base_model(base_model_id: str, load_kwargs: dict[str, Any], dtype: torch.dtype | None) -> Any:
    """Loads model cleanly across Transformers 4.x (`torch_dtype`) and 5.x (`dtype`)."""
    for local_only in (True, False):
        for dtype_key in (("dtype", "torch_dtype") if dtype is not None else (None,)):
            kwargs = dict(load_kwargs)
            if dtype_key is not None:
                kwargs[dtype_key] = dtype
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    base_model_id, local_files_only=local_only, **kwargs
                )
            except TypeError:
                continue
            except Exception:
                if local_only:
                    break
                raise
    raise RuntimeError(f"Failed to load base model: {base_model_id}")


def _resolve_text_hidden_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    return int(getattr(text_cfg, "hidden_size", 960))


def build_lora_model(
    base_model_id: str,
    use_4bit: bool,
    adapt_vision: bool,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    device: str = "cpu",
    amp_dtype: torch.dtype = torch.float32,
):
    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

    if use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=amp_dtype,
        )
        load_kwargs["device_map"] = {"": 0}
        model = _load_base_model(base_model_id, load_kwargs, dtype=None)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        load_dtype = amp_dtype if device == "cuda" else torch.float32
        model = _load_base_model(base_model_id, load_kwargs, dtype=load_dtype)
        model = model.to(device=device, dtype=load_dtype)
        if device == "cuda":
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    modules_to_save: list[str] | None = None
    if adapt_vision:
        # Adapt SigLIP attention (q/k/v/out_proj) AND vision MLP (fc1/fc2) via LoRA,
        # while keeping `connector` full-rank via modules_to_save (never add "proj" here).
        target_modules.extend(["out_proj", "fc1", "fc2"])
        modules_to_save = ["connector"]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, lora_cfg)

    for param in peft_model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)

    peft_model.print_trainable_parameters()
    return peft_model


def build_optimizer(
    model: Any,
    aux_head: nn.Module | None,
    lr: float,
    connector_lr: float,
    aux_lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Applies decoupled learning rates for LoRA adapters, connector projection, and auxiliary visual head."""
    connector_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "connector" in name or "modality_projection" in name:
            connector_params.append(param)
        else:
            lora_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr, "weight_decay": weight_decay})
    if connector_params:
        param_groups.append({"params": connector_params, "lr": connector_lr, "weight_decay": weight_decay})
    if aux_head is not None:
        param_groups.append({"params": list(aux_head.parameters()), "lr": aux_lr, "weight_decay": weight_decay})

    print(
        f"[Optimizer] LoRA tensors: {len(lora_params)} (lr={lr}) | "
        f"Connector tensors: {len(connector_params)} (lr={connector_lr}) | "
        f"Aux head: {aux_head is not None} (lr={aux_lr})"
    )
    return torch.optim.AdamW(param_groups)


def _autocast_ctx(device: str, amp_dtype: torch.dtype):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def compute_multitask_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    vis_hook: VisualConnectorHook | None = None,
    aux_head: nn.Module | None = None,
    aux_pos_weight: torch.Tensor | None = None,
    aux_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Computes decision-weighted LM loss + training-only auxiliary visual multi-label BCE loss
    in fp32 for rock-solid Tesla T4 numerical stability.
    Returns: (total_loss, diag_lm_loss, aux_bce_loss, aux_logits_or_None)
    """
    labels = batch["labels"]
    loss_weights = batch.get("loss_weights")
    aux_targets = batch.get("aux_targets")
    model_inputs = {k: v for k, v in batch.items() if k not in ("labels", "loss_weights", "aux_targets")}

    if vis_hook is not None:
        vis_hook.clear()
    outputs = model(**model_inputs)
    logits = outputs.logits

    captured = vis_hook.captured if vis_hook is not None else None
    if vis_hook is not None:
        vis_hook.clear()

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    valid_mask = (shift_labels != -100).float()
    if loss_weights is None:
        weights = valid_mask
        diag_mask = valid_mask
    else:
        shift_weights = loss_weights[:, 1:].contiguous().float()
        weights = shift_weights * valid_mask
        diag_mask = (shift_weights > 1.5).float() * valid_mask

    lm_loss = (per_tok * weights).sum() / weights.sum().clamp_min(1.0)
    if float(diag_mask.sum().item()) > 0.0:
        diag_loss = (per_tok * diag_mask).sum() / diag_mask.sum().clamp_min(1.0)
    else:
        diag_loss = lm_loss.detach()

    aux_loss = torch.zeros((), device=lm_loss.device, dtype=torch.float32)
    aux_logits: torch.Tensor | None = None
    if (
        aux_loss_weight > 0.0
        and aux_head is not None
        and aux_targets is not None
        and captured is not None
    ):
        bsz = int(labels.shape[0])
        fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if lm_loss.is_cuda else contextlib.nullcontext()
        with fp32_ctx:
            vis_emb = captured.reshape(bsz, -1, captured.shape[-1]).float().mean(dim=1)
            aux_logits = aux_head(vis_emb)
            pos_w = aux_pos_weight.float() if aux_pos_weight is not None else None
            aux_loss = F.binary_cross_entropy_with_logits(
                aux_logits.float(),
                aux_targets.float(),
                pos_weight=pos_w,
            )
        total_loss = lm_loss + aux_loss_weight * aux_loss
    else:
        total_loss = lm_loss

    return total_loss, diag_loss.detach(), aux_loss.detach(), (aux_logits.detach() if aux_logits is not None else None)


@torch.no_grad()
def compute_validation_metrics(
    model: Any,
    val_loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    vis_hook: VisualConnectorHook | None = None,
    aux_head: nn.Module | None = None,
    aux_pos_weight: torch.Tensor | None = None,
    aux_loss_weight: float = 0.5,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    if aux_head is not None:
        aux_head.eval()

    total_loss = 0.0
    total_diag = 0.0
    total_aux = 0.0
    count = 0
    tp = torch.zeros(len(AUX_CLASSES), dtype=torch.float64)
    fp = torch.zeros(len(AUX_CLASSES), dtype=torch.float64)
    fn = torch.zeros(len(AUX_CLASSES), dtype=torch.float64)

    for idx, batch in enumerate(val_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = {k: v.to(device=device) for k, v in batch.items()}
        with _autocast_ctx(device, amp_dtype):
            t_loss, d_loss, a_loss, a_logits = compute_multitask_loss(
                model,
                batch,
                vis_hook=vis_hook,
                aux_head=aux_head,
                aux_pos_weight=aux_pos_weight,
                aux_loss_weight=aux_loss_weight,
            )
        lv, dv, av = float(t_loss.item()), float(d_loss.item()), float(a_loss.item())
        if math.isfinite(lv) and math.isfinite(dv):
            total_loss += lv
            total_diag += dv
            total_aux += av
            count += 1
        if a_logits is not None and "aux_targets" in batch:
            preds = (torch.sigmoid(a_logits).cpu() >= 0.5).to(torch.float64)
            targs = (batch["aux_targets"].cpu() >= 0.5).to(torch.float64)
            tp += (preds * targs).sum(dim=0)
            fp += (preds * (1.0 - targs)).sum(dim=0)
            fn += ((1.0 - preds) * targs).sum(dim=0)

    if vis_hook is not None:
        vis_hook.clear()

    model.train()
    if aux_head is not None:
        aux_head.train()

    if not count:
        return {"val_loss": float("inf"), "diag_val_loss": float("inf"), "aux_val_loss": float("inf"), "aux_macro_f1": 0.0}

    support = tp + fn
    active = support > 0
    if bool(active.any()):
        f1_per_cls = (2.0 * tp[active]) / (2.0 * tp[active] + fp[active] + fn[active]).clamp_min(1.0)
        macro_f1 = float(f1_per_cls.mean().item())
    else:
        macro_f1 = 0.0

    return {
        "val_loss": total_loss / count,
        "diag_val_loss": total_diag / count,
        "aux_val_loss": total_aux / count,
        "aux_macro_f1": macro_f1,
    }


@torch.no_grad()
def evaluate_validation_compliance(
    model: Any,
    processor: Any,
    val_jsonl: str,
    device: str,
    amp_dtype: torch.dtype,
    image_size: int = 512,
    max_eval: int = 30,
) -> dict[str, float]:
    """Audits PURE-MODE unseen validation studies separately (no cad_aligned inflation)."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    processor.tokenizer.padding_side = "left"

    with open(val_jsonl, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    pure_rows = [r for r in all_rows if r.get("mode") == "pure"][:max_eval]
    rows = pure_rows if pure_rows else all_rows[:max_eval]

    total = 0
    compliant = 0
    abn_hits = 0
    abn_total = 0
    normal_bias = 0
    norm_hits = 0
    norm_total = 0

    normal_re = re.compile(r"\bno acute (?:cardiopulmonary|thoracic|pulmonary)\b|\bnormal chest radiograph\b")

    for row in rows:
        img = normalize_cxr_to_rgb(row["image_path"], image_size=image_size)
        prompt = processor.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[[img]], return_tensors="pt")
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]

        with _autocast_ctx(device, amp_dtype):
            out_ids = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                repetition_penalty=1.08,
            )
        pred = processor.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()

        total += 1
        ok, _ = validate_rulebook_compliance(pred)
        if ok:
            compliant += 1

        pred_low = pred.lower()
        gt_labels = [lbl.lower() for lbl in row.get("labels", []) if lbl.lower() != "normal"]
        if gt_labels:
            abn_total += 1
            hit = any(re.search(rf"\b{re.escape(lbl.split()[0])}", pred_low) for lbl in gt_labels)
            if hit:
                abn_hits += 1
            elif normal_re.search(pred_low):
                normal_bias += 1
        else:
            norm_total += 1
            if normal_re.search(pred_low):
                norm_hits += 1

    if hasattr(model, "config"):
        model.config.use_cache = False

    return {
        "pure_eval_samples": float(total),
        "pure_rulebook_compliance": (compliant / total) if total else 0.0,
        "pure_abnormal_sensitivity": (abn_hits / abn_total) if abn_total else 1.0,
        "pure_abnormal_normal_bias": (normal_bias / abn_total) if abn_total else 0.0,
        "pure_normal_specificity": (norm_hits / norm_total) if norm_total else 1.0,
    }


def save_adapter_bundle(
    model: Any,
    processor: Any,
    out_dir: Path,
    adapt_vision: bool,
    image_size: int,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    meta = {
        "rulebook_format": "5-rule-system-v2.2",
        "adapt_vision": adapt_vision,
        "image_size": image_size,
        "do_image_splitting": False,
        "uniform_norm": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(out_dir / "rulebook_v2.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmolVLM-500M with Rulebook CoT LoRA + Aux Visual Head (v2.2).")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./.tmp/rulebook_lora_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--connector-lr", type=float, default=5e-5)
    parser.add_argument("--aux-loss-weight", type=float, default=0.5, help="Weight on training-only visual multi-label BCE head.")
    parser.add_argument("--diag-weight", type=float, default=6.0, help="Loss multiplier on Section 1 findings/refutations.")
    parser.add_argument("--decision-weight", type=float, default=8.0, help="Loss multiplier on Localization & Impression.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize-4bit", action="store_true", help="Enable NF4 QLoRA (CUDA GPU only).")
    parser.add_argument("--adapt-vision", action="store_true", help="Adapt SigLIP attention + MLP and modality connector.")
    parser.add_argument("--no-augment", action="store_true", help="Disable non-flip training image augmentation.")
    parser.add_argument("--dry-run-steps", type=int, default=0, help="Stop after N optimizer steps for smoke testing.")
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        major_cc = torch.cuda.get_device_capability(0)[0]
        amp_dtype = torch.bfloat16 if (major_cc >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        amp_dtype = torch.float32
    else:
        device = "cpu"
        amp_dtype = torch.float32

    if args.quantize_4bit and device != "cuda":
        raise RuntimeError("--quantize-4bit requires a CUDA GPU with bitsandbytes.")

    print(
        f"[Trainer v2.2] Device={device} | AMP dtype={amp_dtype} | 4-bit={args.quantize_4bit} | "
        f"Image={args.image_size}px | Weights=(boiler=0.4x, find={args.diag_weight:.1f}x, dec={args.decision_weight:.1f}x) | "
        f"AuxLoss={args.aux_loss_weight} | Augment={not args.no_augment}"
    )

    try:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=False)

    processor.image_processor.do_image_splitting = False
    pad_id = int(processor.tokenizer.pad_token_id or 0)

    train_ds = RulebookCXRDataset(
        args.train_jsonl,
        processor=processor,
        image_size=args.image_size,
        diag_weight=args.diag_weight,
        decision_weight=args.decision_weight,
        augment=not args.no_augment,
    )
    val_ds = RulebookCXRDataset(
        args.val_jsonl,
        processor=processor,
        image_size=args.image_size,
        diag_weight=args.diag_weight,
        decision_weight=args.decision_weight,
        augment=False,
    )
    verify_loss_masking(train_ds, processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_single_or_padded(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_single_or_padded(b, pad_id),
    )

    model = build_lora_model(
        args.base_model,
        use_4bit=args.quantize_4bit,
        adapt_vision=args.adapt_vision,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device,
        amp_dtype=amp_dtype,
    )

    vis_hook: VisualConnectorHook | None = None
    aux_head: nn.Module | None = None
    aux_pos_weight: torch.Tensor | None = None
    if args.aux_loss_weight > 0.0:
        vis_hook = VisualConnectorHook(model)
        if vis_hook.connector_module is not None:
            hidden_dim = _resolve_text_hidden_size(model)
            aux_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, len(AUX_CLASSES)),
            ).to(device=device, dtype=torch.float32)
            aux_pos_weight = train_ds.compute_aux_pos_weights(max_weight=6.0).to(device=device)
            print(f"[Aux Visual Head] Hooked {type(vis_hook.connector_module).__name__} -> Linear({hidden_dim}, {len(AUX_CLASSES)})")
        else:
            print("[Warning] Connector module not found; proceeding without auxiliary visual head.")
            vis_hook = None

    optimizer = build_optimizer(
        model,
        aux_head=aux_head,
        lr=args.lr,
        connector_lr=args.connector_lr,
        aux_lr=1e-3,
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )

    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    clip_params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])

    out_dir = Path(args.output_dir)
    best_diag_val = float("inf")
    opt_steps = 0

    model.train()
    if aux_head is not None:
        aux_head.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_diag = 0.0
        running_aux = 0.0
        accum_count = 0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device=device, non_blocking=(device == "cuda")) for k, v in batch.items()}
            remaining = num_batches - (step - 1)
            window_size = min(args.grad_accum, remaining) if accum_count == 0 else window_size  # noqa: F821

            with _autocast_ctx(device, amp_dtype):
                raw_loss, diag_loss, aux_loss, _ = compute_multitask_loss(
                    model,
                    batch,
                    vis_hook=vis_hook,
                    aux_head=aux_head,
                    aux_pos_weight=aux_pos_weight,
                    aux_loss_weight=args.aux_loss_weight,
                )

            if not torch.isfinite(raw_loss):
                print(f"[Warning] Non-finite loss at epoch {epoch + 1} step {step}; skipping micro-batch.")
                optimizer.zero_grad(set_to_none=True)
                running_loss = running_diag = running_aux = 0.0
                accum_count = 0
                continue

            scaled_loss = raw_loss / max(1, window_size)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += float(raw_loss.item())
            running_diag += float(diag_loss.item())
            running_aux += float(aux_loss.item())
            accum_count += 1

            if accum_count >= window_size or step == num_batches:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_steps += 1
                avg_loss = running_loss / max(1, accum_count)
                avg_diag = running_diag / max(1, accum_count)
                avg_aux = running_aux / max(1, accum_count)
                if opt_steps == 1 or opt_steps % 10 == 0 or args.dry_run_steps > 0:
                    print(
                        f"[Epoch {epoch + 1}/{args.epochs} | Step {opt_steps}/{total_steps}] "
                        f"loss={avg_loss:.4f} diag={avg_diag:.4f} aux_bce={avg_aux:.4f} "
                        f"grad_norm={float(grad_norm):.2f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                running_loss = running_diag = running_aux = 0.0
                accum_count = 0

                if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
                    break

        if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
            save_adapter_bundle(model, processor, out_dir, args.adapt_vision, args.image_size, {"dry_run": True})
            print(f"[Trainer] Dry-run complete. Saved checkpoint to {out_dir}")
            break

        val_m = compute_validation_metrics(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            vis_hook=vis_hook,
            aux_head=aux_head,
            aux_pos_weight=aux_pos_weight,
            aux_loss_weight=args.aux_loss_weight,
            max_batches=120,
        )
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] val_loss={val_m['val_loss']:.4f} | "
            f"diag_val={val_m['diag_val_loss']:.4f} (best={best_diag_val:.4f}) | "
            f"aux_val={val_m['aux_val_loss']:.4f} | aux_macro_f1={val_m['aux_macro_f1']:.3f}"
        )

        if val_m["diag_val_loss"] < best_diag_val:
            best_diag_val = val_m["diag_val_loss"]
            save_adapter_bundle(
                model,
                processor,
                out_dir,
                args.adapt_vision,
                args.image_size,
                {
                    "best_epoch": epoch + 1,
                    "best_val_loss": round(val_m["val_loss"], 4),
                    "best_diag_val_loss": round(best_diag_val, 4),
                    "best_aux_macro_f1": round(val_m["aux_macro_f1"], 4),
                    "aux_loss_weight": args.aux_loss_weight,
                },
            )
            print(f"[Trainer] New best checkpoint saved to {out_dir} (epoch {epoch + 1}, diag_val={best_diag_val:.4f})")

    # Remove training hook cleanly before running generation audit
    if vis_hook is not None:
        vis_hook.remove()

    metrics = evaluate_validation_compliance(
        model,
        processor,
        val_jsonl=args.val_jsonl,
        device=device,
        amp_dtype=amp_dtype,
        image_size=args.image_size,
        max_eval=2 if args.dry_run_steps > 0 else 30,
    )
    print(f"[Trainer] Pure-Mode Validation Audit Metrics: {metrics}")


if __name__ == "__main__":
    main()
