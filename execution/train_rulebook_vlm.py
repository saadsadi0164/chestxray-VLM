"""
ChestXStudio-VLM: Multi-Dataset LoRA / QLoRA Rulebook Trainer & Anti-Hallucination Evaluator
Path: execution/train_rulebook_vlm.py

Trains `HuggingFaceTB/SmolVLM-500M-Instruct` on balanced multi-dataset 5-Rule CoT
records produced by `execution/build_rulebook_dataset.py`.

Upgrades in v2 audited trainer:
1. Verified Assistant-Only Loss Masking (`labels[:prompt_len] = -100`) with startup forensic printout.
2. Safe Vision-Language Bridge Adaptation (`--adapt-vision`): LoRA on LLM + ViT `out_proj`,
   full-rank `modules_to_save=["connector"]` (no double-wrapping `"proj"` conflict), and
   decoupled slower learning rate (`--connector-lr 2e-5`) for `connector`.
3. Rock-Solid Kaggle T4 / A100 / MPS Precision:
   - A100/L4 (`bf16`): native bf16 autocast + fp32 trainable master weights.
   - Kaggle T4 (`fp16`): fp16 autocast + GradScaler + fp32 trainable master weights (prevents NaN).
   - Single-GPU pin (`device_map={"": 0}`) prevents T4 x2 split errors.
   - Local Apple Silicon MPS / CPU: fp32 dry-run support.
4. Validation loss tracking per epoch with automatic best-checkpoint saving + 384-token
   5-Rule compliance, abnormal sensitivity, and Normal-bias audit.
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
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from execution.build_rulebook_dataset import validate_rulebook_compliance


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


class RulebookCXRDataset(Dataset):
    """Loads JSONL chat records and tokenizes image+text with exact assistant-only loss masking."""

    def __init__(self, jsonl_path: str, processor: Any, image_size: int = 512):
        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        messages = item["messages"]
        prompt_messages = messages[:-1]  # system + user

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
        labels[:prompt_len] = -100  # Mask system, image tokens, user query, and "Assistant:" prefix

        batch = {k: v[0] for k, v in full_inputs.items()}
        batch["labels"] = labels
        return batch


def verify_loss_masking(dataset: RulebookCXRDataset, processor: Any) -> None:
    """Prints a one-time forensic check of sample 0 so masking errors are caught before Epoch 1."""
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    sample = dataset[0]
    ids = sample["input_ids"].tolist()
    lbls = sample["labels"].tolist()

    masked_ids = [t for t, l in zip(ids, lbls) if l == -100]
    unmasked_ids = [t for t, l in zip(ids, lbls) if l != -100]

    if not unmasked_ids:
        raise RuntimeError("Forensic check failed: 0 unmasked assistant target tokens found!")

    print("[Masking Check] Masked prompt tail  :", repr(processor.tokenizer.decode(masked_ids[-20:])))
    print("[Masking Check] Unmasked target head:", repr(processor.tokenizer.decode(unmasked_ids[:25])))
    print("[Masking Check] Unmasked target tail:", repr(processor.tokenizer.decode(unmasked_ids[-15:])))


def collate_single_or_padded(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pads variable-length multimodal sequences dynamically in a batch."""
    if len(batch) == 1:
        return {k: v.unsqueeze(0) for k, v in batch[0].items()}

    max_len = max(item["input_ids"].shape[0] for item in batch)
    collated: dict[str, torch.Tensor] = {}

    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if key in ("input_ids", "attention_mask", "labels"):
            fill_val = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
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
        # Pin to GPU 0 so Kaggle T4 x2 does not split a 500M model across two GPUs
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
        # Adapt ViT attention output projections via LoRA, and keep `connector` full-rank via modules_to_save.
        # Do NOT add "proj" to target_modules or PEFT double-wraps connector.modality_projection.proj.
        target_modules.append("out_proj")
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

    # Keep all trainable adapter & connector parameters in fp32 for stable T4 fp16 / bf16 updates
    for param in peft_model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)

    peft_model.print_trainable_parameters()
    return peft_model


def build_optimizer(model: Any, lr: float, connector_lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """Applies a gentler learning rate to full-rank connector weights than to LoRA adapters."""
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

    print(
        f"[Optimizer] LoRA tensors: {len(lora_params)} (lr={lr}) | "
        f"Connector tensors: {len(connector_params)} (lr={connector_lr})"
    )
    return torch.optim.AdamW(param_groups)


def _autocast_ctx(device: str, amp_dtype: torch.dtype):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def compute_validation_loss(
    model: Any,
    val_loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for idx, batch in enumerate(val_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        batch = {k: v.to(device=device) for k, v in batch.items()}
        with _autocast_ctx(device, amp_dtype):
            outputs = model(**batch)
        loss_val = float(outputs.loss.item())
        if math.isfinite(loss_val):
            total_loss += loss_val
            count += 1
    model.train()
    return (total_loss / count) if count else float("inf")


@torch.no_grad()
def evaluate_validation_compliance(
    model: Any,
    processor: Any,
    val_jsonl: str,
    device: str,
    amp_dtype: torch.dtype,
    image_size: int = 512,
    max_eval: int = 20,
) -> dict[str, float]:
    """Generates deterministic predictions on validation split and audits format, recall, and normal bias."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    processor.tokenizer.padding_side = "left"

    total = 0
    compliant = 0
    non_normal_recall_hits = 0
    non_normal_total = 0
    normal_bias_on_abnormal = 0

    with open(val_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:max_eval]

    normal_re = re.compile(r"\bno acute (?:cardiopulmonary|thoracic|pulmonary)\b|\bnormal chest radiograph\b")

    for row in rows:
        img = Image.open(row["image_path"]).convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
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

        gt_labels = [lbl.lower() for lbl in row.get("labels", []) if lbl.lower() != "normal"]
        if gt_labels:
            non_normal_total += 1
            pred_low = pred.lower()
            hit = any(re.search(rf"\b{re.escape(lbl.split()[0])}", pred_low) for lbl in gt_labels)
            if hit:
                non_normal_recall_hits += 1
            elif normal_re.search(pred_low):
                normal_bias_on_abnormal += 1

    if hasattr(model, "config"):
        model.config.use_cache = False

    return {
        "eval_samples": float(total),
        "rulebook_compliance_rate": (compliant / total) if total else 0.0,
        "finding_mention_rate": (non_normal_recall_hits / non_normal_total) if non_normal_total else 1.0,
        "abnormal_normal_bias_rate": (normal_bias_on_abnormal / non_normal_total) if non_normal_total else 0.0,
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
        "rulebook_format": "5-rule-system-v2",
        "adapt_vision": adapt_vision,
        "image_size": image_size,
        "do_image_splitting": False,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(out_dir / "rulebook_v2.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmolVLM-500M with Multi-Dataset Rulebook CoT LoRA.")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./.tmp/rulebook_lora_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--connector-lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize-4bit", action="store_true", help="Enable NF4 QLoRA (CUDA GPU only).")
    parser.add_argument("--adapt-vision", action="store_true", help="Adapt ViT out_proj and modality connector.")
    parser.add_argument("--dry-run-steps", type=int, default=0, help="Stop after N optimizer steps for smoke testing.")
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        amp_dtype = torch.float32  # float32 keeps MPS LoRA backward gradients numerically stable
    else:
        device = "cpu"
        amp_dtype = torch.float32

    if args.quantize_4bit and device != "cuda":
        raise RuntimeError("--quantize-4bit requires a CUDA GPU with bitsandbytes.")

    print(f"[Trainer] Device={device} | AMP dtype={amp_dtype} | 4-bit={args.quantize_4bit} | Image={args.image_size}px")

    try:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=False)

    processor.image_processor.do_image_splitting = False
    pad_id = int(processor.tokenizer.pad_token_id or 0)

    train_ds = RulebookCXRDataset(args.train_jsonl, processor=processor, image_size=args.image_size)
    val_ds = RulebookCXRDataset(args.val_jsonl, processor=processor, image_size=args.image_size)
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

    optimizer = build_optimizer(model, lr=args.lr, connector_lr=args.connector_lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )

    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    out_dir = Path(args.output_dir)
    best_val_loss = float("inf")
    opt_steps = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        accum_count = 0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device=device, non_blocking=(device == "cuda")) for k, v in batch.items()}
            remaining = num_batches - (step - 1)
            window_size = min(args.grad_accum, remaining) if accum_count == 0 else window_size  # noqa: F821

            with _autocast_ctx(device, amp_dtype):
                outputs = model(**batch)
                raw_loss = outputs.loss

            if not torch.isfinite(raw_loss):
                print(f"[Warning] Non-finite loss at epoch {epoch + 1} step {step}; skipping micro-batch.")
                optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                accum_count = 0
                continue

            scaled_loss = raw_loss / max(1, window_size)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += float(raw_loss.item())
            accum_count += 1

            if accum_count >= window_size or step == num_batches:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_steps += 1
                avg_loss = running_loss / max(1, accum_count)
                if opt_steps == 1 or opt_steps % 10 == 0 or args.dry_run_steps > 0:
                    print(
                        f"[Epoch {epoch + 1}/{args.epochs} | Step {opt_steps}/{total_steps}] "
                        f"loss={avg_loss:.4f} grad_norm={float(grad_norm):.2f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                running_loss = 0.0
                accum_count = 0

                if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
                    break

        if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
            save_adapter_bundle(model, processor, out_dir, args.adapt_vision, args.image_size, {"dry_run": True})
            print(f"[Trainer] Dry-run complete. Saved checkpoint to {out_dir}")
            break

        val_loss = compute_validation_loss(model, val_loader, device=device, amp_dtype=amp_dtype, max_batches=100)
        print(f"[Epoch {epoch + 1}/{args.epochs}] val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter_bundle(
                model,
                processor,
                out_dir,
                args.adapt_vision,
                args.image_size,
                {"best_epoch": epoch + 1, "best_val_loss": round(best_val_loss, 4)},
            )
            print(f"[Trainer] New best checkpoint saved to {out_dir} (epoch {epoch + 1})")

    metrics = evaluate_validation_compliance(
        model,
        processor,
        val_jsonl=args.val_jsonl,
        device=device,
        amp_dtype=amp_dtype,
        image_size=args.image_size,
        max_eval=2 if args.dry_run_steps > 0 else 25,
    )
    print(f"[Trainer] Validation Audit Metrics: {metrics}")


if __name__ == "__main__":
    main()