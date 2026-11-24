"""
ChestXStudio-VLM: Multi-Dataset QLoRA / LoRA Rulebook Trainer & Anti-Hallucination Evaluator
Path: execution/train_rulebook_vlm.py

Trains `HuggingFaceTB/SmolVLM-500M-Instruct` on balanced multi-dataset 5-Rule CoT
records produced by `execution/build_rulebook_dataset.py`.

Key upgrades over initial Kaggle training:
  1. Assistant-Only Loss Masking (`labels[:prompt_len] = -100`) — prevents prompt memorization.
  2. Vision-Language Modality Projector + ViT `out_proj` Adaptation (`--adapt-vision`) — allows
     the visual pathway itself to adapt to radiographic subtle findings (pleural lines, nodules).
  3. Dual-Hardware Support:
       - Cloud CUDA GPU: 4-bit NF4 QLoRA + bf16 + gradient checkpointing
       - Local Apple Silicon MPS / CPU: LoRA smoke-testing & verification
  4. Automated Post-Training Rulebook Compliance & Hallucination Audit on validation split.
"""

from __future__ import annotations

import argparse
import json
import os
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
        image = Image.open(item["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
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

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask system, image tokens, user query, and "Assistant:" prefix

        batch = {k: v[0] for k, v in full_inputs.items()}
        batch["labels"] = labels
        return batch


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
                padded[i, :t.shape[0]] = t
            collated[key] = padded
        else:
            collated[key] = torch.stack(tensors, dim=0)
    return collated


def build_lora_model(
    base_model_id: str,
    use_4bit: bool,
    adapt_vision: bool,
    lora_r: int = 32,
    lora_alpha: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

    if use_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = dtype

    try:
        model = AutoModelForImageTextToText.from_pretrained(base_model_id, local_files_only=True, **load_kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(base_model_id, local_files_only=False, **load_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device=device, dtype=dtype)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    modules_to_save: list[str] | None = None
    if adapt_vision:
        target_modules.extend(["out_proj", "proj"])
        modules_to_save = ["connector"]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


@torch.no_grad()
def evaluate_validation_compliance(
    model: Any,
    processor: Any,
    val_jsonl: str,
    device: str,
    dtype: torch.dtype,
    max_eval: int = 10,
) -> dict[str, float]:
    """Generates predictions on validation split and checks 5-Rule format compliance."""
    model.eval()
    total = 0
    compliant = 0
    non_normal_recall_hits = 0
    non_normal_total = 0

    with open(val_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:max_eval]

    for row in rows:
        img = Image.open(row["image_path"]).convert("RGB").resize((512, 512))
        prompt = processor.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[[img]], return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
            for k, v in inputs.items()
        }
        in_len = inputs["input_ids"].shape[1]
        out_ids = model.generate(**inputs, max_new_tokens=220, do_sample=False, repetition_penalty=1.15)
        pred = processor.decode(out_ids[0][in_len:], skip_special_tokens=True).strip()

        total += 1
        ok, _ = validate_rulebook_compliance(pred)
        if ok:
            compliant += 1

        gt_labels = [lbl.lower() for lbl in row.get("labels", [])]
        if gt_labels:
            non_normal_total += 1
            pred_low = pred.lower()
            if any(re.search(rf"\b{re.escape(lbl.split()[0])}", pred_low) for lbl in gt_labels):
                non_normal_recall_hits += 1

    return {
        "eval_samples": float(total),
        "rulebook_compliance_rate": (compliant / total) if total else 0.0,
        "finding_mention_rate": (non_normal_recall_hits / non_normal_total) if non_normal_total else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmolVLM-500M with Multi-Dataset Rulebook CoT LoRA.")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./.tmp/rulebook_lora_ckpt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--quantize-4bit", action="store_true", help="Enable NF4 QLoRA (CUDA GPU only).")
    parser.add_argument("--adapt-vision", action="store_true", help="Adapt ViT out_proj and modality connector.")
    parser.add_argument("--dry-run-steps", type=int, default=0, help="Stop after N optimizer steps for smoke testing.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # float32 keeps MPS LoRA backward gradients numerically stable
    else:
        device = "cpu"
        dtype = torch.float32

    if args.quantize_4bit and device != "cuda":
        raise RuntimeError("--quantize-4bit requires a CUDA GPU with bitsandbytes.")

    try:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=False)

    processor.image_processor.do_image_splitting = False
    pad_id = int(processor.tokenizer.pad_token_id or 0)

    train_ds = RulebookCXRDataset(args.train_jsonl, processor=processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_single_or_padded(b, pad_id),
    )

    model = build_lora_model(
        args.base_model,
        use_4bit=args.quantize_4bit,
        adapt_vision=args.adapt_vision,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
        dtype=dtype,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.05 * total_steps)), num_training_steps=total_steps)

    model.train()
    opt_steps = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {
                k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += float(outputs.loss.item())

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_steps += 1
                print(f"[Epoch {epoch+1}/{args.epochs} | Step {opt_steps}] loss={running_loss / min(step, args.grad_accum):.4f}")
                running_loss = 0.0

                if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
                    break
        if args.dry_run_steps > 0 and opt_steps >= args.dry_run_steps:
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    with open(out_dir / "rulebook_v2.json", "w", encoding="utf-8") as mf:
        json.dump({"rulebook_format": "5-rule-system-v2", "adapt_vision": args.adapt_vision}, mf, indent=2)
    print(f"[Trainer] Saved LoRA adapter + processor to {out_dir}")

    metrics = evaluate_validation_compliance(
        model,
        processor,
        val_jsonl=args.val_jsonl,
        device=device,
        dtype=dtype,
        max_eval=2 if args.dry_run_steps > 0 else 20,
    )
    print(f"[Trainer] Validation Audit Metrics: {metrics}")


if __name__ == "__main__":
    main()
