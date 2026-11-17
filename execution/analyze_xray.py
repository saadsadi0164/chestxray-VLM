"""
ChestXStudio-VLM: Core Deterministic Execution Module
Path: execution/analyze_xray.py

Layer 3 tool responsible for model lifecycle, device placement,
prompt structuring according to clinical rulebook, and token generation.
"""

import os
import glob
import time
from typing import Optional, Dict, Any, Tuple
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Default configurations
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_ADAPTER_PATH = "./chestxray_smolvlm_lora"

# Clinical reasoning rulebook aligned with fine-tuning training distribution
SYSTEM_RULEBOOK = (
    "You are an expert diagnostic chest X-ray AI assistant following a strict clinical rulebook:\n"
    "1. Systematic Search: Evaluate airways, lungs/pleura, mediastinum, heart, and bones.\n"
    "2. Observation Before Etiology: Describe physical visual findings first before stating the etiology/impression.\n"
    "3. Spatial Grounding: Specify exact anatomical locations for all positive findings.\n"
    "4. Explicit Negatives: Explicitly state the absence of critical acute pathologies.\n"
    "5. Uncertainty Calibration: Use calibrated diagnostic phrasing when multiple etiologies fit."
)

CLINICAL_PROMPT = (
    "Examine this frontal chest X-ray. Provide systematic physical findings first, "
    "followed by the clinical etiology and impression."
)

LAYPERSON_PROMPT = (
    "Examine this frontal chest X-ray following the clinical rulebook. "
    "Provide a structured, patient-accessible summary with the following sections:\n"
    "1. Systematic Physical Findings\n"
    "2. Anatomical Localization\n"
    "3. Clinical Impression & Next Steps (Educational Only)"
)


class ChestXRayEngine:
    def __init__(self, base_model_id: str = DEFAULT_BASE_MODEL, adapter_path: str = DEFAULT_ADAPTER_PATH):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.device, self.dtype = self._detect_hardware()
        
        self.processor = None
        self.model = None
        self._load_pipeline()

    @staticmethod
    def _detect_hardware() -> Tuple[str, torch.dtype]:
        """Dynamically detect optimal device and safe precision."""
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif torch.backends.mps.is_available():
            return "mps", torch.float16
        else:
            return "cpu", torch.float32

    def _load_pipeline(self):
        """Loads processor, base vision-language model, and attaches LoRA adapter."""
        if not os.path.exists(self.adapter_path):
            raise FileNotFoundError(f"Adapter directory not found at: {self.adapter_path}")

        print(f"[Engine] Initializing on {self.device.upper()} with dtype {self.dtype}...")

        # Load processor from adapter path (preserves customized special tokens and chat templates)
        try:
            self.processor = AutoProcessor.from_pretrained(self.adapter_path)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.base_model_id)

        # Load base model with device-safe precision
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )

        # Attach LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[Engine] Model and adapter loaded successfully.")

    def analyze(
        self,
        image: Image.Image,
        mode: str = "clinical",
        max_new_tokens: int = 250,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes inference on a PIL Image with deterministic token slicing.
        
        Args:
            image: PIL.Image instance
            mode: 'clinical' (radiology report) or 'layperson' (structured patient breakdown)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy deterministic generation)
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        rgb_image = image.convert("RGB")
        user_prompt = LAYPERSON_PROMPT if mode == "layperson" else CLINICAL_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{SYSTEM_RULEBOOK}\n\n{user_prompt}"}
                ]
            }
        ]

        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_str, images=[rgb_image], return_tensors="pt")

        # Route tensors to target device and match model float precision
        inputs = {
            k: v.to(self.device).to(self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        input_token_len = inputs["input_ids"].shape[1]
        start_time = time.time()

        do_sample = temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        # Slicing: decode ONLY the newly generated tokens to eliminate prompt leakage
        generated_ids = outputs[0][input_token_len:]
        report_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        elapsed_sec = round(time.time() - start_time, 2)

        return {
            "report": report_text,
            "latency_seconds": elapsed_sec,
            "device": self.device,
            "mode": mode
        }


def find_default_xray(search_dir: str = ".") -> Optional[str]:
    """Scans for valid X-ray image files in the directory."""
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.dcm.png"]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(search_dir, "**", pattern), recursive=True))
    # Exclude checkpoints or cache
    valid_images = [f for f in found if ".venv" not in f and ".tmp" not in f]
    return valid_images[0] if valid_images else None
