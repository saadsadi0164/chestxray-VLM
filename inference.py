import os
import glob
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

print("--- CHEST X-RAY VLM LOCAL DIAGNOSTIC AUDIT ---")
current_dir = os.getcwd()
print(f"1. Working Directory: {current_dir}")

# Auto-locate the unzipped adapter folder (looks for adapter_config.json)
adapter_path = None
for root, dirs, files in os.walk(current_dir):
    if "adapter_config.json" in files:
        adapter_path = root
        break

# Auto-locate any .png or .jpg chest X-ray in the folder
image_files = glob.glob(os.path.join(current_dir, "**", "*.png"), recursive=True) + glob.glob(os.path.join(current_dir, "**", "*.jpg"), recursive=True)

if not adapter_path:
    print("ERROR: Could not find adapter_config.json inside this folder!")
    print("Files currently in this folder:", os.listdir(current_dir))
    exit(1)

if not image_files:
    print("ERROR: Could not find any .png or .jpg X-ray image inside this folder!")
    print("Files currently in this folder:", os.listdir(current_dir))
    exit(1)

image_path = "4_IM-2050-1001.dcm.png"
print(f"2. Found Trained Adapter: {adapter_path}")
print(f"3. Found Test X-Ray Image: {image_path}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"4. Hardware Acceleration: {device.upper()} ({dtype})")

base_model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
print("\nLoading processor & base model (first run downloads ~1GB base weights to Mac cache)...")
processor = AutoProcessor.from_pretrained(adapter_path)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=dtype
)

print("Attaching your trained QLoRA medical adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()

image = Image.open(image_path).convert("RGB")

SYSTEM_RULEBOOK = (
    "You are an expert diagnostic chest X-ray AI assistant following a strict clinical rulebook:\n"
    "1. Systematic Search: Evaluate airways, lungs/pleura, mediastinum, heart, and bones.\n"
    "2. Observation Before Etiology: Describe physical visual findings first before stating the etiology/impression.\n"
    "3. Spatial Grounding: Specify exact anatomical locations for all positive findings.\n"
    "4. Explicit Negatives: Explicitly state the absence of critical acute pathologies.\n"
    "5. Uncertainty Calibration: Use calibrated diagnostic phrasing when multiple etiologies fit."
)
USER_PROMPT = "Examine this frontal chest X-ray. Provide systematic physical findings first, followed by the clinical etiology and impression."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{SYSTEM_RULEBOOK}\n\n{USER_PROMPT}"}
        ]
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\nAnalyzing Chest X-Ray...")
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
final_report = generated_text.split("Assistant:")[-1].strip() if "Assistant:" in generated_text else generated_text.split(USER_PROMPT)[-1].strip()

print(f"Analysis complete in {round(time.time() - start_time, 1)} seconds.\n")
print("=" * 60)
print("LOCAL AI RADIOLOGY REPORT:")
print("=" * 60)
print(final_report)
print("=" * 60)
