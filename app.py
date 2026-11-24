"""
ChestXStudio: 4-Model Multi-Resolution Ensemble (224 + 512) + Dual-Gate CAM + SmolVLM LoRA
Path: app.py

Thin Gradio 6.0 UI layer delegating all deterministic CAD, CAM, and VLM inference
to `execution/analyze_xray.py` and `execution/ensemble_cad.py`.
"""

from __future__ import annotations

import os
import socket
import sys
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from execution.analyze_xray import ChestXRayPipeline, find_default_xray

# Initialize unified Stage 1 (CAD Ensemble) + Stage 2 (SmolVLM-500M + LoRA) pipeline
pipeline = ChestXRayPipeline(load_cad=True, use_lora=True)


def ui_analyze(
    image: Image.Image | None,
    sensitivity_mode: str,
    pipeline_mode: str,
    use_vlm: bool,
) -> tuple[str, Image.Image | None, list[list[str]]]:
    if image is None:
        return "Please upload a frontal chest X-ray image (PA or AP).", None, []

    try:
        mode_key = "pure_vlm" if "Pure VLM" in pipeline_mode else "audited"
        res = pipeline.analyze(
            image=image,
            sensitivity_mode=sensitivity_mode,
            use_vlm=use_vlm,
            mode=mode_key,
        )
        return res["report"], res["overlay_img"], res["telemetry_rows"]
    except Exception as exc:
        return f"Analysis error: {exc}", None, []


def find_available_port(preferred_port: int = 7860) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


with gr.Blocks(title="ChestXStudio Clinical Workstation") as interface:
    gr.Markdown("# ChestXStudio: Multi-Resolution (224 + 512) + SmolVLM LoRA Workstation")
    gr.Markdown(
        "**Stage 1 (Multi-Resolution CAD):** `DenseNet121-All (224)` + `DenseNet121-CheXpert (224)` + "
        "`DenseNet121-MIMIC (224)` + `ResNet50-All (512x512)` with 3-View TTA & Quorum Dual-Gate  \n"
        "**Stage 2 (Spatial Grounding & VLM Auditor):** 16x16 Multi-Res CAM + "
        "`SmolVLM-500M-Instruct` (`chestxray_smolvlm_lora` QLoRA) + Clause-Level Anti-Hallucination Guardrail"
    )
    gr.Markdown(
        "> *Research & Educational Use Only — Not an FDA/CE-cleared medical device. "
        "Must never replace clinical interpretation by a licensed radiologist or physician.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="1. Upload Frontal Chest X-Ray (PA or AP)")
            pipeline_radio = gr.Radio(
                choices=[
                    "Audited Hybrid (4-Model CAD + SmolVLM LoRA - Recommended)",
                    "Pure VLM (SmolVLM LoRA Standalone CoT)",
                ],
                value="Audited Hybrid (4-Model CAD + SmolVLM LoRA - Recommended)",
                label="2. Inference Architecture Mode",
            )
            sensitivity_radio = gr.Radio(
                choices=[
                    "High Sensitivity (Screening - catches subtle findings)",
                    "Balanced Clinical (Recommended)",
                    "High Specificity (Strict - suppresses borderline findings)",
                ],
                value="Balanced Clinical (Recommended)",
                label="3. Operating Threshold Calibration",
            )
            use_vlm_checkbox = gr.Checkbox(
                value=True,
                label="Enable SmolVLM-500M LoRA Commentary (Audited by Clause Guardrail)",
            )
            submit_btn = gr.Button("Analyze Radiograph", variant="primary", size="lg")

            sample_xray = find_default_xray()
            if sample_xray:
                gr.Examples(examples=[[sample_xray]], inputs=[image_input])

        with gr.Column(scale=1):
            cam_output = gr.Image(type="pil", label="16x16 Multi-Res Class Activation Map (CAM) Overlay")

    with gr.Row():
        with gr.Column(scale=3):
            output_text = gr.Textbox(
                label="5-Rule Clinical Diagnostic Report",
                lines=16,
                placeholder="Upload a chest X-ray and click 'Analyze Radiograph'...",
            )
        with gr.Column(scale=2):
            telemetry_table = gr.Dataframe(
                headers=["Pathology", "Calibrated", "Raw Sigmoid", "Dual-Gate Status", "CAM Location"],
                datatype=["str", "str", "str", "str", "str"],
                label="Full 18-Pathology Dual-Gate Telemetry",
            )

    submit_btn.click(
        fn=ui_analyze,
        inputs=[image_input, sensitivity_radio, pipeline_radio, use_vlm_checkbox],
        outputs=[output_text, cam_output, telemetry_table],
    )


if __name__ == "__main__":
    port = find_available_port(7860)
    print(f"[ChestXStudio] Launching Gradio workstation on http://127.0.0.1:{port}")
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft(),
    )
