"""
ChestXStudio-VLM: Unified Inference & Anti-Hallucination Rulebook Auditor
Path: execution/analyze_xray.py

Layer 3 deterministic execution module combining:
  - Stage 1: 4-Model Multi-Resolution Ensemble CAD + 16x16 Spatial CAM (ensemble_cad.py)
  - Stage 2: SmolVLM-500M-Instruct + QLoRA Medical Adapter (./chestxray_smolvlm_lora)
  - Clause-level Anti-Hallucination Sentence Guardrail & 5-Rule Clinical Rulebook Synthesis
"""

from __future__ import annotations

import gc
import glob
import json
import os
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from execution.ensemble_cad import (
    CLASS_CALIBRATION_GATES,
    evaluate_single_image,
    load_multi_resolution_ensemble,
    resolve_cnn_device,
)
from execution.preprocess import normalize_cxr_to_rgb

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_ADAPTER_PATH = "./chestxray_smolvlm_lora"

SYSTEM_RULEBOOK = (
    "You are an expert diagnostic chest X-ray AI radiologist following the 5-Rule Clinical Rulebook:\n"
    "1. Systematic Search: Evaluate airways, lungs/pleura, mediastinum, heart, and bones.\n"
    "2. Observation Before Etiology: Describe physical visual findings FIRST before stating the diagnosis.\n"
    "3. Spatial Grounding: Include the exact anatomical location for every positive finding.\n"
    "4. Explicit Negatives: State the absence of unmentioned critical acute pathologies "
    "(pneumothorax, pleural effusion, focal consolidation, pulmonary edema).\n"
    "5. Uncertainty Calibration: Use calibrated diagnostic phrasing when findings are mild or overlapping.\n\n"
    "Format the report in these exact 3 sections:\n"
    "1. Systematic Visual Findings\n"
    "2. Spatial Localization\n"
    "3. Clinical Impression"
)

SYSTEM_RULEBOOK_V3 = (
    "You are an expert diagnostic chest X-ray AI radiologist.\n"
    "Examine all airways, lungs, pleura, mediastinum, cardiac silhouette, and bones carefully.\n"
    "Report every visual abnormality concisely with its exact anatomical zone using this exact 2-line format:\n"
    "FINDINGS: <semicolon-separated \"Clinical Name (zone)\" list, or NONE>\n"
    "IMPRESSION: <1-2 concise clinical sentences, or \"No acute cardiopulmonary abnormality identified on this frontal chest radiograph.\">"
)

ALL_PATHOLOGY_STEMS: dict[str, list[str]] = {
    "Pneumothorax": ["pneumothora"],
    "Effusion": ["effusion", "pleural fluid", "costophrenic blunting", "meniscus"],
    "Consolidation": ["consolidat"],
    "Pneumonia": ["pneumonia"],
    "Edema": ["edema", "congest"],
    "Atelectasis": ["atelecta", "volume loss"],
    "Cardiomegaly": ["cardiomegal", "enlarged heart", "enlarged cardiac"],
    "Enlarged Cardiomediastinum": ["widened mediastin", "enlarged cardiomediastin"],
    "Lung Opacity": ["opacit"],
    "Infiltration": ["infiltrat"],
    "Mass": ["mass"],
    "Nodule": ["nodul"],
    "Lung Lesion": ["lesion"],
    "Emphysema": ["emphysem", "hyperinflat"],
    "Fibrosis": ["fibros", "reticular"],
    "Pleural_Thickening": ["pleural thickening"],
    "Fracture": ["fractur"],
    "Hernia": ["hernia"],
}

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,|;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\bwhile\b|\band\b)\s*",
    flags=re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|negative for|free of|absence of|absent|clear of|"
    r"unremarkable|within normal limits|normal|ruled out|no evidence of)\b.*",
    flags=re.IGNORECASE,
)
_NORMAL_BIAS_CLAIM_RE = re.compile(
    r"\b(?:no acute cardiopulmonary|no acute disease|no acute abnormality|"
    r"lungs are clear|lung zones are clear|clear bilaterally|clear without|"
    r"normal chest radiograph|unremarkable chest)\b",
    flags=re.IGNORECASE,
)
_PROMPT_ECHO_RE = re.compile(
    r"(?:systematic search:|observation before etiology:|spatial grounding:|"
    r"explicit negatives:|uncertainty calibration:|evaluate airways|"
    r"describe physical visual|verified ensemble cad|format the report|"
    r"clinical rulebook|analyze this chest x-ray step-by-step|"
    r"^\s*step\s+\d+\s*:|\bexamine the\b|\bobserve the\b|\blook at the\b|\bevaluate the\b)",
    flags=re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(
    r"^(?:[12345]\.|#|\*\*|visual findings:|clinical impression)",
    flags=re.IGNORECASE,
)


def resolve_llm_device(
    preferred: str = "auto",
    allow_mps_bf16: bool = False,
) -> tuple[str, torch.dtype]:
    """
    Selects optimal device and safe dtype.
    Apple Silicon MPS defaults strictly to float32 to prevent float16/bfloat16
    logit instability; bfloat16 on MPS requires explicit `allow_mps_bf16=True`.
    """
    if preferred == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = preferred

    if device == "mps":
        return device, (torch.bfloat16 if allow_mps_bf16 else torch.float32)
    if device == "cuda":
        return device, torch.float16
    return device, torch.float32


def _load_hf_with_offline_fallback(loader_fn, model_id: str, **kwargs):
    """Tries local Hugging Face cache first to avoid DNS timeouts when offline."""
    try:
        return loader_fn(model_id, local_files_only=True, **kwargs)
    except Exception:
        return loader_fn(model_id, local_files_only=False, **kwargs)


def filter_unverified_vlm_sentences(
    vlm_text: str,
    allowed_raw_labels: set[str],
) -> list[str]:
    """
    Clause-level anti-hallucination guardrail:
      - Splits sentences into clauses BEFORE applying negation masking, so a normal
        first clause ('Heart is normal, and there is pneumothorax') cannot mask an
        unverified pathology in a later clause.
      - Rejects contradicting normal-bias sentences when CAD confirmed abnormalities.
    """
    if not vlm_text:
        return []

    allowed_stems: set[str] = set()
    for lbl in allowed_raw_labels:
        allowed_stems.update(ALL_PATHOLOGY_STEMS.get(lbl, []))
        if lbl in ("Consolidation", "Pneumonia", "Atelectasis", "Edema", "Infiltration"):
            allowed_stems.add("opacit")
        if lbl in ("Mass", "Nodule"):
            allowed_stems.add("lesion")

    kept: list[str] = []
    for raw_sent in re.split(r"(?<=[.!?])\s+", vlm_text):
        s_clean = raw_sent.strip()
        if not s_clean or len(s_clean) <= 15:
            continue
        if _SECTION_HEADER_RE.match(s_clean) or _PROMPT_ECHO_RE.search(s_clean):
            continue

        # Block "normal chest" claims when CAD verified real pathology above Dual-Gate
        if allowed_raw_labels and _NORMAL_BIAS_CLAIM_RE.search(s_clean):
            continue

        hallucinated = False
        contradicts_cad = False
        for clause in _CLAUSE_SPLIT_RE.split(s_clean.lower()):
            clause = clause.strip()
            if not clause:
                continue
            neg_match = _NEGATION_RE.search(clause)
            if neg_match and allowed_raw_labels:
                negated_part = clause[neg_match.start():]
                if any(st in negated_part for st in allowed_stems):
                    contradicts_cad = True
                    break
            unnegated = _NEGATION_RE.sub(" ", clause)
            for _, stems in ALL_PATHOLOGY_STEMS.items():
                if any(st in unnegated for st in stems) and not any(st in allowed_stems for st in stems):
                    hallucinated = True
                    break
            if hallucinated:
                break

        if not hallucinated and not contradicts_cad:
            kept.append(s_clean.rstrip("."))

    return kept[:2]


def synthesize_audited_rulebook_report(
    vlm_raw_text: str,
    cad_findings: list[dict[str, Any]],
    total_cad_models: int,
    use_vlm_commentary: bool = True,
) -> str:
    """
    Constructs a strict 5-Rule Clinical Rulebook report grounded on verified Stage 1
    CAD observations, spatial CAM localization, and audited VLM commentary.
    """
    detected_raw_labels = {d["raw_label"] for d in cad_findings}

    acute_checks = [
        ({"Pneumothorax"}, "pneumothorax"),
        ({"Effusion"}, "pleural effusion"),
        ({"Consolidation", "Pneumonia", "Infiltration", "Lung Opacity"}, "focal lobar consolidation"),
        ({"Edema"}, "pulmonary edema"),
        ({"Fracture"}, "acute osseous fracture"),
    ]
    explicit_negatives = [
        neg_name for label_set, neg_name in acute_checks
        if not (label_set & detected_raw_labels)
    ]
    neg_sentence = (
        f"No radiographic evidence of {', '.join(explicit_negatives)}."
        if explicit_negatives else ""
    )

    verified_vlm = (
        filter_unverified_vlm_sentences(vlm_raw_text, detected_raw_labels)
        if use_vlm_commentary and vlm_raw_text
        else []
    )

    if not cad_findings:
        base_obs = [
            "Trachea and central airways are midline and patent",
            "Bilateral lung fields are well-aerated and clear without focal parenchymal opacity",
            "Cardiomediastinal silhouette is within normal limits and bilateral costophrenic angles are sharp",
        ]
        if verified_vlm:
            base_obs.extend(verified_vlm[:1])
        sec1 = ". ".join(base_obs) + ". " + neg_sentence
        sec2 = "Bilateral lung parenchyma, costophrenic angles, pleura, and cardiomediastinal silhouette."
        sec3 = "No acute cardiopulmonary process identified on frontal chest radiograph."
    else:
        obs_sentences: list[str] = []
        loc_entries: list[str] = []
        imp_entries: list[str] = []

        for idx, item in enumerate(cad_findings, start=1):
            loc_lower = item["location"][0].lower() + item["location"][1:]
            obs_sentences.append(
                f"Visual evaluation demonstrates {item['observation']} localized to the {loc_lower}"
            )
            if item["location"] not in loc_entries:
                loc_entries.append(item["location"])

            cal_p = item["cal_prob"]
            raw_p = item["raw_prob"]
            if cal_p >= 0.72 and raw_p >= 0.40:
                cal_phrase = "Findings are strongly consistent with"
            elif cal_p >= 0.62:
                cal_phrase = "Radiographic appearance is consistent with"
            else:
                cal_phrase = "Mild or borderline radiographic features suggestive of"

            imp_entries.append(
                f"{idx}. {cal_phrase} {item['clinical_name']} involving the {loc_lower} "
                f"(Ensemble Calibrated: {cal_p:.0%} | Raw Sigmoid: {raw_p:.0%})"
            )

        if not ({"Cardiomegaly", "Enlarged Cardiomediastinum"} & detected_raw_labels):
            obs_sentences.append("The cardiac silhouette and central mediastinal contours are within normal limits")

        if verified_vlm:
            obs_sentences.extend(verified_vlm[:1])

        sec1 = ". ".join(obs_sentences) + ". " + neg_sentence
        sec2 = "; ".join(loc_entries) + "."
        sec3 = "\n".join(imp_entries) + "\nClinical correlation is recommended."

    ensemble_header = (
        f"[Multi-Resolution Ensemble ({total_cad_models} Models: 224x224 + 512x512) + Quorum Dual-Gate Audit]\n"
        + (
            "Status: STRICT NORMAL (All 18 pathologies below Dual-Gate & Quorum threshold)"
            if not cad_findings
            else " | ".join(
                f"{d['clinical_name'].title()}: {d['cal_prob']:.0%} cal / {d['raw_prob']:.0%} raw ({d['location']})"
                for d in cad_findings
            )
        )
    )

    return (
        f"{ensemble_header}\n\n"
        f"1. Systematic Visual Findings:\n{sec1.strip()}\n\n"
        f"2. Spatial Localization:\n{sec2.strip()}\n\n"
        f"3. Clinical Impression:\n{sec3.strip()}"
    )


class ChestXRayPipeline:
    """
    Unified ChestXStudio Inference Pipeline.
    Supports:
      - Stage 1 CAD Ensemble (4 multi-resolution models + 16x16 CAM)
      - Stage 2 SmolVLM-500M-Instruct + QLoRA medical adapter (do_image_splitting=False)
      - Audited Rulebook synthesis or standalone Pure-VLM mode
    """

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        use_lora: bool = True,
        load_cad: bool = True,
        cnn_device: str = "auto",
        llm_device: str = "auto",
        allow_mps_bf16: bool = False,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.use_lora = use_lora and os.path.isdir(adapter_path)
        self.is_rulebook_v3 = os.path.isfile(os.path.join(adapter_path, "rulebook_v3.json"))
        self.is_rulebook_v2 = os.path.isfile(os.path.join(adapter_path, "rulebook_v2.json"))
        self.is_rulebook_adapted = self.is_rulebook_v3 or self.is_rulebook_v2

        self.cnn_device = resolve_cnn_device(cnn_device)
        self.llm_device, self.llm_dtype = resolve_llm_device(llm_device, allow_mps_bf16=allow_mps_bf16)

        self.ensemble_224: list[dict[str, Any]] = []
        self.resnet_512: dict[str, Any] | None = None
        if load_cad:
            self.ensemble_224, self.resnet_512 = load_multi_resolution_ensemble(device_str=self.cnn_device)

        self.processor = None
        self.llm_model = None
        self._load_vlm()

    def _resolve_image_splitting_config(self) -> tuple[bool, int]:
        """Reads v3/v2 adapter metadata for image splitting (defaults False for v1/v2 backward compatibility)."""
        if not self.use_lora:
            return False, 512
        for fname in ("rulebook_v3.json", "rulebook_v2.json"):
            meta_path = os.path.join(self.adapter_path, fname)
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as mf:
                    meta = json.load(mf)
                split_enabled = bool(meta.get("do_image_splitting", False))
                img_size = int(meta.get("image_size", 512) or 512)
                longest_edge = int(meta.get("image_longest_edge") or (2 * img_size if split_enabled else img_size))
                return split_enabled, longest_edge
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return False, 512

    def _load_vlm(self) -> None:
        print(f"[VLM Engine] Loading {self.base_model_id} on {self.llm_device.upper()} ({self.llm_dtype})...")
        proc_source = self.adapter_path if self.use_lora else self.base_model_id
        try:
            self.processor = _load_hf_with_offline_fallback(AutoProcessor.from_pretrained, proc_source)
        except Exception:
            self.processor = _load_hf_with_offline_fallback(AutoProcessor.from_pretrained, self.base_model_id)

        self.do_image_splitting, self.image_longest_edge = self._resolve_image_splitting_config()
        self.processor.image_processor.do_image_splitting = self.do_image_splitting
        if self.do_image_splitting and hasattr(self.processor.image_processor, "size"):
            self.processor.image_processor.size = {"longest_edge": self.image_longest_edge}
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        base_model = _load_hf_with_offline_fallback(
            AutoModelForImageTextToText.from_pretrained,
            self.base_model_id,
            dtype=self.llm_dtype,
            low_cpu_mem_usage=True,
        )

        if self.use_lora:
            print(f"[VLM Engine] Attaching QLoRA medical adapter: {self.adapter_path}")
            model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            model = base_model

        self.llm_model = model.to(device=self.llm_device, dtype=self.llm_dtype)
        self.llm_model.eval()
        print("[VLM Engine] SmolVLM ready.")

    def _build_vlm_inputs(
        self,
        image: Image.Image | str | os.PathLike[str],
        user_prompt: str,
    ) -> dict[str, torch.Tensor]:
        image_rgb = normalize_cxr_to_rgb(image, image_size=512)
        if self.is_rulebook_v3:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK_V3}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            ]
        elif self.is_rulebook_v2 or not self.use_lora:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_RULEBOOK}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            ]
        else:
            # OpenI v1 adapter was trained without a system role; match its native user prompt
            # to avoid system-prompt echo while preserving systematic findings -> impression order.
            native_prompt = (
                f"{user_prompt}\n"
                "Examine this frontal chest X-ray. Provide systematic physical findings first, "
                "followed by the clinical etiology and impression."
            )
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": native_prompt}]},
            ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=[prompt], images=[[image_rgb]], return_tensors="pt")
        return {
            k: v.to(device=self.llm_device, dtype=self.llm_dtype) if v.is_floating_point() else v.to(device=self.llm_device)
            for k, v in inputs.items()
        }

    def _run_vlm_generate(
        self,
        image: Image.Image | str | os.PathLike[str],
        user_prompt: str,
        max_new_tokens: int = 384,
        return_scores: bool = False,
    ) -> tuple[str, tuple[torch.Tensor, ...] | None]:
        inputs = self._build_vlm_inputs(image, user_prompt)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "repetition_penalty": 1.08 if self.is_rulebook_adapted else 1.15,
            }
            if return_scores:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True
            out = self.llm_model.generate(**inputs, **gen_kwargs)

        if return_scores:
            seq = out.sequences[0]
            scores = tuple(s.detach().cpu() for s in (out.scores or ()))
        else:
            seq = out[0]
            scores = None

        decoded = self.processor.decode(seq[input_len:], skip_special_tokens=True).strip()

        del inputs, out
        gc.collect()
        if self.llm_device == "mps":
            torch.mps.empty_cache()
        elif self.llm_device == "cuda":
            torch.cuda.empty_cache()
        return decoded, scores

    def _generate_vlm_text(
        self,
        image: Image.Image | str | os.PathLike[str],
        user_prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        decoded, _ = self._run_vlm_generate(image, user_prompt, max_new_tokens=max_new_tokens, return_scores=False)
        return decoded

    def analyze(
        self,
        image: Image.Image,
        sensitivity_mode: str = "Balanced Clinical (Recommended)",
        use_vlm: bool = True,
        mode: str = "audited",
    ) -> dict[str, Any]:
        """
        Runs end-to-end chest radiograph interpretation.

        Args:
            image: PIL.Image input
            sensitivity_mode: Screening sensitivity mode string
            use_vlm: Whether to invoke SmolVLM commentary in audited mode
            mode: 'audited' (Stage 1 CAD + Stage 2 LoRA VLM + Guardrail) or 'pure_vlm'
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        t0 = time.perf_counter()

        if mode == "pure_vlm" or not self.ensemble_224:
            if self.is_rulebook_v3:
                pure_prompt = "Inspect this frontal chest X-ray and report FINDINGS and IMPRESSION."
                pure_tokens = 128
            else:
                pure_prompt = "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
                pure_tokens = 384 if self.is_rulebook_v2 else 250
            raw_report = self._generate_vlm_text(
                image,
                pure_prompt,
                max_new_tokens=pure_tokens,
            )
            return {
                "report": raw_report,
                "overlay_img": image.convert("RGB"),
                "telemetry_rows": [],
                "cad_findings": [],
                "near_misses": [],
                "latency_seconds": round(time.perf_counter() - t0, 2),
                "mode": "pure_vlm",
            }

        if "High Sensitivity" in sensitivity_mode:
            sens_offset = 0.04
        elif "High Specificity" in sensitivity_mode:
            sens_offset = -0.05
        else:
            sens_offset = 0.0

        cad_res = evaluate_single_image(
            image,
            ensemble_224=self.ensemble_224,
            resnet_512=self.resnet_512,
            sensitivity_offset=sens_offset,
            device_str=self.cnn_device,
        )
        cad_findings = cad_res["top_findings"]

        raw_vlm_text = ""
        if use_vlm:
            if self.is_rulebook_v3:
                if not cad_findings:
                    cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold"
                else:
                    cad_summary = "; ".join(
                        "{} at {}".format(d["clinical_name"], d["location"])
                        for d in cad_findings[:4]
                    )
                user_prompt = (
                    f"Candidate Ensemble CAD findings: {cad_summary}.\n"
                    "Inspect this frontal chest X-ray, authenticate the CAD alerts, and report FINDINGS and IMPRESSION."
                )
                raw_vlm_text = self._generate_vlm_text(image, user_prompt, max_new_tokens=128)
            elif self.is_rulebook_v2:
                if not cad_findings:
                    cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold"
                else:
                    cad_summary = "; ".join(
                        "{} at {}".format(d["clinical_name"], d["location"])
                        for d in cad_findings[:4]
                    )
                user_prompt = (
                    f"Candidate Ensemble CAD findings: {cad_summary}.\n"
                    "Authenticate these CAD alerts against the radiograph and analyze this chest X-ray "
                    "step-by-step according to the 5-Rule Clinical Rulebook."
                )
                raw_vlm_text = self._generate_vlm_text(image, user_prompt, max_new_tokens=384)
            else:
                if not cad_findings:
                    cad_summary = "No acute cardiopulmonary pathology detected above Dual-Gate threshold."
                else:
                    cad_summary = "; ".join(
                        "{} at {} (Calibrated: {:.0%})".format(d["clinical_name"], d["location"], d["cal_prob"])
                        for d in cad_findings
                    )
                user_prompt = (
                    f"Verified Ensemble CAD findings: {cad_summary}\n"
                    "Analyze this chest X-ray step-by-step according to the 5-Rule Clinical Rulebook."
                )
                raw_vlm_text = self._generate_vlm_text(image, user_prompt, max_new_tokens=160)

        total_models = len(self.ensemble_224) + (1 if self.resnet_512 is not None else 0)
        final_report = synthesize_audited_rulebook_report(
            raw_vlm_text,
            cad_findings,
            total_cad_models=total_models,
            use_vlm_commentary=use_vlm,
        )

        return {
            "report": final_report,
            "overlay_img": cad_res["overlay_img"],
            "telemetry_rows": cad_res["telemetry_rows"],
            "cad_findings": cad_findings,
            "near_misses": cad_res["near_misses"],
            "latency_seconds": round(time.perf_counter() - t0, 2),
            "mode": "audited",
        }


def find_default_xray(search_dir: str = ".") -> str | None:
    """Scans workspace for a sample frontal X-ray image (excluding caches/envs)."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(search_dir, "**", pattern), recursive=True))
    valid_images = sorted(
        f for f in found
        if ".venv" not in f and ".tmp" not in f and "ensemble_audit_outputs" not in f
    )
    return valid_images[0] if valid_images else None
