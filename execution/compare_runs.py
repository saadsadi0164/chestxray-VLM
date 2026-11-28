"""
ChestXStudio-VLM: Run Comparison Markdown Table Generator
Path: execution/compare_runs.py

Scans benchmark run JSON files (e.g. `.tmp/reports/*.json`) containing `summary`
and `records` and generates a concise one-page Markdown comparison table across
phases, adapters, and training checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIORITY_FINDINGS = [
    "Pneumothorax",
    "Effusion",
    "Consolidation",
    "Cardiomegaly",
    "Edema",
    "Nodule",
    "Mass",
    "Atelectasis",
    "Fracture",
]


def load_benchmark_runs(reports_dir: Path) -> list[dict[str, Any]]:
    """Loads valid benchmark run JSON files sorted by name (ignores manifests/lists)."""
    runs: list[dict[str, Any]] = []
    for json_path in sorted(reports_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if "summary" not in payload or "records" not in payload:
            continue
        runs.append({
            "name": json_path.stem,
            "path": str(json_path),
            "metadata": payload.get("metadata", {}),
            "summary": payload["summary"],
        })
    return runs


def format_comparison_markdown(runs: list[dict[str, Any]]) -> str:
    """Renders a clean one-page Markdown comparison report."""
    if not runs:
        return "No benchmark run JSON files found."

    lines: list[str] = [
        "# ChestXStudio-VLM Benchmark Comparison",
        "",
        "## Overall Metrics",
        "",
        "| Run | Adapter | Device/Dtype | Parser | Studies (Abn/Norm) | Normal-Bias | False-Alarm | Macro Sens | 5-Rule Comp | Combined Score |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        s = run["summary"]
        m = run["metadata"]
        dev_dt = f"{m.get('llm_device', '?')} / {str(m.get('llm_dtype', '?')).replace('torch.', '')}"
        parser = str(m.get("parser", "unknown")).replace("execution.", "")
        abn = s.get("abnormal_studies", 0)
        norm = s.get("normal_studies", 0)
        lines.append(
            f"| `{run['name']}` | `{m.get('adapter', '-')}` | `{dev_dt}` | `{parser}` | "
            f"{s.get('total_studies', 0)} ({abn}/{norm}) | "
            f"{s.get('normal_bias_rate', 0.0):.1%} ({s.get('normal_bias_count', 0)}/{abn}) | "
            f"{s.get('false_alarm_rate', 0.0):.1%} ({s.get('false_alarm_count', 0)}/{norm}) | "
            f"{s.get('macro_sensitivity', 0.0):.1%} | "
            f"{s.get('rulebook_compliance', 0.0):.1%} | "
            f"{s.get('combined_score', 0.0):+.4f} |"
        )

    lines.extend([
        "",
        "## Priority Per-Finding Recall (Sensitivity / F1)",
        "",
        "| Finding | " + " | ".join(f"`{r['name']}`" for r in runs) + " |",
        "|---|" + "|".join("---:" for _ in runs) + "|",
    ])

    for cls in PRIORITY_FINDINGS:
        cells: list[str] = []
        any_present = False
        for run in runs:
            st = run["summary"].get("per_finding", {}).get(cls)
            if st and st.get("support", 0) > 0:
                any_present = True
                cells.append(f"{st['sensitivity']:.1%} / {st['f1']:.3f} (n={st['support']})")
            else:
                cells.append("-")
        if any_present:
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown comparison table from .tmp/reports/*.json.")
    parser.add_argument("--reports-dir", default="./.tmp/reports")
    parser.add_argument("--out", default=None, help="Optional Markdown file path to save output.")
    args = parser.parse_args()

    runs = load_benchmark_runs(Path(args.reports_dir))
    md = format_comparison_markdown(runs)
    print(md)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md, encoding="utf-8")
        print(f"[Compare Runs] Saved comparison Markdown -> {out_p}")


if __name__ == "__main__":
    main()
