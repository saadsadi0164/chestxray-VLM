"""
ChestXStudio-VLM: Negation- & Uncertainty-Aware Radiology Report Parser
Path: execution/report_parser.py

Parses structured or free-text radiology reports into per-class clinical status:
  "present" | "absent" | "uncertain"
for every canonical pathology class in `PATHOLOGY_ONTOLOGY`.

Handles:
  - Clause-level splitting reusing `execution/analyze_xray.py`'s `_CLAUSE_SPLIT_RE`
  - Prepositional negative boundaries ("without", "with no", "negative for", "free of")
  - Comma-separated negative list continuation ("No radiographic evidence of A, B, or C")
  - Contrast conjunctions breaking prior negation ("but", "however", "although", "whereas", "while")
  - Uncertainty / hedging phrases ("suggestive of", "possible", "cannot exclude", etc.)
  - Mode-3 CAD refutation prose & prompt echoes ("Step 1: ...", "CAD flagged suspected ... however, direct visual evaluation...")
  - Aggregate clinical metrics & checkpoint selection `combined_score`.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.analyze_xray import _CLAUSE_SPLIT_RE, _PROMPT_ECHO_RE, _SECTION_HEADER_RE
from execution.build_rulebook_dataset import (
    PATHOLOGY_ONTOLOGY,
    canonicalize_label,
    validate_rulebook_compliance,
)

# Canonical class -> list of regex patterns (word-boundary safe so "mass" never matches "massive")
_CLASS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Pneumothorax": [
        re.compile(r"\bpneumothora(?:x|ces)\b|\bptx\b", re.IGNORECASE),
    ],
    "Effusion": [
        re.compile(
            r"\b(?:pleural\s+)?effusions?\b|\bpleural\s+fluid\b|\bcostophrenic\s+(?:angle\s+)?blunting\b|\bmeniscus\b",
            re.IGNORECASE,
        ),
    ],
    "Consolidation": [
        re.compile(
            r"\bconsolidat\w*\b|\bair\s*space\s+(?:consolidat\w*|disease)\b|\blobar\s+opac\w*\b",
            re.IGNORECASE,
        ),
    ],
    "Pneumonia": [
        re.compile(r"\bpneumonia\b|\binfectious\s+airspace\s+process\b", re.IGNORECASE),
    ],
    "Edema": [
        re.compile(
            r"\b(?:pulmonary\s+)?edema\b|\bvascular\s+congestion\b|\bpulmonary\s+congestion\b|\bcephalization\b|\bkerley\b|\bperihilar\s+hazi\w*\b",
            re.IGNORECASE,
        ),
    ],
    "Atelectasis": [
        re.compile(r"\batelecta\w*\b|\bvolume\s+loss\b", re.IGNORECASE),
    ],
    "Cardiomegaly": [
        re.compile(
            r"\bcardiomegal\w*\b|\benlarged\s+(?:heart|cardiac(?:\s+silhouette)?)\b|\bcardiothoracic\s+ratio\s+(?:exceeding|greater|increased)\b",
            re.IGNORECASE,
        ),
    ],
    "Enlarged Cardiomediastinum": [
        re.compile(
            r"\benlarged\s+cardiomediastin\w*\b|\bwidened\s+(?:superior\s+and\s+central\s+)?(?:cardio)?mediastin\w*\b|\bmediastinal\s+widening\b",
            re.IGNORECASE,
        ),
    ],
    "Aortic Enlargement": [
        re.compile(
            r"\baortic\s+(?:enlargement|tortuos\w*|unfolding|elongation|prominence)\b|\bprominent\s+aort\w*\b|\btortuous\s+aort\w*\b",
            re.IGNORECASE,
        ),
    ],
    "Mediastinal Shift": [
        re.compile(r"\bmediastinal\s+shift\b|\btracheal\s+deviation\b", re.IGNORECASE),
    ],
    "Lung Opacity": [
        re.compile(
            r"\b(?:pulmonary\s+|airspace\s+|parenchymal\s+|focal\s+)?opacit(?:y|ies)\b|\bopacification\b",
            re.IGNORECASE,
        ),
    ],
    "Infiltration": [
        re.compile(r"\binfiltrat\w*\b", re.IGNORECASE),
    ],
    "ILD": [
        re.compile(
            r"\bild\b|\binterstitial\s+lung\s+(?:disease|abnormalit\w*)\b|\breticular\s+opacit\w*\b|\breticulonodular\b",
            re.IGNORECASE,
        ),
    ],
    "Mass": [
        re.compile(r"\b(?:pulmonary\s+|lung\s+)?mass(?:es)?\b", re.IGNORECASE),
    ],
    "Nodule": [
        re.compile(r"\b(?:pulmonary\s+|lung\s+)?nodul\w*\b|(?<!calcified\s)\bgranuloma\w*\b", re.IGNORECASE),
    ],
    "Lung Cavity": [
        re.compile(r"\bcavit\w*\b", re.IGNORECASE),
    ],
    "Lung Cyst": [
        re.compile(r"\b(?:lung\s+|pulmonary\s+)?cysts?\b|\bbull(?:a|ae)\b|\bblebs?\b", re.IGNORECASE),
    ],
    "Lung Lesion": [
        re.compile(r"\b(?:pulmonary\s+|lung\s+|focal\s+)?lesions?\b", re.IGNORECASE),
    ],
    "Emphysema": [
        re.compile(
            r"\bemphysem\w*\b|\bhyperinflat\w*\b|\bcopd\b|\bchronic\s+obstructive\s+pulmonary\b|\bflattened\s+hemidiaphragm\w*\b",
            re.IGNORECASE,
        ),
    ],
    "Fibrosis": [
        re.compile(r"\bfibros\w*\b|\bscarring\b|\bfibrotic\b", re.IGNORECASE),
    ],
    "Pleural Thickening": [
        re.compile(r"\bpleural\s+thickening\b|\bapical\s+cap\b|\bapical\s+pleural\s+thickening\b", re.IGNORECASE),
    ],
    "Calcification": [
        re.compile(r"\bcalcif\w*\b", re.IGNORECASE),
    ],
    "Fracture": [
        re.compile(r"\bfractur\w*\b|\bcortical\s+step[- ]?off\b|\bcortical\s+discontinuit\w*\b", re.IGNORECASE),
    ],
    "Hernia": [
        re.compile(r"\b(?:hiatal\s+)?hernias?\b", re.IGNORECASE),
    ],
    "Support Devices": [
        re.compile(
            r"\bsupport\s+devices?\b|\bpacemaker\b|\bendotracheal\s+tube\b|\bcentral\s+(?:venous\s+)?(?:line|catheter)\b|\bchest\s+tube\b|\bcatheters?\b|\bsternotomy\b|\bprosthes(?:is|es)\b",
            re.IGNORECASE,
        ),
    ],
}

# Prepositional negation cues starting a new sub-clause within an otherwise positive clause
_PREP_NEG_SPLIT_RE = re.compile(
    r"(?=\b(?:without|with\s+no|negative\s+for|free\s+of|absence\s+of|no\s+evidence\s+of|no\s+radiographic\s+evidence\s+of)\b)",
    re.IGNORECASE,
)

# Contrast conjunctions that break any carried-over list negation
_CONTRAST_BREAK_RE = re.compile(
    r"\b(?:but|however|although|whereas|while|yet|despite)\b|;",
    re.IGNORECASE,
)

# Positive / finite assertion verbs indicating a fresh clause rather than list continuation
_NEW_ASSERTION_RE = re.compile(
    r"\b(?:there\s+(?:is|are)|we\s+(?:see|note)|shows?|demonstrates?|reveals?|indicates?|"
    r"consistent\s+with|compatible\s+with|suggestive\s+of|suspicious\s+for|concerning\s+for|"
    r"may\s+represent|might\s+represent|could\s+represent|cannot\s+exclude|cannot\s+be\s+excluded|"
    r"is\s+present|are\s+present|is\s+seen|is\s+noted|is\s+identified|appears\s+as)\b",
    re.IGNORECASE,
)

# Phrases that introduce a comma-separated negative list
_LIST_NEG_TRIGGER_RE = re.compile(
    r"\b(?:no\s+(?:radiographic\s+|visible\s+|obvious\s+|significant\s+|acute\s+|definite\s+|focal\s+)?evidence\s+of|"
    r"without(?:\s+(?:evidence\s+of|signs\s+of))?|"
    r"negative\s+for|free\s+of|absence\s+of|ruled\s+out|rule\s+out|neither|no\s+(?:definite|focal|acute|obvious|visible|significant)?)\b",
    re.IGNORECASE,
)

# Explicit hedging / uncertainty phrases (evaluated before generic negation)
_HEDGE_RE = re.compile(
    r"\b(?:suggestive\s+of|possible|possibly|cannot\s+exclude|cannot\s+be\s+excluded|"
    r"could\s+not\s+exclude|may\s+(?:represent|indicate|reflect|suggest)|"
    r"might\s+(?:represent|indicate|reflect|suggest)|could\s+(?:represent|indicate)|"
    r"suspicious\s+for|concerning\s+for|questionable|equivocal|indeterminate|"
    r"borderline(?:\s+radiographic)?(?:\s+features)?|probable|probably|likely)\b",
    re.IGNORECASE,
)

# Explicit negation markers within a clause
_CLAUSE_NEG_RE = re.compile(
    r"\b(?:no|not|without|negative\s+for|free\s+of|absence\s+of|absent|clear\s+of|"
    r"unremarkable|within\s+normal\s+limits|normal|ruled\s+out|rule\s+out|excluded|"
    r"no\s+evidence\s+of|no\s+radiographic\s+evidence\s+of|not\s+seen|not\s+identified|"
    r"not\s+demonstrated|not\s+appreciated|not\s+present|unlikely|does\s+not\s+substantiate|"
    r"fails\s+to\s+show|clear\s+without|are\s+clear|is\s+clear)\b",
    re.IGNORECASE,
)

NORMAL_CLAIM_PATTERNS = re.compile(
    r"\b(?:no\s+acute\s+(?:cardiopulmonary|thoracic|pulmonary|radiological|radiographic)\s*(?:disease|abnormality|abnormalities|process|findings?)?|"
    r"no\s+active\s+(?:cardiopulmonary\s+)?disease|"
    r"no\s+significant\s+abnormality|"
    r"no\s+obvious\s+abnormality|"
    r"no\s+abnormal\s+findings?|"
    r"normal\s+chest(?:\s+radiograph)?|"
    r"unremarkable\s+(?:chest(?:\s+radiograph)?|for\s+any\s+acute)|"
    r"lungs\s+are\s+clear|"
    r"no\s+cardiopulmonary\s+abnormality)\b",
    re.IGNORECASE,
)

# Strips CAD echo/refutation headers so injected false CAD flags are never miscounted as asserted
_CAD_PROMPT_LINE_RE = re.compile(
    r"^(?:candidate\s+ensemble\s+cad\s+findings|verified\s+ensemble\s+cad\s+findings)\b.*",
    re.IGNORECASE,
)
_CAD_REFUTATION_RE = re.compile(
    r"cad\s+(?:additionally\s+)?flagged\s+suspected\s+[^.;]+(?:[.;]\s*however,?\s+direct\s+visual\s+evaluation[^.]+)?",
    re.IGNORECASE,
)


def _split_atomic_clauses(sentence: str) -> list[tuple[str, bool]]:
    """
    Splits a sentence using `analyze_xray._CLAUSE_SPLIT_RE` (preserving delimiter
    contrast cues) plus prepositional negative sub-clause splits.
    Returns list of (clause_text, breaks_prior_list_negation).
    """
    parts = re.split(
        r"(\s*(?:,|;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\bwhile\b|\band\b)\s*)",
        sentence,
        flags=re.IGNORECASE,
    )
    atomic: list[tuple[str, bool]] = []
    pending_break = False

    for part in parts:
        if not part:
            continue
        if _CLAUSE_SPLIT_RE.fullmatch(part) or re.fullmatch(r"\s*(?:,|;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\bwhile\b|\band\b)\s*", part, re.IGNORECASE):
            if _CONTRAST_BREAK_RE.search(part):
                pending_break = True
            elif re.search(r"\band\b", part, re.IGNORECASE):
                # Keep `and` breaking unless it's `, and` empty or followed by NP list; handled via _NEW_ASSERTION_RE below
                pass
            continue

        sub_chunks = _PREP_NEG_SPLIT_RE.split(part.strip())
        for sub_idx, chunk in enumerate(sub_chunks):
            c = chunk.strip()
            if not c:
                continue
            atomic.append((c, pending_break and sub_idx == 0))
            pending_break = False

    return atomic


def _extract_v3_sections(report_text: str) -> tuple[str | None, str | None]:
    """Extracts FINDINGS and IMPRESSION bodies if present in a v3 structured report."""
    m_f = re.search(r"^\s*FINDINGS\s*:\s*(.*)$", report_text, flags=re.IGNORECASE | re.MULTILINE)
    if not m_f:
        return None, None
    f_body = m_f.group(1).strip()
    m_i = re.search(r"^\s*IMPRESSION\s*:\s*(.+)$", report_text[m_f.end():], flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    i_body = m_i.group(1).strip() if m_i else ""
    return f_body, i_body


def _match_v3_finding_item(item_text: str) -> str | None:
    """Matches one semicolon-delimited v3 FINDING item unambiguously via ontology name first."""
    clean = item_text.strip()
    if not clean:
        return None
    low = clean.lower()
    if low == "none" or _CLAUSE_NEG_RE.search(low) or _HEDGE_RE.search(low):
        return None
    before_paren = clean.split("(", 1)[0].strip()
    canon = canonicalize_label(before_paren)
    if canon and canon in PATHOLOGY_ONTOLOGY:
        return canon
    for cls_name, patterns in _CLASS_PATTERNS.items():
        if any(p.search(before_paren) for p in patterns):
            return cls_name
    return None


def parse_report_findings(report_text: str) -> dict[str, str]:
    """
    Parses a radiology report into per-class clinical assertions:
      `"present" | "absent" | "uncertain"`
    for every canonical pathology class in `PATHOLOGY_ONTOLOGY`.
    Supports both v3 (`FINDINGS: ... / IMPRESSION: ...`) and v1/v2 formats.
    """
    statuses: dict[str, str] = {canon: "absent" for canon in PATHOLOGY_ONTOLOGY}
    if not report_text or not report_text.strip():
        return statuses

    # Track highest-priority assertion observed per class across clauses:
    # present (2) > uncertain (1) > absent (0)
    priority: dict[str, int] = {canon: 0 for canon in PATHOLOGY_ONTOLOGY}
    status_name = {0: "absent", 1: "uncertain", 2: "present"}

    v3_findings, v3_impression = _extract_v3_sections(report_text)
    if v3_findings is not None:
        if v3_findings.strip().upper() != "NONE":
            for part in v3_findings.split(";"):
                low_part = part.strip().lower()
                if not low_part:
                    continue
                if _HEDGE_RE.search(low_part):
                    before_p = part.split("(", 1)[0].strip()
                    canon_h = canonicalize_label(before_p)
                    if not canon_h:
                        for c_name, pats in _CLASS_PATTERNS.items():
                            if any(p.search(before_p) for p in pats):
                                canon_h = c_name
                                break
                    if canon_h and priority[canon_h] < 1:
                        priority[canon_h] = 1
                else:
                    matched_canon = _match_v3_finding_item(part)
                    if matched_canon:
                        priority[matched_canon] = 2

        text_to_scan = v3_impression or ""
    else:
        text_to_scan = report_text

    if text_to_scan:
        # Strip any FINDINGS: line already handled above if present
        for raw_line in re.split(r"(?<=[.!?])\s+|\n+", text_to_scan):
            line = raw_line.strip()
            if not line or re.match(r"^\s*FINDINGS\s*:", line, flags=re.IGNORECASE):
                continue
            line = re.sub(r"^\s*IMPRESSION\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if not line:
                continue
            if _SECTION_HEADER_RE.match(line) or _PROMPT_ECHO_RE.search(line) or _CAD_PROMPT_LINE_RE.match(line):
                continue

            # Remove Mode-3 CAD refutation text so the refuted fake CAD term isn't scored as present
            line = _CAD_REFUTATION_RE.sub(" ", line).strip()
            if not line:
                continue

            list_neg_active = False
            for clause, breaks_list in _split_atomic_clauses(line):
                low = clause.lower().strip()
                if not low:
                    continue

                if breaks_list or _NEW_ASSERTION_RE.search(low):
                    list_neg_active = False

                # Check if this clause starts or reinforces a negative list (e.g. "no evidence of pneumonia, pneumothorax, or effusion")
                if _LIST_NEG_TRIGGER_RE.search(low) and not _HEDGE_RE.search(low):
                    list_neg_active = True

                has_hedge = bool(_HEDGE_RE.search(low))
                # Treat "cannot exclude" as uncertain even though it contains "cannot"/"exclude"
                clause_wo_hedge = _HEDGE_RE.sub(" ", low)
                has_neg = bool(_CLAUSE_NEG_RE.search(clause_wo_hedge)) or list_neg_active

                if has_hedge and not _CLAUSE_NEG_RE.search(clause_wo_hedge):
                    clause_state = 1  # uncertain
                elif has_neg:
                    clause_state = 0  # absent
                elif has_hedge:
                    clause_state = 1
                else:
                    clause_state = 2  # present

                for canon, patterns in _CLASS_PATTERNS.items():
                    if any(p.search(low) for p in patterns):
                        if clause_state > priority[canon]:
                            priority[canon] = clause_state

    for canon, prio in priority.items():
        statuses[canon] = status_name[prio]
    return statuses


def report_has_normal_claim(report_text: str) -> bool:
    """Returns True when the report explicitly states no acute / normal chest findings."""
    return bool(report_text and NORMAL_CLAIM_PATTERNS.search(report_text))


def normalize_gt_labels(raw_labels: Any) -> list[str]:
    """Converts dataset labels or finding dicts into sorted unique canonical names."""
    out: set[str] = set()
    for item in raw_labels or []:
        lbl = item.get("label") if isinstance(item, dict) else item
        if not lbl:
            continue
        canon = canonicalize_label(str(lbl))
        if canon and canon in PATHOLOGY_ONTOLOGY:
            out.add(canon)
    return sorted(out)


def compute_clinical_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Computes per-finding Sensitivity, Specificity, and F1 plus overall:
      - `normal_bias_rate`: fraction of abnormal studies reported as normal
      - `false_alarm_rate`: fraction of normal studies reported as abnormal
      - `macro_sensitivity`: mean per-finding sensitivity across present GT classes
      - `rulebook_compliance`: fraction passing 5-Rule format validation
      - `combined_score`: rewards higher macro sensitivity & lower normal-bias,
        and penalizes `false_alarm_rate > 0.25`.
    """
    if not records:
        return {
            "total_studies": 0,
            "abnormal_studies": 0,
            "normal_studies": 0,
            "normal_bias_count": 0,
            "false_alarm_count": 0,
            "rulebook_compliant_count": 0,
            "normal_bias_rate": 0.0,
            "false_alarm_rate": 0.0,
            "macro_sensitivity": 0.0,
            "rulebook_compliance": 0.0,
            "combined_score": 0.0,
            "per_finding": {},
        }

    enriched: list[dict[str, Any]] = []
    for r in records:
        gt_findings = normalize_gt_labels(r.get("gt_findings", r.get("labels", [])))
        gt_is_normal = len(gt_findings) == 0

        report_text = str(r.get("report", ""))
        if "pred_status" in r and isinstance(r["pred_status"], dict):
            status_map = r["pred_status"]
        else:
            status_map = parse_report_findings(report_text)

        pred_present = sorted(k for k, v in status_map.items() if v == "present")
        pred_uncertain = sorted(k for k, v in status_map.items() if v == "uncertain")

        rule_ok = r.get("rulebook_compliant")
        if rule_ok is None:
            rule_ok, _ = validate_rulebook_compliance(report_text)

        # Reported normal only when zero findings are asserted present OR flagged uncertain
        reported_normal = (len(pred_present) == 0) and (len(pred_uncertain) == 0)
        false_alarm = gt_is_normal and (len(pred_present) > 0)
        normal_bias = (not gt_is_normal) and reported_normal

        enriched.append({
            "gt_findings": gt_findings,
            "gt_is_normal": gt_is_normal,
            "pred_present": pred_present,
            "pred_uncertain": pred_uncertain,
            "rulebook_compliant": bool(rule_ok),
            "normal_bias": normal_bias,
            "false_alarm": false_alarm,
        })

    classes = sorted({lbl for e in enriched for lbl in e["gt_findings"]})
    per_finding: dict[str, dict[str, Any]] = {}
    sens_values: list[float] = []

    for cls in classes:
        tp = fp = fn = tn = 0
        for e in enriched:
            gt_pos = cls in e["gt_findings"]
            pred_pos = cls in e["pred_present"]
            if gt_pos and pred_pos:
                tp += 1
            elif (not gt_pos) and pred_pos:
                fp += 1
            elif gt_pos and (not pred_pos):
                fn += 1
            else:
                tn += 1

        support = tp + fn
        sens = tp / support if support > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        per_finding[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
        }
        if support > 0:
            sens_values.append(sens)

    abn = [e for e in enriched if not e["gt_is_normal"]]
    norm = [e for e in enriched if e["gt_is_normal"]]

    nbias_cnt = sum(1 for e in abn if e["normal_bias"])
    fa_cnt = sum(1 for e in norm if e["false_alarm"])
    comp_cnt = sum(1 for e in enriched if e["rulebook_compliant"])

    normal_bias_rate = (nbias_cnt / len(abn)) if abn else 0.0
    false_alarm_rate = (fa_cnt / len(norm)) if norm else 0.0
    macro_sensitivity = (sum(sens_values) / len(sens_values)) if sens_values else 0.0
    rulebook_compliance = comp_cnt / len(enriched)

    fa_penalty = max(0.0, false_alarm_rate - 0.25) * 1.5
    combined_score = macro_sensitivity - 0.5 * normal_bias_rate - fa_penalty

    return {
        "total_studies": len(enriched),
        "abnormal_studies": len(abn),
        "normal_studies": len(norm),
        "normal_bias_count": nbias_cnt,
        "false_alarm_count": fa_cnt,
        "rulebook_compliant_count": comp_cnt,
        "normal_bias_rate": round(normal_bias_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "macro_sensitivity": round(macro_sensitivity, 4),
        "rulebook_compliance": round(rulebook_compliance, 4),
        "combined_score": round(combined_score, 4),
        "per_finding": per_finding,
    }
