#!/usr/bin/env python3
"""Parse and summarize Vitis AI Inspector hardware-constraint reports."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


REPORT_ROW_PATTERN = re.compile(
    r"^(?P<node>\S+::\S+)\s{2,}(?P<operator>\S+)\s{2,}(?P<reason>\S.*)$"
)
DIRECT_MARKERS = (
    "can't be converted to xir",
    "can't be assigned to dpu",
    "try to assign",
    "convert nndct graph to xir failed",
    "does not support",
    "only supports",
)
DIRECT_CPU_ASSIGNMENT_MARKERS = (
    "has been assigned to cpu",
)
DOWNSTREAM_MARKERS = (
    "all input of concat are on cpu",
    "input of reshape is not on dpu",
    "input is from non-dpu device",
)
MAX_DIRECT_EXAMPLES = 3

ConstraintRow = Tuple[str, str, str]
ReportSignature = Tuple[int, int]


class InspectorReportError(RuntimeError):
    """Raised when an Inspector report is missing or cannot be interpreted."""


def classify_constraint_reason(reason: str) -> str:
    """Classify an Inspector constraint as direct, downstream, or other."""
    normalized = reason.lower()
    if any(marker in normalized for marker in DIRECT_MARKERS):
        return "direct_unsupported"
    if any(marker in normalized for marker in DIRECT_CPU_ASSIGNMENT_MARKERS):
        return "direct_cpu_assignment"
    if any(marker in normalized for marker in DOWNSTREAM_MARKERS):
        return "downstream_cpu"
    return "other_cpu_constraint"


def parse_constraint_rows(text: str) -> List[ConstraintRow]:
    """Extract hardware-constraint rows from Inspector report text."""
    rows = []
    for line in text.splitlines():
        match = REPORT_ROW_PATTERN.match(line.rstrip())
        if match:
            rows.append(
                (match.group("node"), match.group("operator"), match.group("reason"))
            )
    return rows


def _sorted_counts(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_constraint_rows(rows: List[ConstraintRow]) -> Dict[str, Any]:
    """Deduplicate and summarize parsed hardware-constraint rows."""
    unique_rows = sorted(set(rows))
    categorized = defaultdict(list)
    for row in unique_rows:
        categorized[classify_constraint_reason(row[2])].append(row)

    category_counts = {
        category: len(category_rows)
        for category, category_rows in sorted(categorized.items())
    }
    operators_by_category = {
        category: _sorted_counts(row[1] for row in category_rows)
        for category, category_rows in sorted(categorized.items())
    }

    direct_by_operator = defaultdict(list)
    for category in ("direct_unsupported", "direct_cpu_assignment"):
        for node, operator, reason in categorized.get(category, []):
            direct_by_operator[operator].append(
                {"node": node, "reason": reason, "category": category}
            )
    direct_blockers = {
        operator: {
            "count": len(examples),
            "examples": examples[:MAX_DIRECT_EXAMPLES],
        }
        for operator, examples in sorted(direct_by_operator.items())
    }

    return {
        "parsed_row_count": len(rows),
        "unique_row_count": len(unique_rows),
        "repeated_row_count": len(rows) - len(unique_rows),
        "unique_node_count": len({row[0] for row in unique_rows}),
        "category_counts": category_counts,
        "operators_by_category": operators_by_category,
        "direct_blockers": direct_blockers,
        "requires_cpu_fallback": bool(unique_rows),
    }


def summarize_inspector_report_text(text: str) -> Dict[str, Any]:
    """Validate and summarize a complete Inspector text report."""
    rows = parse_constraint_rows(text)
    if not rows and "hardware constraints" not in text.lower():
        raise InspectorReportError(
            "Inspector report does not contain a hardware constraints table"
        )
    return summarize_constraint_rows(rows)


def summarize_inspector_report(report_path: Path) -> Dict[str, Any]:
    """Read an Inspector text report and include its resolved path in the summary."""
    path = Path(report_path).resolve()
    summary = summarize_inspector_report_text(
        path.read_text(encoding="utf-8", errors="replace")
    )
    return {"report_path": str(path), **summary}


def snapshot_report_signatures(output_dir: Path) -> Dict[Path, ReportSignature]:
    """Record modification signatures for existing Inspector text reports."""
    directory = Path(output_dir)
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in directory.glob("inspect_*.txt")
        if path.is_file()
    }


def find_generated_report(
    output_dir: Path, before: Mapping[Path, ReportSignature]
) -> Path:
    """Return the single report created or updated by an Inspector invocation."""
    after = snapshot_report_signatures(output_dir)
    changed = sorted(
        path for path, signature in after.items() if before.get(path) != signature
    )
    if not changed:
        raise InspectorReportError(
            "Inspector did not generate or update an inspect_*.txt report"
        )
    if len(changed) > 1:
        raise InspectorReportError(
            "Inspector generated or updated multiple inspect_*.txt reports"
        )
    return changed[0]
