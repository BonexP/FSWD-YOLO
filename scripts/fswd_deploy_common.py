#!/usr/bin/env python3
"""Shared, dependency-free helpers for FSWD-YOLO deployment tools."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


REVIEW_OPERATIONS = frozenset(
    {"ReduceMean", "ReduceSum", "Pow", "Div", "MatMul", "Softmax", "Expand", "Transpose"}
)


class DependencyError(RuntimeError):
    """Raised when a deployment operation is missing an explicit dependency."""


def add_repo_root_to_path(script_file: Path) -> Path:
    """Expose source packages beside scripts/ during direct script execution."""
    repo_root = Path(script_file).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root


def normalize_version_attribute(module: Any, attribute: str = "__version__") -> str:
    """Replace version-like string subclasses with a hashable plain string."""
    normalized = str(getattr(module, attribute))
    setattr(module, attribute, normalized)
    return normalized


def require_modules(requirements: Mapping[str, str], operation: str) -> Dict[str, str]:
    """Verify Python modules without installing or importing them."""
    found: Dict[str, str] = {}
    missing = []
    for module_name, manual_hint in requirements.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if not available:
            missing.append((module_name, manual_hint))
            continue
        try:
            found[module_name] = importlib.metadata.version(module_name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            found[module_name] = "installed"

    if missing:
        details = "\n".join(f"  - {name}: {hint}" for name, hint in missing)
        raise DependencyError(
            f"Missing dependencies for {operation}:\n{details}\n"
            f"Python interpreter: {sys.executable}\n"
            "This tool only checks dependencies and does not install packages."
        )
    return found


def validate_weights(path: Path) -> Path:
    """Return a resolved PyTorch checkpoint path after strict validation."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        raise ValueError(f"Weights path must use the .pt suffix: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Weights file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Weights path is not a file: {resolved}")
    return resolved


def prepare_output(path: Path, overwrite: bool, expected_suffix: str = ".onnx") -> Path:
    """Validate an output path and protect existing artifacts."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != expected_suffix.lower():
        raise ValueError(f"Output path must use the {expected_suffix} suffix: {resolved}")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {resolved}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Output path is not a file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA256 without loading the full artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root: Path) -> Dict[str, Any]:
    """Collect best-effort, read-only Git metadata."""

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={Path(repo_root).resolve()}", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
            "dirty": bool(run_git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def package_versions(package_names: Iterable[str]) -> Dict[str, Optional[str]]:
    """Return installed distribution versions without importing packages."""
    versions: Dict[str, Optional[str]] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def summarize_operators(operator_types: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """Build stable ONNX operator counts and a target-review subset."""
    counts = Counter(operator_types)
    histogram = dict(sorted(counts.items()))
    review = {name: histogram[name] for name in sorted(REVIEW_OPERATIONS) if name in histogram}
    return {"histogram": histogram, "review_operations": review}


def write_json_atomic(path: Path, payload: Mapping[str, Any], overwrite: bool = True) -> None:
    """Write complete JSON through a same-directory temporary file."""
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
