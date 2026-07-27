"""Audited weight migration helpers for the FSWD-YOLO DPU candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

from ultralytics.nn.modules import C2PSFCA, C3k2, C3k2GhostSimAMinner, Conv
from ultralytics.nn.modules.fswd_dpu import (
    C2PSFCADPU,
    C3k2DPU,
    C3k2GhostSimAMinnerDPU,
    SplitFreeC2f,
)


class MigrationError(RuntimeError):
    """Raised when a checkpoint cannot be mapped without silent loss."""


@dataclass
class MigrationSummary:
    """Tensor-level accounting for a migration operation."""

    copied: List[str] = field(default_factory=list)
    sliced: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)

    def extend(self, other: "MigrationSummary", prefix: str = "") -> None:
        """Merge another summary while applying a stable path prefix."""
        for field_name in ("copied", "sliced", "new", "unmapped", "unexpected"):
            target = getattr(self, field_name)
            target.extend(f"{prefix}{item}" for item in getattr(other, field_name))

    def to_dict(self) -> Dict[str, List[str]]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _shape(value: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def copy_conv_output_slice(source: Conv, destination: Conv, start: int, end: int) -> None:
    """Copy an output-channel slice from one Conv/BN block into another."""
    width = end - start
    if start < 0 or end <= start or end > source.conv.out_channels:
        raise MigrationError(f"invalid convolution output slice [{start}:{end}]")
    if destination.conv.out_channels != width:
        raise MigrationError(
            f"slice width {width} does not match destination outputs {destination.conv.out_channels}"
        )
    if source.conv.in_channels != destination.conv.in_channels:
        raise MigrationError("source and destination convolution inputs do not match")
    if source.conv.kernel_size != destination.conv.kernel_size:
        raise MigrationError("source and destination convolution kernels do not match")

    with torch.no_grad():
        destination.conv.weight.copy_(source.conv.weight[start:end])
        if destination.conv.bias is not None:
            if source.conv.bias is None:
                raise MigrationError("destination convolution bias has no source value")
            destination.conv.bias.copy_(source.conv.bias[start:end])

        for name in ("weight", "bias", "running_mean", "running_var"):
            source_value = getattr(source.bn, name)
            destination_value = getattr(destination.bn, name)
            destination_value.copy_(source_value[start:end])
        destination.bn.num_batches_tracked.copy_(source.bn.num_batches_tracked)


def copy_matching_state(source: nn.Module, destination: nn.Module) -> MigrationSummary:
    """Copy same-name, same-shape tensors and account for every other tensor."""
    summary = MigrationSummary()
    source_state = source.state_dict()
    destination_state = destination.state_dict()

    with torch.no_grad():
        for name, destination_value in destination_state.items():
            source_value = source_state.get(name)
            if source_value is None:
                summary.new.append(name)
                continue
            if _shape(source_value) != _shape(destination_value):
                summary.unexpected.append(
                    f"{name}: source {_shape(source_value)} != destination {_shape(destination_value)}"
                )
                continue
            destination_value.copy_(source_value)
            summary.copied.append(name)

    summary.unmapped.extend(name for name in source_state if name not in destination_state)
    return summary


def _copy_exact_module(
    source: nn.Module,
    destination: nn.Module,
    summary: MigrationSummary,
    source_prefix: str,
    destination_prefix: str,
) -> None:
    source_state = source.state_dict()
    destination_state = destination.state_dict()
    if source_state.keys() != destination_state.keys():
        summary.unexpected.append(
            f"{source_prefix} and {destination_prefix} expose different state keys"
        )
        return

    with torch.no_grad():
        for name, destination_value in destination_state.items():
            source_value = source_state[name]
            if _shape(source_value) != _shape(destination_value):
                summary.unexpected.append(
                    f"{source_prefix}{name}: source {_shape(source_value)} != destination {_shape(destination_value)}"
                )
                continue
            destination_value.copy_(source_value)
            summary.copied.append(f"{source_prefix}{name} -> {destination_prefix}{name}")


def migrate_split_free_module(source: nn.Module, destination: SplitFreeC2f) -> MigrationSummary:
    """Migrate a C2f-style fused projection into two independent projections."""
    if not isinstance(source, (C3k2, C3k2GhostSimAMinner)):
        raise MigrationError(f"unsupported split-free source module: {type(source).__name__}")
    if not isinstance(destination, (C3k2DPU, C3k2GhostSimAMinnerDPU)):
        raise MigrationError(f"unsupported split-free destination module: {type(destination).__name__}")
    if source.c != destination.c:
        raise MigrationError(f"hidden channel mismatch: {source.c} != {destination.c}")

    summary = MigrationSummary()
    copy_conv_output_slice(source.cv1, destination.cv_keep, 0, destination.c)
    copy_conv_output_slice(source.cv1, destination.cv_process, destination.c, 2 * destination.c)
    summary.sliced.extend(
        (
            f"cv1[0:{destination.c}] -> cv_keep",
            f"cv1[{destination.c}:{2 * destination.c}] -> cv_process",
        )
    )
    _copy_exact_module(source.cv2, destination.cv_fuse, summary, "cv2.", "cv_fuse.")
    _copy_exact_module(source.m, destination.m, summary, "m.", "m.")
    if summary.unexpected:
        raise MigrationError("; ".join(summary.unexpected))
    return summary


def migrate_c2psfca(source: C2PSFCA, destination: C2PSFCADPU) -> MigrationSummary:
    """Initialize the three DPU branches and account for redesigned attention state."""
    if source.c != destination.c:
        raise MigrationError(f"hidden channel mismatch: {source.c} != {destination.c}")

    summary = MigrationSummary()
    projections = (
        (destination.keep_projection, 0, "keep_projection"),
        (destination.spatial_projection, destination.c, "spatial_projection"),
        (destination.channel_projection, 2 * destination.c, "channel_projection"),
    )
    for projection, start, name in projections:
        copy_conv_output_slice(source.cv1, projection, start, start + destination.c)
        summary.sliced.append(f"cv1[{start}:{start + destination.c}] -> {name}")

    _copy_exact_module(source.cv2, destination.fuse, summary, "cv2.", "fuse.")
    summary.unmapped.extend(
        name for name in source.state_dict() if name.startswith(("m_psa.", "m_fca."))
    )
    summary.new.extend(
        name
        for name in destination.state_dict()
        if name.startswith(("spatial_blocks.", "channel_blocks."))
    )
    if summary.unexpected:
        raise MigrationError("; ".join(summary.unexpected))
    return summary


def _paired_children(source: nn.Sequential, destination: nn.Sequential) -> Iterable[Tuple[int, nn.Module, nn.Module]]:
    if len(source) != len(destination):
        raise MigrationError(f"sequential length mismatch: {len(source)} != {len(destination)}")
    return ((index, source[index], destination[index]) for index in range(len(source)))


def _migrate_pair(source: nn.Module, destination: nn.Module) -> MigrationSummary:
    if isinstance(source, C3k2) and isinstance(destination, C3k2DPU):
        return migrate_split_free_module(source, destination)
    if isinstance(source, C3k2GhostSimAMinner) and isinstance(destination, C3k2GhostSimAMinnerDPU):
        return migrate_split_free_module(source, destination)
    if isinstance(source, C2PSFCA) and isinstance(destination, C2PSFCADPU):
        return migrate_c2psfca(source, destination)
    if isinstance(source, nn.Sequential) and isinstance(destination, nn.Sequential) and type(source) is type(destination):
        summary = MigrationSummary()
        for index, source_child, destination_child in _paired_children(source, destination):
            summary.extend(_migrate_pair(source_child, destination_child), prefix=f"{index}.")
        return summary
    if type(source) is not type(destination):
        raise MigrationError(
            f"unsupported module pair: {type(source).__name__} -> {type(destination).__name__}"
        )

    summary = copy_matching_state(source, destination)
    if summary.new or summary.unmapped or summary.unexpected:
        details = summary.unexpected + [f"new tensor {name}" for name in summary.new] + [
            f"unmapped tensor {name}" for name in summary.unmapped
        ]
        raise MigrationError("; ".join(details))
    return summary


def _model_layers(model: nn.Module) -> nn.Sequential:
    layers = getattr(model, "model", model)
    if not isinstance(layers, nn.Sequential):
        raise MigrationError(f"expected a sequential Ultralytics model, got {type(layers).__name__}")
    return layers


def migrate_fswd_model(source: nn.Module, destination: nn.Module) -> Dict[str, Any]:
    """Migrate a complete FSWD model and return auditable tensor accounting."""
    source_layers = _model_layers(source)
    destination_layers = _model_layers(destination)
    if len(source_layers) != len(destination_layers):
        raise MigrationError(
            f"model layer count mismatch: {len(source_layers)} != {len(destination_layers)}"
        )

    total = MigrationSummary()
    layer_reports = []
    for index, source_layer, destination_layer in _paired_children(source_layers, destination_layers):
        layer_summary = _migrate_pair(source_layer, destination_layer)
        total.extend(layer_summary, prefix=f"model.{index}.")
        layer_reports.append(
            {
                "index": index,
                "source_type": type(source_layer).__name__,
                "destination_type": type(destination_layer).__name__,
                **layer_summary.to_dict(),
            }
        )

    if total.unexpected:
        raise MigrationError("; ".join(total.unexpected))
    return {
        "status": "ok",
        "summary": total.to_dict(),
        "counts": {name: len(getattr(total, name)) for name in total.to_dict()},
        "layers": layer_reports,
    }
