"""Model adapters for keeping FSWD-YOLO decode and NMS on the host."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from ultralytics.nn.modules import Detect


def find_detect_head(model: nn.Module) -> Detect:
    """Return the single Detect head in an Ultralytics detection model."""
    heads = [module for module in model.modules() if isinstance(module, Detect)]
    if len(heads) != 1:
        raise ValueError(f"expected exactly one Detect head, found {len(heads)}")
    return heads[0]


class RawDetectHeadAdapter(nn.Module):
    """Expose only the convolutional Detect feature maps for DPU inspection."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.detect = find_detect_head(model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run the model while bypassing DFL, decode, sigmoid, and NMS."""
        previous = getattr(self.detect, "raw_output", False)
        self.detect.raw_output = True
        try:
            outputs = self.model(x)
        finally:
            self.detect.raw_output = previous

        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise TypeError("raw Detect output must be a non-empty tensor sequence")
        if not all(isinstance(output, torch.Tensor) for output in outputs):
            raise TypeError("raw Detect output contains a non-tensor value")
        return tuple(outputs)


def decode_raw_predictions(model: nn.Module, raw_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Decode raw feature maps with the model's native Detect implementation."""
    detect = find_detect_head(model)
    if len(raw_outputs) != detect.nl:
        raise ValueError(f"expected {detect.nl} raw feature maps, received {len(raw_outputs)}")
    return detect._inference(list(raw_outputs))


def _native_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("native model output does not contain a decoded prediction tensor")


def compare_decode_parity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare native inference with host decoding of the raw DPU outputs."""
    was_training = model.training
    try:
        model.eval()
        adapter = RawDetectHeadAdapter(model).eval()
        with torch.no_grad():
            native = _native_prediction(model(input_tensor)).detach().clone()
            raw_outputs = adapter(input_tensor)
            decoded = decode_raw_predictions(model, raw_outputs).detach().clone()
    finally:
        model.train(was_training)

    native_shape = list(native.shape)
    decoded_shape = list(decoded.shape)
    shape_matches = native_shape == decoded_shape
    if not shape_matches:
        return {
            "native_shape": native_shape,
            "decoded_shape": decoded_shape,
            "shape_matches": False,
            "allclose": False,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "rtol": rtol,
            "atol": atol,
        }

    absolute_error = (native - decoded).abs()
    return {
        "native_shape": native_shape,
        "decoded_shape": decoded_shape,
        "shape_matches": True,
        "allclose": bool(torch.allclose(native, decoded, rtol=rtol, atol=atol)),
        "max_absolute_error": float(absolute_error.max().item()),
        "mean_absolute_error": float(absolute_error.mean().item()),
        "rtol": rtol,
        "atol": atol,
    }
