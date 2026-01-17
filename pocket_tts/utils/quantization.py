from __future__ import annotations

import torch
import torch.nn as nn

from pocket_tts.modules.quant_linear import QuantLinear

FLOW_LM_PREFIXES = ("flow_lm.transformer.", "flow_lm.flow_net.")


def _normalize_scope(scope: str) -> str:
    scope = scope.strip().lower()
    if scope in {"flow_lm", "flowlm"}:
        return "flow_lm"
    raise ValueError(f"Unsupported quantization scope: {scope}")


def should_quantize_module(name: str, module: nn.Module, scope: str) -> bool:
    scope = _normalize_scope(scope)
    if scope == "flow_lm":
        return isinstance(module, nn.Linear) and name.startswith(FLOW_LM_PREFIXES)
    return False


def should_quantize_key(key: str, tensor: torch.Tensor, scope: str) -> bool:
    scope = _normalize_scope(scope)
    if scope == "flow_lm":
        return key.startswith(FLOW_LM_PREFIXES) and key.endswith(".weight") and tensor.ndim == 2
    return False


def quantize_weight_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float()
    max_abs = weight.abs().amax(dim=1)
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    weight_q = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return weight_q, scale


def iter_quant_linear(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            yield module


def start_calibration(model: nn.Module, observer: str = "histogram") -> None:
    for module in iter_quant_linear(model):
        module.start_calibration(observer=observer)


def finish_calibration(model: nn.Module) -> None:
    for module in iter_quant_linear(model):
        module.finish_calibration()
