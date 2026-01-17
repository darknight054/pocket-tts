import torch
import torch.nn as nn
from torch.nn import functional as F


class QuantLinear(nn.Module):
    """Weight-only int8 Linear using per-row scales with dequant-on-forward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(0.0))
        self.register_buffer("input_zero_point", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("output_scale", torch.tensor(0.0))
        self.register_buffer("output_zero_point", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("use_static_activation", torch.tensor(0, dtype=torch.uint8))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self._packed_weight = None
        self._calibrating = False
        self._input_observer = None
        self._output_observer = None

    def _ensure_backend(self) -> bool:
        supported = torch.backends.quantized.supported_engines
        if not supported:
            return False
        if torch.backends.quantized.engine not in supported:
            torch.backends.quantized.engine = supported[0]
        return True

    def _pack_weight(self) -> None:
        if self._packed_weight is not None:
            return
        weight_q = self.weight_q.detach()
        weight_scale = self.weight_scale.detach()
        if weight_q.device.type != "cpu":
            weight_q = weight_q.cpu()
        if weight_scale.device.type != "cpu":
            weight_scale = weight_scale.cpu()
        zero_points = torch.zeros_like(weight_scale, dtype=torch.int64)
        qweight = torch._make_per_channel_quantized_tensor(weight_q, weight_scale, zero_points, 0)
        bias = self.bias.detach().cpu() if self.bias is not None else None
        self._packed_weight = torch.ops.quantized.linear_prepack(qweight, bias)

    def start_calibration(self, observer: str = "histogram") -> None:
        from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver

        observer = observer.lower()
        if observer == "histogram":
            obs_cls = HistogramObserver
        elif observer == "minmax":
            obs_cls = MinMaxObserver
        else:
            raise ValueError(f"Unsupported observer: {observer}")
        self._input_observer = obs_cls(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        self._output_observer = obs_cls(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        self._calibrating = True

    def finish_calibration(self) -> None:
        if not self._calibrating or self._input_observer is None or self._output_observer is None:
            return
        in_scale, in_zero = self._input_observer.calculate_qparams()
        out_scale, out_zero = self._output_observer.calculate_qparams()
        self.input_scale.fill_(float(in_scale.item()))
        self.input_zero_point.fill_(int(in_zero.item()))
        self.output_scale.fill_(float(out_scale.item()))
        self.output_zero_point.fill_(int(out_zero.item()))
        self.use_static_activation.fill_(1)
        self._calibrating = False
        self._input_observer = None
        self._output_observer = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantLinear":
        quant = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        if linear.bias is not None:
            quant.bias.data.copy_(linear.bias.data)
        return quant.to(linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._calibrating and self._input_observer is not None:
            x_cpu = x.detach().float().cpu()
            self._input_observer(x_cpu)
            weight = self.weight_q.float() * self.weight_scale[:, None]
            weight = weight.to(dtype=x.dtype, device=x.device)
            bias = self.bias
            if bias is not None:
                bias = bias.to(dtype=x.dtype, device=x.device)
            out = F.linear(x, weight, bias)
            if self._output_observer is not None:
                self._output_observer(out.detach().float().cpu())
            return out

        if (
            self.use_static_activation.item() == 1
            and x.device.type == "cpu"
            and self.input_scale.item() != 0.0
            and self.output_scale.item() != 0.0
            and self._ensure_backend()
        ):
            try:
                self._pack_weight()
                x_q = torch.quantize_per_tensor(
                    x.float(),
                    float(self.input_scale.item()),
                    int(self.input_zero_point.item()),
                    torch.quint8,
                )
                y_q = torch.ops.quantized.linear(
                    x_q,
                    self._packed_weight,
                    float(self.output_scale.item()),
                    int(self.output_zero_point.item()),
                )
                return y_q.dequantize().to(dtype=x.dtype)
            except Exception:
                # Fall back to dynamic or float path if static kernels fail.
                pass
        if x.device.type == "cpu" and self._ensure_backend():
            try:
                self._pack_weight()
                return torch.ops.quantized.linear_dynamic(x.float(), self._packed_weight)
            except Exception:
                # Fall back to dequantized weight if quantized kernels fail.
                pass
        weight = self.weight_q.float() * self.weight_scale[:, None]
        weight = weight.to(dtype=x.dtype, device=x.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype, device=x.device)
        return F.linear(x, weight, bias)
