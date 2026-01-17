#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import safetensors.torch
import torch

from pocket_tts.utils.config import load_config
from pocket_tts.utils.quantization import quantize_weight_per_row, should_quantize_key
from pocket_tts.utils.utils import download_if_necessary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize weights to int8 (weight-only).")
    parser.add_argument("--variant", default="b6369a24")
    parser.add_argument("--weights-in", default=None)
    parser.add_argument("--weights-out", default=None)
    parser.add_argument("--scope", default="flow_lm")
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.weights_in is not None:
        return download_if_necessary(args.weights_in)
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "pocket_tts" / "config" / f"{args.variant}.yaml")
    if config.weights_path is None:
        raise ValueError("No weights_path found in config; pass --weights-in explicitly.")
    return download_if_necessary(config.weights_path)


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.weights_out is not None:
        return Path(args.weights_out)
    out_dir = Path("weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"tts_{args.variant}_int8.safetensors"


def main() -> None:
    args = parse_args()
    in_path = resolve_input_path(args)
    out_path = resolve_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = safetensors.torch.load_file(in_path)
    output: dict[str, torch.Tensor] = {}
    quantized = 0

    for key, tensor in state.items():
        if should_quantize_key(key, tensor, args.scope):
            weight_q, scale = quantize_weight_per_row(tensor)
            base = key.removesuffix(".weight")
            output[f"{base}.weight_q"] = weight_q
            output[f"{base}.weight_scale"] = scale
            output[f"{base}.input_scale"] = torch.tensor(0.0)
            output[f"{base}.input_zero_point"] = torch.tensor(0, dtype=torch.int64)
            output[f"{base}.output_scale"] = torch.tensor(0.0)
            output[f"{base}.output_zero_point"] = torch.tensor(0, dtype=torch.int64)
            output[f"{base}.use_static_activation"] = torch.tensor(0, dtype=torch.uint8)
            quantized += 1
        else:
            output[key] = tensor

    metadata = {
        "quantization": "weight_only_int8",
        "scope": args.scope,
        "symmetric": "true",
        "per_channel": "true",
        "quantized_weights": str(quantized),
    }
    safetensors.torch.save_file(output, out_path, metadata=metadata)

    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"quantized_weights={quantized}")


if __name__ == "__main__":
    main()
