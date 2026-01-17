#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.state_utils import trim_flow_lm_kv_cache
from pocket_tts.utils.utils import size_of_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report FlowLM/Mimi KV cache sizes for current vs legacy sizing."
    )
    parser.add_argument("--variant", default="b6369a24")
    parser.add_argument("--voice", default="alba")
    parser.add_argument(
        "--text",
        default="Hello there. This is a quick speed test.",
        help="Text prompt used to estimate FlowLM cache length.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _estimate_required_len(tts: TTSModel, text: str, model_state: dict) -> dict[str, int]:
    prepared = tts.flow_lm.conditioner.prepare(text)
    word_count = len(text.split())
    token_count = prepared.tokens.shape[1]
    max_gen_len = tts._estimate_max_gen_len(token_count, word_count)
    current_end = tts._flow_lm_current_end(model_state)
    return {
        "required_len": current_end + token_count + max_gen_len,
        "current_end": current_end,
        "token_count": token_count,
        "word_count": word_count,
        "max_gen_len": max_gen_len,
    }


def _flow_cache_stats(tts: TTSModel, model_state: dict) -> dict[str, int]:
    cache_lengths = []
    current_ends = []
    for module_name, module in tts.flow_lm.named_modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        state = model_state.get(module_name)
        if state is None:
            continue
        cache = state.get("cache")
        current_end = state.get("current_end")
        if cache is None or current_end is None:
            continue
        cache_lengths.append(cache.shape[2])
        current_ends.append(current_end.shape[0])
    if not cache_lengths:
        return {}
    return {
        "cache_len_min": min(cache_lengths),
        "cache_len_max": max(cache_lengths),
        "current_end_min": min(current_ends),
        "current_end_max": max(current_ends),
    }


def _print_flow_cache_layers(tts: TTSModel, model_state: dict, label: str) -> None:
    for module_name, module in tts.flow_lm.named_modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        state = model_state.get(module_name)
        if state is None:
            continue
        cache = state.get("cache")
        current_end = state.get("current_end")
        if cache is None or current_end is None:
            continue
        print(
            f"{label}_layer={module_name} "
            f"cache_len={cache.shape[2]} current_end={current_end.shape[0]}"
        )


def main() -> None:
    args = parse_args()
    tts = TTSModel.load_model(args.variant)

    if args.voice.lower() == "none":
        flow_state = init_states(tts.flow_lm, batch_size=1, sequence_length=1)
    else:
        flow_state = tts.get_state_for_audio_prompt(args.voice, truncate=True)

    prompt_state_mb = size_of_dict(flow_state) / 1e6
    prompt_stats = _flow_cache_stats(tts, flow_state)

    estimate = _estimate_required_len(tts, args.text, flow_state)
    required_len = estimate["required_len"]

    flow_state_gen = copy.deepcopy(flow_state)
    tts._ensure_flow_lm_cache_capacity(flow_state_gen, required_len)
    gen_state_mb = size_of_dict(flow_state_gen) / 1e6
    gen_stats = _flow_cache_stats(tts, flow_state_gen)

    flow_state_gen_trim = copy.deepcopy(flow_state_gen)
    trim_flow_lm_kv_cache(flow_state_gen_trim, tts.flow_lm)
    gen_state_trim_mb = size_of_dict(flow_state_gen_trim) / 1e6
    gen_stats_trim = _flow_cache_stats(tts, flow_state_gen_trim)

    flow_state_old = init_states(tts.flow_lm, batch_size=1, sequence_length=1000)

    mimi_context = max(1, int(tts.config.mimi.transformer.context))
    mimi_state_new = init_states(tts.mimi, batch_size=1, sequence_length=mimi_context)
    mimi_state_old = init_states(tts.mimi, batch_size=1, sequence_length=1000)

    print(f"variant={args.variant}")
    print(f"voice={args.voice}")
    print(f"text_len={len(args.text)}")
    print(f"flow_tokens={estimate['token_count']}")
    print(f"flow_words={estimate['word_count']}")
    print(f"flow_max_gen_len={estimate['max_gen_len']}")
    print(f"flow_required_len={required_len}")
    print(f"flow_state_mb_prompt={prompt_state_mb:.2f}")
    if prompt_stats:
        print(
            "flow_state_prompt_cache_len="
            f"{prompt_stats['cache_len_min']}..{prompt_stats['cache_len_max']}"
        )
        print(
            "flow_state_prompt_current_end="
            f"{prompt_stats['current_end_min']}..{prompt_stats['current_end_max']}"
        )
    print(f"flow_state_mb_gen_estimate={gen_state_mb:.2f}")
    if gen_stats:
        print(
            f"flow_state_gen_cache_len={gen_stats['cache_len_min']}..{gen_stats['cache_len_max']}"
        )
    print(f"flow_state_mb_gen_trimmed={gen_state_trim_mb:.2f}")
    if gen_stats_trim:
        print(
            "flow_state_gen_trim_cache_len="
            f"{gen_stats_trim['cache_len_min']}..{gen_stats_trim['cache_len_max']}"
        )
    print(f"flow_state_mb_old={size_of_dict(flow_state_old) / 1e6:.2f}")
    print(f"mimi_context={mimi_context}")
    print(f"mimi_state_mb_new={size_of_dict(mimi_state_new) / 1e6:.2f}")
    print(f"mimi_state_mb_old={size_of_dict(mimi_state_old) / 1e6:.2f}")

    if args.verbose:
        _print_flow_cache_layers(tts, flow_state, "flow_prompt")
        _print_flow_cache_layers(tts, flow_state_gen, "flow_gen")
        _print_flow_cache_layers(tts, flow_state_gen_trim, "flow_gen_trim")


if __name__ == "__main__":
    main()
# uv run python scripts/kv_cache_report.py --voice alba --text "Hello there. This is a quick speed test."
