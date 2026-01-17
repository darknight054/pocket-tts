#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import safetensors.torch

from pocket_tts.default_parameters import DEFAULT_AUDIO_PROMPT
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.quantization import finish_calibration, start_calibration

DEFAULT_TEXTS = [
    "Hello there. This is a quick speed test.",
    "Good morning. This is pocket tts.",
    "The quick brown fox jumps over the lazy dog. Testing latency and clarity on a short paragraph.",
    "Today we measure generation speed on a few sentences to compare float32 and int8 runs.",
    "In this benchmark we generate a longer passage to stress the transformer and the audio decoder.",
    "This is a longer script with multiple sentences intended to keep the model busy for a while.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate static PTQ activation ranges.")
    parser.add_argument("--variant", default="b6369a24_int8")
    parser.add_argument("--voice", default=DEFAULT_AUDIO_PROMPT)
    parser.add_argument("--observer", default="histogram", choices=["histogram", "minmax"])
    parser.add_argument("--frames-after-eos", type=int, default=1)
    parser.add_argument("--weights-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tts = TTSModel.load_model(args.variant)
    start_calibration(tts, observer=args.observer)

    model_state = tts.get_state_for_audio_prompt(args.voice, truncate=True)
    for text in DEFAULT_TEXTS:
        tts.generate_audio(
            model_state=model_state,
            text_to_generate=text,
            frames_after_eos=args.frames_after_eos,
            copy_state=True,
        )

    finish_calibration(tts)

    out_path = (
        Path(args.weights_out)
        if args.weights_out is not None
        else Path("weights") / f"tts_{args.variant}_static.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tts.state_dict(), out_path)
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
