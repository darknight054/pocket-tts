#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import statistics
import time
from pathlib import Path

import torch

from pocket_tts.data.audio import audio_read, stream_audio_chunks
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.utils import PREDEFINED_VOICES

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = ROOT_DIR / "audios"
DEFAULT_TEXT = "Hello there! This is a short benchmark prompt to measure TTS speed."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pocket-tts generation speed.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=DEFAULT_AUDIO_PROMPT)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--lsd-steps", type=int, default=DEFAULT_LSD_DECODE_STEPS)
    parser.add_argument("--noise-clamp", type=float, default=DEFAULT_NOISE_CLAMP)
    parser.add_argument("--eos-threshold", type=float, default=DEFAULT_EOS_THRESHOLD)
    parser.add_argument("--frames-after-eos", type=int, default=DEFAULT_FRAMES_AFTER_EOS)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument(
        "--run-id",
        default=None,
        help="Directory name under audio-dir for this run (default: timestamp).",
    )
    parser.add_argument(
        "--compare-run",
        default=None,
        help="Run-id directory to compare against (same relative audio paths).",
    )
    parser.add_argument(
        "--save-audio-iter",
        type=int,
        default=0,
        help="0-based measured iteration index to save (default: 0).",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Path to reference WAV for similarity metrics (same prompt).",
    )
    parser.add_argument(
        "--include-voice-encoding",
        action="store_true",
        help="Include voice prompt encoding in timing.",
    )
    parser.add_argument(
        "--truncate-voice",
        action="store_true",
        help="Truncate voice prompts to 30s before encoding.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--list-voices", action="store_true", help="Print predefined voices and exit."
    )
    return parser.parse_args()


def _slugify(text: str, max_len: int = 40) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if text else "text"


def _safe_tag(value: str) -> str:
    return _slugify(str(value), max_len=24)


def _resolve_audio_dir(path: str) -> Path:
    audio_dir = Path(path)
    if not audio_dir.is_absolute():
        audio_dir = ROOT_DIR / audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def _save_audio(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    if audio.dim() == 2 and audio.shape[0] == 1:
        audio = audio[0]
    audio = audio.detach().cpu()
    stream_audio_chunks(path, iter([audio]), sample_rate)


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        return audio
    if audio.dim() == 2:
        if audio.shape[0] == 1:
            return audio[0]
        return audio.mean(dim=0)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def _load_ref_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    ref, ref_sr = audio_read(path)
    if ref_sr != target_sr:
        ref = convert_audio(ref, ref_sr, target_sr, 1)
    return _to_mono(ref)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    device: torch.device,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    if f_max is None:
        f_max = sample_rate / 2
    m_min = 2595.0 * torch.log10(torch.tensor(1.0 + f_min / 700.0, device=device))
    m_max = 2595.0 * torch.log10(torch.tensor(1.0 + f_max / 700.0, device=device))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    bins = torch.floor((n_fft // 2 + 1) * f_pts / sample_rate).to(torch.int64)
    fb = torch.zeros((n_mels, n_fft // 2 + 1), device=device)
    for i in range(n_mels):
        start, center, end = bins[i], bins[i + 1], bins[i + 2]
        if center <= start:
            center = start + 1
        if end <= center:
            end = center + 1
        fb[i, start:center] = (torch.arange(start, center, device=device) - start) / (
            center - start
        )
        fb[i, center:end] = (end - torch.arange(center, end, device=device)) / (end - center)
    return fb


def _mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> torch.Tensor:
    audio = _to_mono(audio).detach().cpu().float()
    if audio.numel() == 0:
        return torch.empty((n_mels, 0))
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().pow(2.0)
    fb = _mel_filterbank(sample_rate, n_fft, n_mels, device=mag.device)
    mel = fb @ mag
    return torch.log(mel + 1e-6)


def _compute_similarity(
    pred: torch.Tensor, ref: torch.Tensor, sample_rate: int
) -> dict[str, float]:
    pred = _to_mono(pred).detach().cpu().float()
    ref = _to_mono(ref).detach().cpu().float()
    length = min(pred.shape[-1], ref.shape[-1])
    if length == 0:
        return {}
    pred = pred[..., :length]
    ref = ref[..., :length]
    diff = pred - ref
    mse = torch.mean(diff**2).item()
    mae = torch.mean(torch.abs(diff)).item()
    cos = torch.nn.functional.cosine_similarity(pred, ref, dim=0).item()
    signal = torch.mean(ref**2).item()
    noise = torch.mean(diff**2).item()
    eps = 1e-12
    snr = 10.0 * torch.log10(torch.tensor((signal + eps) / (noise + eps))).item()
    metrics = {"mse": mse, "mae": mae, "cosine": cos, "snr_db": snr}
    mel_pred = _mel_spectrogram(pred, sample_rate)
    mel_ref = _mel_spectrogram(ref, sample_rate)
    mel_len = min(mel_pred.shape[-1], mel_ref.shape[-1])
    if mel_len > 0:
        mel_pred = mel_pred[..., :mel_len]
        mel_ref = mel_ref[..., :mel_len]
        mel_diff = mel_pred - mel_ref
        mel_mse = torch.mean(mel_diff**2).item()
        mel_mae = torch.mean(torch.abs(mel_diff)).item()
        mel_cos = torch.nn.functional.cosine_similarity(
            mel_pred.flatten(), mel_ref.flatten(), dim=0
        ).item()
        metrics.update({"mel_mse": mel_mse, "mel_mae": mel_mae, "mel_cosine": mel_cos})
    return metrics


def build_model_state(tts_model: TTSModel, voice: str, truncate: bool) -> dict:
    if voice.lower() == "none":
        return init_states(tts_model.flow_lm, batch_size=1, sequence_length=1)
    return tts_model.get_state_for_audio_prompt(voice, truncate=truncate)


def run_once(
    tts_model: TTSModel, model_state: dict, text: str, frames_after_eos: int | None
) -> tuple[float, float, float, torch.Tensor]:
    start = time.perf_counter()
    audio = tts_model.generate_audio(
        model_state=model_state,
        text_to_generate=text,
        frames_after_eos=frames_after_eos,
        copy_state=True,
    )
    elapsed = time.perf_counter() - start
    samples = audio.shape[-1] if audio.numel() else 0
    audio_sec = samples / tts_model.sample_rate if samples else 0.0
    rtf = audio_sec / elapsed if elapsed else float("inf")
    return elapsed, audio_sec, rtf, audio


def main() -> None:
    args = parse_args()

    if args.list_voices:
        print("\n".join(sorted(PREDEFINED_VOICES)))
        return

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    torch.set_num_threads(args.torch_threads)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tts_model = TTSModel.load_model(
        variant=args.variant,
        temp=args.temp,
        lsd_decode_steps=args.lsd_steps,
        noise_clamp=args.noise_clamp,
        eos_threshold=args.eos_threshold,
    )

    if args.include_voice_encoding:
        model_state = None
    else:
        model_state = build_model_state(tts_model, args.voice, args.truncate_voice)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    audio_root = _resolve_audio_dir(args.audio_dir) if args.save_audio else None
    run_dir = audio_root / run_id if audio_root else None
    compare_dir = audio_root / args.compare_run if (audio_root and args.compare_run) else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    ref_audio = None
    if args.ref_audio:
        ref_audio = _load_ref_audio(args.ref_audio, tts_model.sample_rate)

    for _ in range(args.warmup):
        state = (
            build_model_state(tts_model, args.voice, args.truncate_voice)
            if args.include_voice_encoding
            else model_state
        )
        run_once(tts_model, state, args.text, args.frames_after_eos)

    times = []
    rtfs = []
    audio_secs = []
    mse_vals = []
    mae_vals = []
    cos_vals = []
    snr_vals = []
    mel_mse_vals = []
    mel_mae_vals = []
    mel_cos_vals = []
    saved_audio_path = None
    missing_ref = 0
    for _ in range(args.iters):
        iter_idx = len(times)
        state = (
            build_model_state(tts_model, args.voice, args.truncate_voice)
            if args.include_voice_encoding
            else model_state
        )
        elapsed, audio_sec, rtf, audio = run_once(
            tts_model, state, args.text, args.frames_after_eos
        )
        times.append(elapsed)
        rtfs.append(rtf)
        audio_secs.append(audio_sec)
        if ref_audio is not None:
            metrics = _compute_similarity(audio, ref_audio, tts_model.sample_rate)
            if metrics:
                mse_vals.append(metrics["mse"])
                mae_vals.append(metrics["mae"])
                cos_vals.append(metrics["cosine"])
                snr_vals.append(metrics["snr_db"])
                if "mel_mse" in metrics:
                    mel_mse_vals.append(metrics["mel_mse"])
                    mel_mae_vals.append(metrics["mel_mae"])
                    mel_cos_vals.append(metrics["mel_cosine"])
        if args.save_audio and iter_idx == args.save_audio_iter:
            voice_tag = _safe_tag(args.voice)
            text_tag = _slugify(args.text)
            text_hash = hashlib.sha1(args.text.encode("utf-8")).hexdigest()[:8]
            filename = f"{voice_tag}_{text_tag}_{text_hash}.wav"
            saved_audio_path = run_dir / filename if run_dir else None
            if saved_audio_path is not None:
                _save_audio(audio, saved_audio_path, tts_model.sample_rate)
                if compare_dir is not None:
                    ref_path = compare_dir / filename
                    if ref_path.exists():
                        ref_audio = _load_ref_audio(ref_path, tts_model.sample_rate)
                        metrics = _compute_similarity(audio, ref_audio, tts_model.sample_rate)
                        if metrics:
                            mse_vals.append(metrics["mse"])
                            mae_vals.append(metrics["mae"])
                            cos_vals.append(metrics["cosine"])
                            snr_vals.append(metrics["snr_db"])
                            if "mel_mse" in metrics:
                                mel_mse_vals.append(metrics["mel_mse"])
                                mel_mae_vals.append(metrics["mel_mae"])
                                mel_cos_vals.append(metrics["mel_cosine"])
                    else:
                        missing_ref += 1

    median_time = statistics.median(times)
    mean_time = statistics.fmean(times)
    p90_time = statistics.quantiles(times, n=10)[-1] if len(times) >= 2 else times[0]
    median_rtf = statistics.median(rtfs)
    mean_audio = statistics.fmean(audio_secs)

    print("benchmark_result")
    print(f"variant={args.variant}")
    print(f"voice={args.voice}")
    print(f"text_len={len(args.text)}")
    print(f"iters={args.iters}")
    print(f"warmup={args.warmup}")
    print(f"audio_sec_mean={mean_audio:.3f}")
    print(f"time_sec_median={median_time:.3f}")
    print(f"time_sec_mean={mean_time:.3f}")
    print(f"time_sec_p90={p90_time:.3f}")
    print(f"rtf_median={median_rtf:.3f}")
    if saved_audio_path is not None:
        print(f"audio_file={saved_audio_path}")
        print(f"run_id={run_id}")
        if args.compare_run:
            print(f"compare_run={args.compare_run}")
    if mse_vals:
        if args.ref_audio:
            print(f"similarity_ref_audio={args.ref_audio}")
        if args.compare_run:
            print(f"similarity_ref_run={args.compare_run}")
        print(f"similarity_mse_median={statistics.median(mse_vals):.6f}")
        print(f"similarity_mae_median={statistics.median(mae_vals):.6f}")
        print(f"similarity_cosine_median={statistics.median(cos_vals):.6f}")
        print(f"similarity_snr_db_median={statistics.median(snr_vals):.3f}")
        if mel_mse_vals:
            print(f"similarity_mel_mse_median={statistics.median(mel_mse_vals):.6f}")
            print(f"similarity_mel_mae_median={statistics.median(mel_mae_vals):.6f}")
            print(f"similarity_mel_cosine_median={statistics.median(mel_cos_vals):.6f}")
    if missing_ref:
        print(f"similarity_missing_ref_count={missing_ref}")


if __name__ == "__main__":
    main()
