#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _run_in_repo(repo_path: Path, env: dict) -> dict:
    code = textwrap.dedent(
        """
import json
import os
import statistics
import time

import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.utils import size_of_dict

text = os.environ["POCKET_TTS_TEXT"]
voice = os.environ["POCKET_TTS_VOICE"]
variant = os.environ["POCKET_TTS_VARIANT"]
frames_after_eos = int(os.environ["POCKET_TTS_FRAMES_AFTER_EOS"])
iters = int(os.environ["POCKET_TTS_ITERS"])
seed = int(os.environ["POCKET_TTS_SEED"])

torch.manual_seed(seed)

tts = TTSModel.load_model(variant)
state = tts.get_state_for_audio_prompt(voice, truncate=True)

prepared = tts.flow_lm.conditioner.prepare(text)
token_count = prepared.tokens.shape[1]
word_count = len(text.split())

current_end = 0
cache_len_prompt = None
for module_name, module in tts.flow_lm.named_modules():
    if isinstance(module, StreamingMultiheadAttention):
        current_end = state[module_name]["current_end"].shape[0]
        cache_len_prompt = state[module_name]["cache"].shape[2]
        break

if hasattr(tts, "_estimate_max_gen_len"):
    max_gen_len = tts._estimate_max_gen_len(token_count, word_count)
else:
    max_gen_len = int((word_count * 1 + 2.0) * 12.5)

required_len = current_end + token_count + max_gen_len

state_mb_prompt = size_of_dict(state) / 1e6
state_mb_estimate = None
cache_len_estimate = None
if hasattr(tts, "_ensure_flow_lm_cache_capacity"):
    import copy

    state_est = copy.deepcopy(state)
    tts._ensure_flow_lm_cache_capacity(state_est, required_len)
    state_mb_estimate = size_of_dict(state_est) / 1e6
    for module_name, module in tts.flow_lm.named_modules():
        if isinstance(module, StreamingMultiheadAttention):
            cache_len_estimate = state_est[module_name]["cache"].shape[2]
            break

times = []
rtfs = []
audio_secs = []
for _ in range(iters):
    t0 = time.perf_counter()
    audio = tts.generate_audio(
        model_state=state,
        text_to_generate=text,
        frames_after_eos=frames_after_eos,
        copy_state=True,
    )
    elapsed = time.perf_counter() - t0
    audio_sec = audio.shape[-1] / tts.sample_rate if audio.numel() else 0.0
    rtf = audio_sec / elapsed if elapsed else float("inf")
    times.append(elapsed)
    rtfs.append(rtf)
    audio_secs.append(audio_sec)

result = {
    "token_count": token_count,
    "word_count": word_count,
    "current_end": current_end,
    "max_gen_len": max_gen_len,
    "required_len": required_len,
    "cache_len_prompt": cache_len_prompt,
    "cache_len_estimate": cache_len_estimate,
    "state_mb_prompt": state_mb_prompt,
    "state_mb_estimate": state_mb_estimate,
    "time_sec_median": statistics.median(times),
    "rtf_median": statistics.median(rtfs),
    "audio_sec_median": statistics.median(audio_secs),
}
print(json.dumps(result))
"""
    )
    cmd = [sys.executable, "-c", code]
    env = {**os.environ, **env}
    env["PYTHONPATH"] = str(repo_path)
    proc = subprocess.run(cmd, cwd=repo_path, env=env, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed in {repo_path}:\n{proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No output from run in {repo_path}")
    return json.loads(lines[-1])


def _add_worktree(repo_root: Path, ref: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix="pockettts_compare_"))
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), ref],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _remove_worktree(repo_root: Path, path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs reduced-memory-usage on a live example."
    )
    parser.add_argument("--baseline-ref", default="upstream/main")
    parser.add_argument("--candidate-ref", default="reduced-memory-usage")
    parser.add_argument("--baseline-path", default=None)
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--variant", default="b6369a24")
    parser.add_argument("--voice", default="alba")
    parser.add_argument(
        "--text",
        default=None,
        help="Optional single prompt. If omitted, a built-in set of long prompts is used.",
    )
    parser.add_argument("--frames-after-eos", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-worktrees", action="store_true")
    return parser.parse_args()


DEFAULT_PROMPTS = [
    (
        "When I started timing this model, I expected small gains, but the memory profile told a "
        "different story, and I want a consistent run that stresses streaming generation while "
        "still sounding natural, so this sentence is deliberately long and descriptive."
    ),
    (
        "We should measure performance on a passage that mixes short and long clauses, includes "
        "commas and conjunctions, and keeps the speech flowing, because that better represents "
        "real usage than a tiny prompt or a single phrase."
    ),
    (
        "A longer benchmark prompt is useful because it warms the transformer, exercises the "
        "autoregressive loop, and gives the decoder enough frames to reveal whether cache sizing "
        "is helping or hurting across multiple attention layers."
    ),
    (
        "To make this test meaningful, I am describing a brief narrative about calibrating a "
        "speech model, checking the latency, watching the audio stream, and comparing a baseline "
        "path against an optimized path, all in one breath."
    ),
    (
        "This is another extended sentence that keeps the model busy, includes varied vocabulary, "
        "and should provide enough tokens to observe consistent behavior from the KV cache, which "
        "is exactly what we need for a trustworthy comparison."
    ),
    (
        "If the benchmark prompt is too short, the results swing wildly, so I am adding this "
        "longer sentence to stabilize the measurement, capture steady state, and reduce the "
        "effect of one-off overheads."
    ),
    (
        "The purpose here is to generate audio that is long enough to reveal performance trends, "
        "yet not so long that the test is impractical, which is why I am writing this long "
        "sentence with clear pacing and multiple clauses."
    ),
    (
        "In practice, a prompt like this may be read aloud in a tutorial or a product demo, "
        "and it should be long enough to show the time-to-first-audio and the sustained speed, "
        "so we can compare baseline and reduced-memory runs."
    ),
    (
        "We also want a prompt that exercises punctuation, spacing, and a few numbers like twenty "
        "five or seventy two, and still remains a single cohesive line of speech to keep the "
        "measurement consistent across runs."
    ),
    (
        "Finally, this long sentence provides a stable target for regression testing, and it "
        "ensures that the measured differences reflect cache sizing and dtype behavior rather "
        "than random variance from tiny or irregular prompts."
    ),
]


def _ensure_min_words(prompts: list[str], min_words: int) -> list[str]:
    filtered: list[str] = []
    for prompt in prompts:
        word_count = len(prompt.split())
        if word_count < min_words:
            raise ValueError(f"Prompt too short ({word_count} words, min {min_words}): {prompt}")
        filtered.append(prompt)
    return filtered


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    baseline_path = Path(args.baseline_path).resolve() if args.baseline_path else None
    candidate_path = Path(args.candidate_path).resolve() if args.candidate_path else None

    temp_paths: list[Path] = []
    if baseline_path is None:
        baseline_path = _add_worktree(repo_root, args.baseline_ref)
        temp_paths.append(baseline_path)
    if candidate_path is None:
        candidate_path = _add_worktree(repo_root, args.candidate_ref)
        temp_paths.append(candidate_path)

    prompts = [args.text] if args.text else DEFAULT_PROMPTS
    prompts = _ensure_min_words(prompts, min_words=30)
    if len(prompts) < 10:
        raise ValueError(f"Need at least 10 prompts, found {len(prompts)}")

    base_env = {
        "POCKET_TTS_VOICE": args.voice,
        "POCKET_TTS_VARIANT": args.variant,
        "POCKET_TTS_FRAMES_AFTER_EOS": str(args.frames_after_eos),
        "POCKET_TTS_ITERS": str(args.iters),
        "POCKET_TTS_SEED": str(args.seed),
    }

    try:
        baseline_runs = []
        candidate_runs = []
        for idx, prompt in enumerate(prompts, start=1):
            env = {**base_env, "POCKET_TTS_TEXT": prompt}
            baseline = _run_in_repo(baseline_path, env)
            candidate = _run_in_repo(candidate_path, env)
            baseline_runs.append(baseline)
            candidate_runs.append(candidate)
            print(
                "prompt_result "
                f"index={idx} words={baseline['word_count']} "
                f"baseline_prompt_mb={baseline['state_mb_prompt']:.2f} "
                f"candidate_prompt_mb={candidate['state_mb_prompt']:.2f} "
                f"baseline_rtf={baseline['rtf_median']:.3f} "
                f"candidate_rtf={candidate['rtf_median']:.3f}"
            )
    finally:
        if not args.keep_worktrees:
            for path in temp_paths:
                _remove_worktree(repo_root, path)

    def _median(values):
        return sorted(values)[len(values) // 2]

    baseline_mb = [item["state_mb_prompt"] for item in baseline_runs]
    candidate_mb = [item["state_mb_prompt"] for item in candidate_runs]
    baseline_rtf = [item["rtf_median"] for item in baseline_runs]
    candidate_rtf = [item["rtf_median"] for item in candidate_runs]

    reduction = _median([b / c for b, c in zip(baseline_mb, candidate_mb)])
    rtf_gain = _median([c / b for b, c in zip(baseline_rtf, candidate_rtf)])
    baseline_audio = _median([item["audio_sec_median"] for item in baseline_runs])
    candidate_audio = _median([item["audio_sec_median"] for item in candidate_runs])

    print("compare_result")
    print(f"prompts={len(prompts)}")
    print(f"baseline_prompt_mb_median={_median(baseline_mb):.2f}")
    print(f"candidate_prompt_mb_median={_median(candidate_mb):.2f}")
    print(f"prompt_mb_reduction_x_median={reduction:.2f}")
    print(f"baseline_rtf_median={_median(baseline_rtf):.3f}")
    print(f"candidate_rtf_median={_median(candidate_rtf):.3f}")
    print(f"rtf_gain_x_median={rtf_gain:.2f}")
    print(f"baseline_audio_sec_median={baseline_audio:.3f}")
    print(f"candidate_audio_sec_median={candidate_audio:.3f}")


if __name__ == "__main__":
    main()
