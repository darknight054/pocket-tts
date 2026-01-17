# Pocket-TTS Quantization + Memory Plan (TorchAO, BF16, KV cache)

Date: 2026-01-17

This document captures (1) what’s implemented today in this repo, (2) why PTQ made RTF worse on a MacBook Air M2, and (3) a step-by-step, implementation-ready plan to migrate to **TorchAO (`torchao`)** and to land low-hanging memory wins (BF16 + KV cache trimming) before doing bigger changes.

---

## 0) Current State (“what’s done till now”)

### 0.1 Manual / custom weight-only int8 (implemented)

- **Offline weight quantization script**: `scripts/quantize_weights.py`
  - Produces `weights/tts_<variant>_int8.safetensors`
  - Replaces selected `nn.Linear.weight` tensors with:
    - `<base>.weight_q` (int8)
    - `<base>.weight_scale` (float32 per-row scales)
    - plus activation qparam placeholders (`input_scale`, `output_scale`, etc.)
- **Runtime module swap**: `pocket_tts/models/tts_model.py` → `_apply_weight_only_int8()`
  - Replaces eligible `nn.Linear` layers with `pocket_tts/modules/quant_linear.py::QuantLinear`
  - Selection is controlled by:
    - `pocket_tts/utils/quantization.py::should_quantize_module()` (name-prefix based)
    - `FLOW_LM_PREFIXES = ("flow_lm.transformer.", "flow_lm.flow_net.")`
- **QuantLinear runtime behavior**: `pocket_tts/modules/quant_linear.py`
  - Default path: tries `torch.ops.quantized.linear_dynamic(x.float(), packed_weight)` (dynamic activation quantization)
  - Static path (after calibration): uses `torch.ops.quantized.linear(...)` after `torch.quantize_per_tensor(...)`
  - Fallback path: dequantizes weight every forward (`weight_q.float() * scale`) and runs `F.linear` (slow).

### 0.2 Static PTQ calibration (implemented, but currently too small)

- Script: `scripts/ptq_static_calibrate.py`
  - Calls `start_calibration(tts)` / `finish_calibration(tts)` from `pocket_tts/utils/quantization.py`
  - Calibration currently runs only **6 hardcoded texts** (`DEFAULT_TEXTS`)
  - Saves a “static-ready” checkpoint `weights/tts_<variant>_static.safetensors`

### 0.3 KV cache sizing + memory tooling (already improved)

These “low-hanging fruits” are already implemented (but should be validated on your workloads):

- **FlowLM KV cache grows to required length** before generation:
  - `pocket_tts/models/tts_model.py::_ensure_flow_lm_cache_capacity()`
  - `required_len = current_end + text_tokens + max_gen_len_guess`
- **Mimi KV cache uses transformer context**, not a fixed 1000:
  - `pocket_tts/models/tts_model.py::_decode_audio_worker()` → `mimi_context = config.mimi.transformer.context`
- **Report script**: `scripts/kv_cache_report.py`
  - Compares “new sizing” vs “legacy 1000” for FlowLM and Mimi state memory

### 0.4 Dtype behavior today (relevant to BF16 work)

- Configs set `flow_lm.dtype: float32` and `mimi.dtype: float32` (e.g., `pocket_tts/config/b6369a24.yaml`).
- The upstream weights may be stored in BF16, but if the model params are float32 then
  `load_state_dict()` will cast to float32.
- Some operations explicitly cast to float32 (e.g., `FlowLMModel.forward()` does `transformer_out.to(torch.float32)`).

---

## 1) Problem Statement + Hypotheses

Observed on your runs:

- After doing PTQ (static or dynamic), **RTF decreased** (i.e., got slower), which is unexpected.

Most likely causes (we will validate with instrumentation):

1) **Quantized kernel path not actually taken** (engine not supported / op mismatch) → frequent fallback to “dequantize-every-forward” path.
2) **Dynamic activation quantization overhead** dominates for our shapes (batch=1, small matmuls, streaming).
3) On **Apple Silicon**, the quantized backend/kernels for `quantized.linear(_dynamic)` may be less optimized than float32 kernels.
4) Extra overhead from quantization plumbing (packing, quant/dequant, Python-level try/except paths).

Important constraint:

- A MacBook Air M2 is **ARM64**. Many quantization speedups are strongest on x86 (FBGEMM/x86 backend).

---

## 2) What TorchAO (torchao) Adds (from docs)

TorchAO provides a higher-level quantization stack than our custom `QuantLinear`.

Key points (from `torchao.quantization` docs/README):

- One-line in-place API: `torchao.quantization.quantize_(model, <Config>)`
- Useful configs for our goals:
  - `Int8WeightOnlyConfig()` (W8, activations float)
  - `Int8DynamicActivationInt8WeightConfig()` (W8A8)
- TorchAO is **heavily optimized around `torch.compile` + Inductor** for speed.
- There are **experimental ARM CPU kernels** in TorchAO for dynamic activation + groupwise low-bit weights
  (note: these are int4/int3 weights; we can ignore for now, but they are relevant if M2 speedups are elusive).

Relevant references:

- `https://github.com/pytorch/ao/raw/refs/heads/main/torchao/quantization/README.md`
- `https://docs.pytorch.org/ao/stable/api_ref_quantization.html`
- `https://docs.pytorch.org/ao/stable/serialization.html`

---

## 3) Guiding Principles

- **Don’t regress audio quality silently.** Always save audio from benchmark runs and do quick listening checks.
- **Prefer reversible, opt-in toggles.** New dtype/quantization modes should be behind config flags / variant YAMLs.
- **Benchmark on the actual target (M2) first.** Quantization speedups on ARM CPUs are not guaranteed.
- **Keep the streaming/stateful architecture intact.** Avoid “batching” assumptions.

---

## 4) Workstreams (prioritized)

### Workstream A — Reproduce + Explain the RTF Regression (must-do first)

Goal: determine whether we are using the intended kernel path, and where time is spent.

1) Add a “quantization debug” mode that prints once per run:
   - `torch.backends.quantized.supported_engines`
   - `torch.backends.quantized.engine`
   - whether each `QuantLinear` used:
     - static kernel
     - dynamic kernel
     - float fallback

2) Add counters in `QuantLinear.forward()` (guarded by env var, e.g. `POCKET_TTS_QUANT_DEBUG=1`):
   - `static_calls`, `dynamic_calls`, `fallback_calls`, `pack_calls`, `static_failures`, `dynamic_failures`
   - print aggregated totals at the end of `scripts/benchmark.py` (or via `atexit`).

3) Baseline benchmarks to run on the M2:
   - Float: `--variant b6369a24`
   - Int8 weight-only: `--variant b6369a24_int8`
   - Int8 static (calibrated): `--variant b6369a24_int8_static`

Acceptance criteria:
   - We can answer: “Which kernel path executed?” and “What % of calls fell back?”

---

### Workstream B — Memory Wins: BF16 weights + KV cache trimming

#### B1) KV cache trimming in the **cached voice state**

Status: KV cache sizing is already improved globally (see `scripts/kv_cache_report.py`).

Remaining potential issue:

- The voice prompt state is cached via `TTSModel._cached_get_state_for_audio_prompt()`.
- If any state dict contains **preallocated** tensors larger than the “used” prefix, we should trim them before caching.

Implementation plan:

1) Add a utility: `pocket_tts/utils/state_trim.py` (name TBD)
   - `trim_state_inplace(model_state, model)`:
     - for each `StreamingMultiheadAttention` state:
       - slice `cache` to `:current_end` (or `:current_end + safety_margin`)
     - for Mimi caches (if cached anywhere later), slice to context
   - Ensure the trimming is **lossless** for the voice prompt state.

2) Call trimming only on the cached voice state path:
   - in `_cached_get_state_for_audio_prompt` just before returning

Validation:
   - Compare `size_of_dict(model_state)` before/after
   - Ensure audio output unchanged vs baseline for a fixed prompt

#### B2) Add a BF16 variant (weights + selected buffers)

Goal: reduce memory footprint and potentially speed up on M2 if BF16 kernels are good.

Plan:

1) Introduce a config-level dtype switch (minimal surface area):
   - Add to YAML:
     - `flow_lm.dtype: bfloat16`
     - `mimi.dtype: bfloat16`
   - Add new variants:
     - `pocket_tts/config/b6369a24_bf16.yaml`
     - optionally `pocket_tts/config/b6369a24_int8_bf16.yaml`

2) Make `TTSModel._from_pydantic_config_with_weights()` actually respect these dtypes:
   - Ensure model modules are converted to config dtype either:
     - before loading weights (to avoid upcasting), or
     - after loading (if we keep float weights intentionally)

3) Decide compute dtype strategy:
   - Option 1: “pure BF16 compute” (max memory win, risky for CPU kernels)
   - Option 2: “BF16 weights, float32 activations” (less kernel support risk, might insert casts)
   - Start with Option 1 and benchmark; fall back if slower/worse.

Validation:
   - Benchmark RTF and memory for float32 vs bf16 on M2.
   - Ensure no runtime errors in attention (SDPA) and LayerNorm.

---

### Workstream C — Fix Static PTQ Calibration (dataset + 100–150 examples)

Goal: calibration data should be representative and large enough to stabilize activation ranges.

Plan to upgrade `scripts/ptq_static_calibrate.py`:

1) Add CLI args:
   - `--num-examples` (default: 128)
   - `--seed` (default: 0)
   - `--dataset` (default: `wikitext`)
   - `--dataset-config` / `--split`
   - `--text-field` (default depends on dataset)
   - `--min-words` / `--max-words` (to control runtime)
   - `--text-file` (optional local newline-separated prompts; no extra deps)

2) Dataset choice (recommended order):
   - Primary: **Wikitext** (text-only, small download) → easy to adopt
   - Optional (more “spoken”): **LibriSpeech transcripts** (large dataset; use streaming + text-only)

3) Dependency strategy:
   - Add optional dependency group (e.g., `calibration`) with `datasets>=...`
   - Script should work without `datasets` if `--text-file` is provided

4) Text cleaning:
   - Strip whitespace, collapse multiple spaces
   - Drop empty lines
   - Optionally remove markup tokens if using Wikitext
   - Enforce `min_words <= len(text.split()) <= max_words`

5) Calibration runtime controls:
   - Use a single voice prompt (default) first
   - Optionally add `--voices` to calibrate on 2–3 predefined voices (future)

Acceptance criteria:
   - Calibrated checkpoint consistently uses the static path (if backend supports it)
   - Quality does not regress significantly vs non-static int8 on a short prompt set

---

### Workstream D — Migrate Quantization to TorchAO (torchao)

Goal: replace the homegrown quantization machinery with TorchAO to (a) simplify code and (b) unlock ARM-friendly kernels + Inductor optimizations.

#### D1) First milestone: in-memory TorchAO quantization (no new checkpoint format)

Pros:
   - Minimal changes to weights packaging
   - Easy to A/B benchmark

Cons:
   - Does not reduce download size (weights are quantized after load)

Plan:

1) Add `torchao` as an optional dependency (and pin a minimum version that matches docs, e.g. `>=0.15`).
2) Extend `QuantizationConfig` (`pocket_tts/utils/config.py`) to support TorchAO modes, e.g.:
   - `mode: torchao_int8_weight_only`
   - `mode: torchao_int8_dynamic_w8a8`
3) Implement `pocket_tts/utils/torchao_quantization.py`:
   - `apply_torchao_quantization_(model, mode, scope)`
   - Use `torchao.quantization.quantize_` with:
     - `Int8WeightOnlyConfig()` OR `Int8DynamicActivationInt8WeightConfig()`
   - Use `filter_fn` to limit to FlowLM modules (reuse `FLOW_LM_PREFIXES` logic)
4) Wire into `TTSModel._from_pydantic_config_with_weights()`:
   - After loading weights (initially), call TorchAO quantize on `tts_model.flow_lm` (or whole model)
5) Benchmark both TorchAO modes vs current custom int8 on M2.

Notes:
   - TorchAO expects best performance with `torch.compile`. We will not assume speedup without it.

#### D2) Second milestone: TorchAO + `torch.compile` (targeting M2 speed)

Plan:

1) Add a benchmarking knob for compilation:
   - env var `POCKET_TTS_COMPILE=1` or CLI flag `--compile` in `scripts/benchmark.py`
2) Decide compilation boundary:
   - Start with compiling **FlowLM forward** (or the transformer backbone) only.
   - Avoid compiling the multithreaded streaming wrapper initially.
3) Use TorchAO’s recommended Inductor config setter when enabled (per TorchAO config docs).
4) Measure:
   - time-to-first-audio
   - steady-state step time
   - overall RTF

Risks:
   - Dynamic shapes/stateful caches may trigger recompiles or graph breaks.
   - Compilation overhead may dominate short prompts.

#### D3) Third milestone: smaller downloads (optional; choose one)

Option 1 (TorchAO-native): save quantized weights with `torch.save(state_dict)`
   - Use TorchAO “tensor subclass” serialization (`docs.pytorch.org/ao/stable/serialization.html`)
   - Add config support for `.pt` weight files (in addition to safetensors)

Option 2 (keep safetensors): keep our current safetensors int8 format
   - Use TorchAO for runtime kernels/compute only
   - Keep `scripts/quantize_weights.py` until we replace safetensors flow

Recommendation:
   - Start with Option 1 only if we can accept `.pt` weights distribution.
   - Otherwise keep safetensors and focus on runtime speed first.

---

## 5) Benchmarks + Evaluation Protocol

### Standard benchmark command set (on your M2)

Use `scripts/benchmark.bash` with fixed prompts (already present).

Run matrix:

1) Baseline float32
2) Custom int8 (current)
3) Custom int8 + calibrated static
4) BF16 (float weights or bf16 weights; depending on implementation)
5) TorchAO int8 weight-only (no compile)
6) TorchAO int8 weight-only + compile
7) TorchAO int8 dynamic W8A8 + compile

### What to record

- `rtf_median`, `time_sec_median`, `time_sec_p90` from `scripts/benchmark.py`
- Audio artifacts in `audios/<run_id>/...wav` for quick listening
- Quantization debug summary (kernel path counts)
- Memory snapshots:
  - `size_of_dict(tts.state_dict())`
  - `scripts/kv_cache_report.py` output (for state size)

Acceptance criteria (initial):

- “No worse than baseline” target:
  - RTF does not drop by >5% vs float32 baseline on the same prompt set.
- Memory target:
  - Demonstrate at least one measurable reduction (BF16 weights or trimmed cached states).

---

## 6) Concrete File Change List (for implementation pass)

New / modified files likely needed:

- Modify:
  - `pocket_tts/models/tts_model.py` (dtype handling, TorchAO hook, cached-state trimming)
  - `pocket_tts/utils/config.py` (new quantization modes, maybe dtype fields if expanded)
  - `scripts/ptq_static_calibrate.py` (dataset/text-file loader, 100–150 examples)
  - `scripts/benchmark.py` (optional: compile flag + quant debug printing)
  - `pocket_tts/modules/quant_linear.py` (debug counters + clearer fallback logging)
- Add:
  - `pocket_tts/utils/torchao_quantization.py` (apply_torchao_quantization_)
  - `pocket_tts/utils/state_trim.py` (trim_state_inplace)
  - `pocket_tts/config/b6369a24_bf16.yaml` (and optional int8_bf16 variants)
- Optional:
  - `scripts/torchao_quantize_weights.py` (if we decide to produce `.pt` weights)

---

## 7) Open Questions / Decisions Needed Before Coding

1) Do we accept distributing quantized weights as `.pt` (torch.save) files, or must we keep safetensors only?
2) For BF16:
   - Are we targeting memory-only wins, or do we require speed wins too?
3) For calibration text dataset:
   - Is adding `datasets` dependency acceptable (even as optional)?
   - Do we want a committed local text file to avoid network?
4) How much engineering time do we want to spend on `torch.compile` stability for a streaming, stateful model?

