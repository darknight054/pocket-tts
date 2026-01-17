# Approach 2 (Weight-Only INT8) - Findings and Gaps

## Current State (Implemented)
- We generate a separate int8 checkpoint with per-row scales for FlowLM Linear layers (transformer + flow MLP).
- QuantLinear uses INT8 kernels via `quantized.linear_dynamic` when a quantized backend is available; otherwise it falls back to dequantized float.
- This reduces checkpoint size and RAM footprint for those Linear weights, but does not guarantee faster inference.

## What’s Missing for “Faster” Inference
1) **Dynamic quantization overhead**
   - The current kernel path uses dynamic activation quantization (activations are quantized on-the-fly for compute).
   - PyTorch notes this adds overhead and static quantization is typically faster because it avoids float<->int conversions between ops.

2) **Static PTQ pipeline + calibration**
   - Static post‑training quantization needs representative calibration data and (optionally) module fusion.
   - We do not yet perform calibration or fusion, so we’re not accessing the faster static‑quantized path.

3) **Backend and CPU‑specific kernel availability**
   - PyTorch uses different backends: QNNPACK for ARM; FBGEMM/X86 for x86 (with the newer x86 backend integrating oneDNN).
   - Speedups are highly backend‑ and shape‑dependent. On ARM/QNNPACK the speed benefits for small batch, small matrices can be limited.

4) **Operator coverage limits**
   - Dynamic quantization primarily supports Linear/RNN layers.
   - MultiheadAttention is not supported in dynamic quantization, so attention math still runs in float.

## Summary
We already get the memory/download wins from a weight‑only int8 checkpoint, but **speedups require more than just int8 weights**. The missing pieces are a static PTQ path with calibration/fusion, plus backend‑appropriate kernels (x86/oneDNN on Intel; QNNPACK on ARM), and broader operator coverage beyond Linear layers.

## Static PTQ: Recommended Algorithm Choice
- **Backend‑default qconfig mapping**: Use `get_default_qconfig_mapping(<backend>)`, which is tuned for the selected backend.
- **Activation observers**: `HistogramObserver` searches for min/max that minimize quantization error vs float and is often preferred when outliers exist.
- **Weight observers**: Per‑channel min/max observers are commonly used for weights.

## Suggested Next Experiments
- Add a calibration + static PTQ path for FlowLM (potentially Mimi later).
- Evaluate performance on x86 with the x86 backend (if available), where PyTorch reports stronger INT8 speedups.
- Narrow quantization to layer subsets (e.g., only MLPs) if accuracy remains sensitive.

## What We Tried
- **Dynamic int8 (Approach 1)**: Removed due to poor latency and unstable quality metrics.
- **Weight‑only int8 (Approach 2)**: Offline quantization of FlowLM Linear layers with per‑row scales.
- **INT8 kernels (dynamic)**: `quantized.linear_dynamic` path for int8 matmul; improved vs dequant‑to‑float but still slower than float32 in our runs.
- **Static PTQ calibration path**: Added calibration + static kernel path (`quantized.linear`) with per‑tensor activation qparams (ready for testing).
- **KV cache sizing**: Reduced FlowLM/Mimi cache sizes to prompt/context‑based lengths (memory win, not latency‑driven).

## Potential Next Steps
- **Finish static PTQ evaluation**: Run calibrated static variant and compare against baseline; tune observer choice (Histogram vs MinMax).
- **Backend‑specific testing**: Benchmark on x86 (FBGEMM/x86 backend) to validate expected int8 speedups.
- **Layer‑subset quantization**: Quantize only MLPs (or only attention projections) to reduce overhead while preserving quality.
- **Per‑tensor weights**: Try per‑tensor weight quantization for faster packing (potentially lower quality) as a speed tradeoff.
- **Better similarity metric**: Use mel‑spec + alignment (or perceptual metrics) to avoid waveform mismatch issues.
## References
- PyTorch Quantization in Practice (dynamic vs static, overhead, calibration): https://pytorch.org/blog/quantization-in-practice/
- PyTorch INT8 x86 backend overview and speedups: https://pytorch.org/blog/int8-quantization/
- Intel x86 backend article (oneDNN + FBGEMM): https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-int8-inf-with-new-x86-backend.html
- PyTorch quantization docs (operator coverage, dynamic vs static): https://docs.pytorch.org/docs/stable/quantization
