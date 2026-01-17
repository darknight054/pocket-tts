# BF16 Optimization Plan (Low Risk)

Goal: reduce runtime memory by keeping weights in bfloat16 while preserving audio quality by
keeping numerically sensitive computation in float32.

## Why this helps
- Checkpoints are stored in bfloat16; today we load into float32, so we lose the memory win.
- Keeping weights in bfloat16 cuts parameter memory roughly in half.
- On CPU (Apple M‑series), bf16 compute speedups are not guaranteed; plan focuses on memory first.

## Minimal, low‑risk approach
1) **Weights in BF16, compute mostly FP32**
   - Cast model parameters to bf16 after loading.
   - Keep attention, norms, EOS logic, and LSD flow math in float32.
   - Accept minor casting overhead; prioritize stability.

2) **Limit scope to FlowLM first**
   - Apply bf16 to FlowLM weights only.
   - Leave Mimi/SEANet in float32 initially to avoid audio quality regressions.

3) **Explicit casts at sensitive points**
   - SDPA inputs and LayerNorms remain float32.
   - EOS head logits and thresholding remain float32.
   - Flow integration (LSD decode) remains float32.

## Validation
- Run `scripts/compare_memory.py` with a fixed prompt set.
- Listen to outputs for artifacts and EOS stability.
- Report:
  - prompt state MB reduction
  - median RTF vs baseline
  - subjective audio quality

## Rollout steps
1) Add a bf16 config variant (`b6369a24_bf16.yaml`).
2) Add a safe bf16 cast helper (`dtype_utils.py`) and call it after loading weights.
3) Keep all safety‑critical computations in float32; verify with short + long prompts.
4) If stable, optionally extend bf16 to Mimi weights.
