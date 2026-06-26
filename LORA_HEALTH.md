# LoRA training-health check — is the adapter firing & storing?

Two distinct questions for any QLoRA run:
- **Firing** — did the adapter actually *learn* (vs. dead/frozen)?
- **Storing** — are the checkpoints saved correctly (loadable, consistent, resumable)?

This doc holds the diagnostic method, the healthy ranges, and the **cooking baseline** to compare future runs against (e.g. the diverse-50-domain run).

Reproduce with (CPU-only, safe while a GPU run is active):
```
CUDA_VISIBLE_DEVICES="" python scripts/lora_health_check.py <results_dir> [--base <Wan transformer dir>]
```

## What "well-learning" looks like (the signals, not an absolute number)

There is **no universal ideal value for ‖ΔW‖** (Frobenius norm of the update `ΔW = B·A`). It scales with layer size, base-weight magnitude, LR, rank, and α. Judge these instead:

1. **0 dead layers** — every `lora_B` must be ≠ 0. (LoRA inits `B=0`, so ΔW=0 until B moves; if B stays 0 the layer never fired.) **Binary, must hold.**
2. **Relative update `‖ΔW‖/‖W_base‖`** — the metric with an interpretable band:
   - `< ~1%` → barely adapting (underfit / LR too low / too few steps)
   - **`~1–20%` → healthy** (adapts while preserving the base video prior)
   - `> ~50%` → overwriting the base (forgetting / instability; LR or α/r too high)
3. **Monotonic growth that *decelerates* toward a plateau** — a converged run's ‖ΔW‖ rises then flattens. Still-rising-linearly = not yet converged.
4. **Per-layer spread** max/mean ≈ 3–5 (a few "hot" layers + many moderate). Uniform tiny updates = nothing learned.
5. **Loss/val is the final arbiter** — ‖ΔW‖ is a *process* metric; loss below the predict-zero baseline (~1.55) is the *outcome*.

Storing: all checkpoints load cleanly, consistent size/param-count, no NaN/Inf, `optimizer.bin`+`scheduler.bin`+`random_states` present (exact-resumable).

## BASELINE — cooking, rank-128 (`EgoX/results/EgoX_cooking_r128`), VERIFIED 2026-06-25

320 adapted linears. `ΔW = B·A` (α/r scaling = 1.0).

| ckpt | #layers | B==0 (dead) | mean‖ΔW‖ | max‖ΔW‖ | total |
|------|---------|-------------|----------|---------|-------|
| 100  | 320 | 0 | 0.197 | 0.49 | 63 |
| 300  | 320 | 0 | 0.360 | 1.24 | 115 |
| 600  | 320 | 0 | 0.519 | 1.97 | 166 |
| 1000 | 320 | 0 | 0.687 | 2.64 | 220 |

**Verdict:** firing well — 0 dead layers, monotonic growth, max/mean ≈ 3.8 (healthy spread). **But ‖ΔW‖ was still rising ~linearly at step 1000** (increments 0.16/0.16/0.17, not decelerating) → the run **had not converged** when stopped — consistent with the undertrained yellow-cast output. More steps would have kept helping.

Storing: all 10 checkpoints (100–1000) loaded cleanly, consistent 1.68 GB / 320 layers, monotonic — no corruption/reset.

> TODO: relative `‖ΔW‖/‖W_base‖` % not yet computed (needs `--base` against the Wan transformer snapshot) — run once to place the baseline in the 1–20% band.

## diverse-50-domain run — TO FILL IN AFTER TRAINING

Run `scripts/lora_health_check.py <diverse50 results dir> --base <Wan transformer>` and paste the table here. Compare against the cooking baseline:
- Same/lower B==0 (expect 0).
- ‖ΔW‖ trajectory: does it **plateau** (converged) or keep rising (undertrained, like cooking)?
- Relative % in the 1–20% band?
- More diverse data → expect a *different* per-layer hot-spot distribution than cooking-only.
