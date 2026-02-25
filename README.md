# GaLore (fork): offline benchmark suite and projector improvements

This repository is a fork of [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore). The fork has two goals:

1. Provide a **fully offline, reproducible** benchmark to quantify GaLore’s memory/throughput trade-offs on CPU/MPS/CUDA.
2. Introduce a **projector implementation improvement** by making the SVD backend explicit (full vs randomized/low-rank).

## Introduction

AdamW maintains first- and second-moment estimates for each parameter tensor. For large models, optimizer state can be a substantial component of training memory. GaLore mitigates this by projecting gradients for selected parameters into a rank-`r` subspace, maintaining AdamW moments in that subspace, and projecting the update back to the original parameter shape. The rank `r` is the primary control knob governing memory reduction and projection overhead.

## Gap addressed

In the upstream codebase, it is non-trivial to evaluate GaLore locally in a controlled manner because typical workflows depend on external assets (datasets and/or pretrained weights) and do not standardize:

- how optimizer-state memory is measured,
- how timing is reported across devices,
- how runs are documented for reproducibility.

## Method (changes and additions)

### Offline benchmark harness

- Runner: `scripts/local_benchmark.py`
- Model: tiny LLaMA instantiated from `configs/llama_9m.json` via `AutoModelForCausalLM.from_config` (no downloads).
- Data: deterministic “repeat-biased” synthetic token batches generated from a fixed seed (offline).
- Comparison: `torch.optim.AdamW` vs `GaLoreAdamW` over a rank sweep (default: 8/16/32/64) with identical initialized weights.
- Protocol: warm-up steps excluded from timing; multiple trials reported as mean ± standard deviation; device synchronization used for accurate timing (`torch.mps.synchronize()` / `torch.cuda.synchronize()`).
- Artifacts (per run directory): `report.md`, `results.json`, `run_config.json`.

### Diagnostics utilities

- Library: `galore_torch/diagnostics.py`
- Implements reusable measurements for:
  - model parameter bytes,
  - optimizer state tensor bytes (recursive traversal of `optimizer.state`),
  - GaLore projector tensor bytes (tensors reachable from `projector.ortho_matrix`).

### Projector improvement (GaLore-side code change)

- Implementation: `galore_torch/galore_projector.py`
- Adds an explicit SVD backend selection:
  - `svd_method=full`: `torch.linalg.svd`
  - `svd_method=randomized`: `torch.svd_lowrank`, with fallback to full SVD when unsupported or not beneficial
- Corrects device placement by using `.to(tensor.device)` (instead of `.to(tensor.device.type)`), which preserves device identity in multi-device settings.
- The benchmark and optimizers accept projector controls via param-group keys: `svd_method`, `rsvd_oversample`, `rsvd_n_iter`.

### Tests

- `tests/test_diagnostics.py`: GaLore optimizer state < AdamW; projector memory non-zero.
- `tests/test_projector.py`: randomized SVD mode shape checks and rank validation.

Optional dependencies remain optional:

- `bitsandbytes` is only required for `GaLoreAdamW8bit`.
- `tensorly` is only required for tensor projection (`GaLoreProjectorTensor`, i.e., dim > 2).

## Results (Apple Silicon / MPS, real run)

Published run directory: `reports/published/20260225_184856/`

Repro command (embedded in the report):

```bash
python3 scripts/local_benchmark.py --device auto --output_root reports/published --svd_method full
```

Configuration summary:
- Device: `mps`, dtype: `float32`
- Batch size: `4`, sequence length: `128`
- Warm-up steps: `10`, measured steps: `50`, trials: `3`
- GaLore target selection: `target_modules=attn,mlp`
- Projector: `proj_type=std`, `svd_method=full`
- GaLore coverage: 28 Linear modules; 802,816 weights (≈ 8.9% of model parameters)

| method | rank | target_modules | avg_step_ms | tokens/s | opt_state_MB | projector_MB | peak_mps_MB | avg_loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| adamw | - | - | 29.910 ± 1.115 | 17134.263 ± 646.677 | 68.634 | - | 228.452 | 10.384 ± 0.000 |
| galore_adamw | 8 | attn,mlp | 35.598 ± 1.250 | 14394.421 ± 495.495 | 62.892 | 0.109 | 229.934 | 10.384 ± 0.000 |
| galore_adamw | 16 | attn,mlp | 36.319 ± 1.337 | 14110.509 ± 530.522 | 63.274 | 0.219 | 230.896 | 10.385 ± 0.000 |
| galore_adamw | 32 | attn,mlp | 36.263 ± 0.316 | 14119.721 ± 122.326 | 64.040 | 0.438 | 231.367 | 10.384 ± 0.000 |
| galore_adamw | 64 | attn,mlp | 37.632 ± 1.168 | 13614.314 ± 427.385 | 65.571 | 0.875 | 231.571 | 10.388 ± 0.000 |

Summary:

- Optimizer-state memory decreases under GaLore (e.g., rank 8: 68.634 → 62.892 MB; −5.742 MB, −8.4%).
- Lower ranks yield larger optimizer-state reductions; projector memory increases with rank (0.109 → 0.875 MB).
- Throughput decreases in this configuration due to projection/SVD overhead (rank 8 tokens/s: 17134 → 14394; −16.0%).
- `peak_mps_MB` is a coarse device-level metric; the optimizer-state measurements (`opt_state_MB`) are the direct signal for GaLore’s intended savings.

## How to run (offline)

Install benchmark dependencies:

```bash
python3 -m pip install torch transformers
```

Default run (auto-selects MPS if available, otherwise CPU):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Randomized/low-rank SVD projector:

```bash
python3 scripts/local_benchmark.py --device auto --svd_method randomized --rsvd_oversample 8 --rsvd_n_iter 1
```

To publish results into version-controlled directories:

```bash
python3 scripts/local_benchmark.py --device auto --output_root reports/published
```

See `README_LOCAL_BENCHMARK.md` for benchmark internals and metric definitions.

## Tests

```bash
python3 -m unittest -q
```

## Future work

- Add an explicit A/B comparison report for `svd_method=full` vs `svd_method=randomized` under identical settings.
- Add plotting utilities from `results.json` (rank vs memory/throughput).
- Add additional offline model configurations (e.g., 50–100M parameters) to increase measurement sensitivity while remaining laptop-friendly.
- Explore per-layer rank selection policies.

## Attribution

- Original repository: [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore)
- Paper: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
