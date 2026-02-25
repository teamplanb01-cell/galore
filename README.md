# GaLore (Fork): Offline Benchmark + Projector Improvements

This repo is a fork of [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore). This fork focuses on two things:

1. **Make GaLore easy to run and measure locally (offline)** on CPU / Apple Silicon MPS / CUDA.
2. **Improve the GaLore projector** with a configurable SVD method (full vs randomized/low‑rank).

---

## Intro (what is GaLore?)

When you train with **Adam/AdamW**, the optimizer stores extra tensors for each weight (moving averages like `exp_avg` and `exp_avg_sq`). For large models, that optimizer “state” can be a big part of training memory.

**GaLore** reduces optimizer-state memory for selected layers by:

- projecting a layer’s gradient into a **low-rank** space (rank = 8/16/32/64…),
- storing Adam’s moments **in that smaller space**,
- projecting the update back to the original shape.

The usual tradeoff:
- **Lower optimizer-state memory** (especially at small ranks),
- **Extra compute** to update/maintain the projector.

---

## Gap (what was missing / hard in upstream)

Upstream GaLore is research code and assumes you already have a full training setup. On a local laptop, it’s hard to:

- run a **fully offline** comparison (no dataset/model downloads),
- get a **reproducible report** (numbers + settings + versions),
- measure optimizer-state memory in a consistent way,
- experiment with projector update cost (SVD) and device handling safely.

---

## Method (what we added/changed)

### 1) Offline local benchmark runner

- **CLI:** `scripts/local_benchmark.py`
- **Tiny model config:** `configs/llama_9m.json` (≈ 9M params, built locally from config; no downloads)
- **Data:** deterministic “repeat-biased” synthetic token batches (offline, but structured enough for loss to move)
- **Comparison:** baseline `torch.optim.AdamW` vs `GaLoreAdamW` over a **rank sweep** (default: 8/16/32/64)
- **Metrics captured:**
  - speed: `avg_step_ms` (mean ± std over trials), `tokens/s`
  - optimizer memory: `optimizer_state_mb` (tensor bytes inside `optimizer.state`)
  - GaLore projector memory: `projector_mb` (tensor bytes under `projector.ortho_matrix`)
  - device memory: `peak_mps_MB` (or CUDA peak if on CUDA)
  - sanity: `avg_loss`
- **Outputs:** `report.md`, `results.json`, `run_config.json` under `reports/runs/<timestamp>/` or `reports/published/<timestamp>/`

### 2) Diagnostics helpers

- **Library:** `galore_torch/diagnostics.py`
- Adds reusable functions to measure:
  - model parameter bytes,
  - optimizer state tensor bytes (recursive),
  - GaLore projector tensor bytes.

### 3) GaLore projector improvement (real optimizer-side change)

- **Implementation:** `galore_torch/galore_projector.py`
- Adds `svd_method`:
  - `full` (default): `torch.linalg.svd`
  - `randomized`: `torch.svd_lowrank` (with safe fallback to full SVD)
- Fixes device placement by using `.to(tensor.device)` (the original pattern `.to(tensor.device.type)` can be wrong on multi-device setups).
- Plumbed through optimizers and the benchmark via param-group keys:
  - `svd_method`, `rsvd_oversample`, `rsvd_n_iter`

### 4) Tests (built-in `unittest`)

- `tests/test_diagnostics.py`: GaLore state < AdamW; projector memory non-zero.
- `tests/test_projector.py`: randomized SVD works on tall/wide matrices; rank bounds validated.

### 5) Optional dependencies stay optional

This fork is more “import-safe” for local use:

- `bitsandbytes` is only needed if you use `GaLoreAdamW8bit`.
- `tensorly` is only needed if you use tensor projection (`GaLoreProjectorTensor`, i.e., dim > 2).

---

## Results (real run on Apple Silicon / MPS)

Published run directory: `reports/published/20260225_184856/`

Repro command (embedded in the report):

```bash
python3 scripts/local_benchmark.py --device auto --output_root reports/published --svd_method full
```

Key settings:
- `device=mps`, `dtype=float32`, `batch_size=4`, `seq_len=128`
- `warmup_steps=10`, `steps=50`, `trials=3`
- GaLore targets: `target_modules=attn,mlp`
- Projector: `proj_type=std`, `svd_method=full`
- GaLore coverage: `28` Linear modules, `802,816` weights (≈ `8.9%` of model parameters)

| method | rank | target_modules | avg_step_ms | tokens/s | opt_state_MB | projector_MB | peak_mps_MB | avg_loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| adamw | - | - | 29.910 ± 1.115 | 17134.263 ± 646.677 | 68.634 | - | 228.452 | 10.384 ± 0.000 |
| galore_adamw | 8 | attn,mlp | 35.598 ± 1.250 | 14394.421 ± 495.495 | 62.892 | 0.109 | 229.934 | 10.384 ± 0.000 |
| galore_adamw | 16 | attn,mlp | 36.319 ± 1.337 | 14110.509 ± 530.522 | 63.274 | 0.219 | 230.896 | 10.385 ± 0.000 |
| galore_adamw | 32 | attn,mlp | 36.263 ± 0.316 | 14119.721 ± 122.326 | 64.040 | 0.438 | 231.367 | 10.384 ± 0.000 |
| galore_adamw | 64 | attn,mlp | 37.632 ± 1.168 | 13614.314 ± 427.385 | 65.571 | 0.875 | 231.571 | 10.388 ± 0.000 |

What this shows (plain English):

- **Optimizer-state memory drops with GaLore.** Example: AdamW `68.634 MB` → GaLore rank 8 `62.892 MB` (≈ **5.7 MB saved**).
- **Lower rank saves more optimizer-state memory** (rank 8 saves more than rank 64).
- **Projector overhead is small** here (≈ `0.1–0.9 MB`, increasing with rank).
- **Throughput is lower** in this benchmark because projector updates add work. This is expected; the point of the harness is to quantify this tradeoff on your own machine.
- `peak_mps_MB` changes only slightly because total allocated memory also includes model weights/activations; `opt_state_MB` is the more direct measurement of optimizer-state savings.

---

## Run it locally (offline)

Install only what the benchmark needs:

```bash
python3 -m pip install torch transformers
```

Run the default sweep (uses MPS if available, otherwise CPU):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Try the new projector option:

```bash
python3 scripts/local_benchmark.py --device auto --svd_method randomized --rsvd_oversample 8 --rsvd_n_iter 1
```

Outputs go to:
- `reports/runs/<timestamp>/report.md`
- `reports/runs/<timestamp>/results.json`
- `reports/runs/<timestamp>/run_config.json`

If you want to commit results, use:

```bash
python3 scripts/local_benchmark.py --device auto --output_root reports/published
```

For a deeper explanation of the benchmark internals, see `README_LOCAL_BENCHMARK.md`.

---

## Conclusion

This fork makes GaLore measurable and reproducible on a laptop (offline) and adds a real GaLore-side improvement: a configurable projector SVD method (`full` vs `randomized`) plus safer device placement.

---

## Future work (good next “real” additions)

- Add an automatic **A/B comparison** in the report: `svd_method=full` vs `randomized` on the same run.
- Add a `scripts/plot_results.py` that generates a single plot (rank vs memory/speed).
- Add a slightly larger local model config (e.g., ~50–100M params) to make optimizer-state deltas more obvious.
- Add an optional “real data” mode (still small) to track loss curves, while keeping the default fully offline.
- Explore adaptive rank / per-layer rank schedules based on layer shape or gradient statistics.

---

## Attribution

- Original repo: [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore)
- Paper: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
