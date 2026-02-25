# GaLore (Fork) — Offline Benchmark + Diagnostics

This fork of [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore) adds a **self-contained, offline** way to study GaLore locally: a reproducible benchmark script, memory accounting utilities, and minimal tests.

## What’s included

- **Offline benchmark runner:** `scripts/local_benchmark.py`
  - Builds a tiny LLaMA from a local config (default: `configs/llama_9m.json`)
  - Trains on **synthetic tokens** (no downloads)
  - Compares **AdamW** vs **GaLoreAdamW** across a rank sweep (default ranks: `8 16 32 64`)
  - Writes a Markdown report + JSON results
- **Diagnostics utilities:** `galore_torch/diagnostics.py`
  - `model_param_nbytes(model)`
  - `optimizer_state_tensor_nbytes(optimizer)`
  - `galore_projector_nbytes(optimizer)`
- **Tests:** `tests/test_diagnostics.py` (built-in `unittest`)

## Quickstart (no internet)

Install minimal dependencies for the benchmark:

```bash
python3 -m pip install torch transformers
```

Run the benchmark (uses MPS if available, otherwise CPU):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Outputs:

- `reports/runs/<timestamp>/report.md`
- `reports/runs/<timestamp>/results.json`
- `reports/runs/<timestamp>/run_config.json`

## Benchmark options (most useful flags)

```bash
# smaller/faster run
python3 scripts/local_benchmark.py --device cpu --batch_size 2 --seq_len 64 --warmup_steps 2 --steps 10 --ranks 8 16

# change which Linear layers get GaLore (substring match on module name)
python3 scripts/local_benchmark.py --target_modules attn,mlp

# change projection hyperparams
python3 scripts/local_benchmark.py --update_proj_gap 10 --galore_scale 1.0 --proj_type std
```

## What the report measures

Per run (baseline AdamW + each GaLore rank), the benchmark records:

- Timing: `avg_step_time_ms`, `p50_step_time_ms`, `p90_step_time_ms`
- Throughput: `tokens_per_sec`
- Memory:
  - `optimizer_state_mb` (tensor storage inside `optimizer.state`)
  - `galore_projector_mb` (tensor storage under `projector.ortho_matrix`, GaLore only)
  - Peak device memory (MPS/CUDA) + optional CPU RSS (if `psutil` is installed)
- Loss: `avg_loss`, `final_loss`

## Tests

```bash
python3 -m unittest -q
```

## Optional dependencies

These are **not required** for the offline benchmark:

- `bitsandbytes`: only needed for `GaLoreAdamW8bit`
- `tensorly`: only needed for tensor projection (`GaLoreProjectorTensor`, i.e., dim > 2)

## Attribution

- Original project: [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore)
- Paper: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
