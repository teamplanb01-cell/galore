# GaLore (Fork): Offline Benchmark + Diagnostics

This fork of [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore) adds a **clean, offline workflow** to study GaLore locally: a reproducible benchmark script, memory diagnostics, and minimal tests.

## Method

The benchmark (`scripts/local_benchmark.py`) trains a tiny causal LM for a few steps and reports timing + memory:

- **Model:** `AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(<local_json>))` (default: `configs/llama_9m.json`)
- **Data:** repeat-biased **synthetic tokens** generated offline (no dataset/model downloads)
- **Runs:** baseline **AdamW** vs **GaLoreAdamW** across a rank sweep (default: `8 16 32 64`)
- **GaLore targets:** apply GaLore to `nn.Linear.weight` where module name contains any of `--target_modules` (default: `attn,mlp`)
- **Key metrics:** `avg_step_time_ms`, `tokens_per_sec`, `optimizer_state_mb`, `galore_projector_mb`, peak device memory (MPS/CUDA; optional CPU RSS)

## Run

Install minimal deps:

```bash
python3 -m pip install torch transformers
```

Run (uses MPS if available, otherwise CPU):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Outputs:

- `reports/runs/<timestamp>/report.md`
- `reports/runs/<timestamp>/results.json`
- `reports/runs/<timestamp>/run_config.json`

More details: `README_LOCAL_BENCHMARK.md`.

## Results (example)

Example from a short CPU run (command: `--batch_size 2 --seq_len 32 --warmup_steps 1 --steps 2 --ranks 8 16 32 64`, targets: `attn,mlp`):

| method | rank | avg_step_ms | opt_state_MB | projector_MB |
|---|---:|---:|---:|---:|
| adamw | - | 25.235 | 68.634 | - |
| galore_adamw | 8 | 23.500 | 62.892 | 0.109 |
| galore_adamw | 16 | 23.930 | 63.274 | 0.219 |
| galore_adamw | 32 | 22.257 | 64.040 | 0.438 |
| galore_adamw | 64 | 23.140 | 65.571 | 0.875 |

Notes:
- This is a pipeline sanity-check run (2 measured steps). Use the defaults (`--steps 50`) for more stable timing.
- Expected trend: **lower rank → lower optimizer state**, while projector memory grows with rank.

## Conclusion

- This fork makes GaLore easy to **evaluate locally** with reproducible reports.
- The benchmark quantifies the tradeoffs between **rank**, **optimizer-state memory**, **projector overhead**, and **step time**.

## Where the additions live

- Benchmark runner: `scripts/local_benchmark.py`
- Diagnostics: `galore_torch/diagnostics.py`
- Tests: `tests/test_diagnostics.py`

## Tests

```bash
python3 -m unittest -q
```

## Optional dependencies

Not required for the offline benchmark:
- `bitsandbytes` (only for `GaLoreAdamW8bit`)
- `tensorly` (only for tensor projection via `GaLoreProjectorTensor`, i.e., dim > 2)

## Attribution

- Original repo: [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore)
- Paper: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)

