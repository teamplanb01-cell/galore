# Local GaLore Benchmark (Offline)

This repo includes a self-contained benchmark harness you can run on your laptop (CPU/MPS/CUDA) to compare **baseline AdamW** vs **GaLoreAdamW** on a tiny LLaMA config **without downloading any datasets**.

It produces:
- A human-readable **Markdown report**
- A machine-readable **JSON results file**

## What was added

- `scripts/local_benchmark.py`: Offline benchmark runner (synthetic tokens, tiny LLaMA, rank sweep).
- `galore_torch/diagnostics.py`: Utilities to measure model/optimizer memory from PyTorch objects.
- `tests/test_diagnostics.py`: Minimal `unittest` coverage for diagnostics + GaLore state size.

In addition, `galore_torch` is now more **import-safe** when optional dependencies aren’t installed:
- `bitsandbytes` is only required if you use `GaLoreAdamW8bit`.
- `tensorly` is only required if you use GaLore tensor projection (dim > 2) via `GaLoreProjectorTensor`.

## Quickstart (no internet, no dataset)

From the repo root:

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Outputs are written to:

`reports/runs/<timestamp>/report.md`  
`reports/runs/<timestamp>/results.json`  
`reports/runs/<timestamp>/run_config.json`

## How it works

### Model
- Loads a local config JSON (default: `configs/llama_9m.json`) via `AutoConfig.from_pretrained(<path>)`.
- Instantiates the model via `AutoModelForCausalLM.from_config(config)` (no downloads).
- Ensures runs are comparable by reusing the same base model weights for each method.

### Data: “repeat-biased” synthetic tokens
The benchmark generates offline token batches on CPU with a fixed seed. For each timestep, the next token repeats the previous one with high probability (default 0.85); otherwise it’s random. This creates simple structure so the loss is non-trivial, while remaining fully offline.

### What gets GaLore by default
For GaLore runs, parameters are selected by scanning `model.named_modules()` and including `nn.Linear.weight` when the module name contains any token in:

`--target_modules attn,mlp` (default)

### Metrics recorded
For each run (AdamW baseline + each GaLore rank), the script records:
- Step timing: `avg_step_time_ms`, `p50_step_time_ms`, `p90_step_time_ms`
- Throughput: `tokens_per_sec`
- Optimizer memory: `optimizer_state_mb` (sum of tensor storage in `optimizer.state`)
- GaLore projector memory: `galore_projector_mb` (tensor storage under `projector.ortho_matrix`)
- Peak device memory:
  - MPS: `torch.mps.current_allocated_memory()` (tracked max)
  - CUDA: `torch.cuda.max_memory_allocated()` (tracked max)
  - CPU: optional RSS via `psutil` if installed
- Loss: `avg_loss`, `final_loss`

Timing uses `torch.mps.synchronize()` / `torch.cuda.synchronize()` when applicable so measurements reflect real device execution.

## Recommended runs

Default sweep (quick and employer-friendly):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Short smoke test (fast):

```bash
python3 scripts/local_benchmark.py --device cpu --batch_size 2 --seq_len 64 --warmup_steps 2 --steps 5 --ranks 8 16
```

## Interpreting results

- **Optimizer state MB** should be **lower** for GaLore vs AdamW when GaLore is enabled on some layers.
- **Lower rank** should generally use **less** optimizer-state memory (and also smaller projector memory).
- Step time may change because GaLore periodically updates a projector (SVD-based) and changes per-step compute.

## Tests

Run unit tests from the repo root:

```bash
python3 -m unittest -q
```

## Troubleshooting

- If `--device auto` picks CPU on Apple Silicon, verify MPS support in your environment:
  - `python3 -c "import torch; print(torch.backends.mps.is_available())"`
- If the benchmark is slow, reduce `--steps`, `--seq_len`, or `--batch_size`.
