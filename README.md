# GaLore (Fork): Offline Local Benchmark + Diagnostics

This repo is a fork of [jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore). The goal of this fork is simple:

**Make GaLore easy to understand and measure on a laptop** (CPU / Apple MPS / NVIDIA CUDA) with an **offline** benchmark and a clear report.

## What is GaLore (plain English)?

When you train a model with Adam/AdamW, the optimizer keeps extra “memory” for every weight (moving averages of gradients). For large models, that optimizer memory can be huge.

**GaLore** reduces that optimizer memory for selected layers by:

1. Taking the full gradient for a layer
2. Projecting it down into a **low‑rank** space (rank = 8, 16, 32, …)
3. Letting AdamW store its moving averages **in that smaller space**
4. Projecting the update back to the original weight shape

So the tradeoff is usually:
- **Less optimizer memory**, especially at small ranks
- **Some extra computation** to maintain the projection (“projector”)

## What we added in this fork

- **Offline benchmark runner:** `scripts/local_benchmark.py`
  - Builds a tiny LLaMA model from a local config (default: `configs/llama_9m.json`)
  - Trains on **synthetic tokens** (no dataset/model downloads)
  - Compares **AdamW** vs **GaLoreAdamW** across ranks (default: `8 16 32 64`)
  - Generates a Markdown report + JSON results
- **Memory diagnostics utilities:** `galore_torch/diagnostics.py`
  - Counts model parameter bytes, optimizer state bytes, and GaLore projector bytes
- **Minimal tests:** `tests/test_diagnostics.py` (built-in `unittest`)
  - Checks key invariants (GaLore state < AdamW; projector memory is non-zero)

## How the offline benchmark works (step-by-step)

1. **Create a tiny model** from `configs/llama_9m.json` (≈ 9M parameters).
2. **Generate fake “text”** as token IDs:
   - “repeat-biased” synthetic tokens (often repeats the previous token), so the loss is non-trivial but fully offline.
3. Run the same short training loop multiple times:
   - **AdamW baseline**
   - **GaLoreAdamW** at each rank (8, 16, 32, 64 by default)
4. Measure and save:
   - **Speed:** average step time and tokens/sec
   - **Memory (important):**
     - `optimizer_state_mb`: how much tensor memory the optimizer stores in `optimizer.state`
     - `galore_projector_mb`: how much tensor memory the GaLore projector stores (orthogonal matrices)
   - **Loss:** avg/final loss (just to confirm the loop is behaving)
5. Write a report to a timestamped folder.

## Run it (no internet)

Install minimal dependencies:

```bash
python3 -m pip install torch transformers
```

Run (uses **MPS** if available, otherwise CPU):

```bash
python3 scripts/local_benchmark.py --device auto --model_config configs/llama_9m.json
```

Outputs:

- `reports/runs/<timestamp>/report.md` (human-readable)
- `reports/runs/<timestamp>/results.json` (all numbers)
- `reports/runs/<timestamp>/run_config.json` (exact settings + versions)

Tip: if you want to confirm MPS is available:

```bash
python3 -c "import torch; print('mps available:', torch.backends.mps.is_available())"
```

## Results (example)

Below is an example from a short CPU sanity-check run (very few measured steps). Your exact numbers will vary by machine.

| method | rank | avg_step_ms | optimizer_state_mb | galore_projector_mb |
|---|---:|---:|---:|---:|
| adamw | - | 25.235 | 68.634 | - |
| galore_adamw | 8 | 23.500 | 62.892 | 0.109 |
| galore_adamw | 16 | 23.930 | 63.274 | 0.219 |
| galore_adamw | 32 | 22.257 | 64.040 | 0.438 |
| galore_adamw | 64 | 23.140 | 65.571 | 0.875 |

How to read this:

- **Optimizer state memory goes down** when GaLore is enabled on some layers (here: `attn,mlp`).
- **Smaller rank → less optimizer memory** (that’s the main knob GaLore gives you).
- The projector itself uses some memory (`galore_projector_mb`), and it grows with rank.

For more stable timing comparisons, run longer (defaults are already reasonable):

```bash
python3 scripts/local_benchmark.py --device auto --steps 50 --warmup_steps 10
```

## Common tweaks

```bash
# Faster run
python3 scripts/local_benchmark.py --device cpu --batch_size 2 --seq_len 64 --warmup_steps 2 --steps 10 --ranks 8 16

# Change which Linear layers get GaLore (substring match on module name)
python3 scripts/local_benchmark.py --target_modules attn,mlp

# Projection hyperparameters
python3 scripts/local_benchmark.py --update_proj_gap 10 --galore_scale 1.0 --proj_type std
```

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

