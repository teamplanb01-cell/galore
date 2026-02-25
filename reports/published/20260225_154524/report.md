# GaLore Local Benchmark Report

- Timestamp: `20260225_154524`
- Device: `cpu`
- Dtype: `float32`
- Torch: `2.9.1`
- Transformers: `4.57.6`
- Python: `3.13.9 (v3.13.9:8183fa5e3f7, Oct 14 2025, 10:27:13) [Clang 16.0.0 (clang-1600.0.26.6)]`
- Platform: `macOS-15.5-arm64-arm-64bit-Mach-O`
- Model params: `8,995,968` (34.3 MB)
- Trials per run: `3`

## Repro command

```bash
python3 /Users/sakshamkapoor/Downloads/GaLore-master/scripts/local_benchmark.py --device cpu --dtype float32 --batch_size 2 --seq_len 128 --warmup_steps 10 --steps 50 --ranks 8 16 32 64 --trials 3 --output_root reports/published
```

## Results

| method | rank | target_modules | avg_step_ms | tokens/s | opt_state_MB | projector_MB | peak_mps_MB | avg_loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| adamw | - | - | 33.621 ± 2.225 | 7635.762 ± 487.126 | 68.634 | - | - | 10.392 |
| galore_adamw | 8 | attn,mlp | 35.505 ± 1.200 | 7215.871 ± 246.003 | 62.892 | 0.109 | - | 10.397 |
| galore_adamw | 16 | attn,mlp | 34.624 ± 0.265 | 7394.014 ± 56.706 | 63.274 | 0.219 | - | 10.397 |
| galore_adamw | 32 | attn,mlp | 34.976 ± 0.438 | 7319.976 ± 91.350 | 64.040 | 0.438 | - | 10.394 |
| galore_adamw | 64 | attn,mlp | 35.194 ± 0.559 | 7275.141 ± 115.961 | 65.571 | 0.875 | - | 10.393 |

## Interpretation

- Optimizer state: AdamW = **68.634 MB**, GaLore(rank=8) = **62.892 MB**.
- As rank decreases, GaLore optimizer-state memory should decrease (smaller exp_avg/exp_avg_sq).
- GaLore may trade some speed for memory due to periodic SVD-based projector updates.
- Note: `peak_mps_MB` is only meaningful on `mps` runs; use JSON for CPU RSS / CUDA peaks.
