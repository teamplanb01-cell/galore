# GaLore Local Benchmark Report

- Timestamp: `20260225_184856`
- Device: `mps`
- Dtype: `float32`
- Torch: `2.9.1`
- Transformers: `4.57.6`
- Python: `3.13.9 (v3.13.9:8183fa5e3f7, Oct 14 2025, 10:27:13) [Clang 16.0.0 (clang-1600.0.26.6)]`
- Platform: `macOS-15.5-arm64-arm-64bit-Mach-O`
- Model params: `8,995,968` (34.3 MB)
- Trials per run: `3`
- GaLore projector: `proj_type=std`, `svd_method=full`, `rsvd_oversample=8`, `rsvd_n_iter=1`

## Repro command

```bash
python3 /Users/sakshamkapoor/Downloads/GaLore-master/scripts/local_benchmark.py --device auto --output_root reports/published --svd_method full
```

## Results

| method | rank | target_modules | avg_step_ms | tokens/s | opt_state_MB | projector_MB | peak_mps_MB | avg_loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| adamw | - | - | 29.910 ± 1.115 | 17134.263 ± 646.677 | 68.634 | - | 228.452 | 10.384 ± 0.000 |
| galore_adamw | 8 | attn,mlp | 35.598 ± 1.250 | 14394.421 ± 495.495 | 62.892 | 0.109 | 229.934 | 10.384 ± 0.000 |
| galore_adamw | 16 | attn,mlp | 36.319 ± 1.337 | 14110.509 ± 530.522 | 63.274 | 0.219 | 230.896 | 10.385 ± 0.000 |
| galore_adamw | 32 | attn,mlp | 36.263 ± 0.316 | 14119.721 ± 122.326 | 64.040 | 0.438 | 231.367 | 10.384 ± 0.000 |
| galore_adamw | 64 | attn,mlp | 37.632 ± 1.168 | 13614.314 ± 427.385 | 65.571 | 0.875 | 231.571 | 10.388 ± 0.000 |

## Interpretation

- Optimizer state: AdamW = **68.634 MB**, GaLore(rank=8) = **62.892 MB**.
- As rank decreases, GaLore optimizer-state memory should decrease (smaller exp_avg/exp_avg_sq).
- GaLore may trade some speed for memory due to periodic SVD-based projector updates.
