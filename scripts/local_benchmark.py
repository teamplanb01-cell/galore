from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from galore_torch import GaLoreAdamW
from galore_torch.diagnostics import (
    format_bytes,
    galore_projector_nbytes,
    model_param_nbytes,
    optimizer_state_tensor_nbytes,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GaLore vs AdamW local benchmark (tiny LLaMA).")
    parser.add_argument("--model_config", type=str, default="configs/llama_9m.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--update_proj_gap", type=int, default=10)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--target_modules", type=str, default="attn,mlp")

    parser.add_argument("--output_root", type=str, default="reports/runs")
    return parser.parse_args(argv)


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device=mps, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cpu")


def _pick_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_arg}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _maybe_get_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def _percentile_ms(samples_ms: list[float], percentile: float) -> float:
    if not samples_ms:
        return float("nan")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100]")
    sorted_samples = sorted(samples_ms)
    k = (len(sorted_samples) - 1) * (percentile / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_samples) - 1)
    if lo == hi:
        return float(sorted_samples[lo])
    w = k - lo
    return float(sorted_samples[lo] * (1.0 - w) + sorted_samples[hi] * w)


def _unique_run_dir(root: Path, timestamp: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / timestamp
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for i in range(1, 1000):
        candidate = root / f"{timestamp}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError("Could not create a unique run directory after 1000 attempts.")


def _generate_repeat_biased_tokens(
    *,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    repeat_prob: float = 0.85,
) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    if not (0.0 <= repeat_prob <= 1.0):
        raise ValueError("repeat_prob must be in [0, 1]")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    tokens = torch.empty((num_batches, batch_size, seq_len), dtype=torch.long)
    tokens[:, :, 0] = torch.randint(0, vocab_size, (num_batches, batch_size), generator=gen)
    for t in range(1, seq_len):
        repeat_mask = torch.rand((num_batches, batch_size), generator=gen) < repeat_prob
        random_tokens = torch.randint(0, vocab_size, (num_batches, batch_size), generator=gen)
        tokens[:, :, t] = torch.where(repeat_mask, tokens[:, :, t - 1], random_tokens)
    return tokens


def _collect_galore_params(model: torch.nn.Module, target_modules: list[str]) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    from torch import nn

    galore_params: list[torch.nn.Parameter] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(key in module_name for key in target_modules):
            continue
        galore_params.append(module.weight)

    id_galore_params = {id(p) for p in galore_params}
    regular_params = [p for p in model.parameters() if p.requires_grad and id(p) not in id_galore_params]
    galore_params = [p for p in galore_params if p.requires_grad]
    return galore_params, regular_params


def _run_training(
    *,
    method: str,
    rank: int | None,
    config,
    base_state_dict: dict[str, torch.Tensor],
    batches_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    weight_decay: float,
    update_proj_gap: int,
    galore_scale: float,
    proj_type: str,
    target_modules: list[str],
    warmup_steps: int,
    steps: int,
) -> dict:
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(base_state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batches = batches_cpu if device.type == "cpu" else batches_cpu.to(device)

    if method == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif method == "galore_adamw":
        if rank is None:
            raise ValueError("rank must be provided for galore_adamw")
        galore_params, regular_params = _collect_galore_params(model, target_modules)
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": rank,
                "update_proj_gap": update_proj_gap,
                "scale": galore_scale,
                "proj_type": proj_type,
            },
        ]
        optimizer = GaLoreAdamW(param_groups, lr=lr, weight_decay=weight_decay, no_deprecation_warning=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    step_times_ms: list[float] = []
    measured_losses: list[float] = []
    peak_mps_bytes = 0
    peak_rss_bytes = 0

    total_steps = warmup_steps + steps

    optimizer.zero_grad(set_to_none=True)
    for step_idx in range(total_steps):
        is_measured = step_idx >= warmup_steps

        if is_measured:
            _sync(device)
            t0 = time.perf_counter()

        input_ids = batches[step_idx]
        labels = input_ids
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if device.type == "mps":
            peak_mps_bytes = max(peak_mps_bytes, int(torch.mps.current_allocated_memory()))

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "mps":
            peak_mps_bytes = max(peak_mps_bytes, int(torch.mps.current_allocated_memory()))

        rss = _maybe_get_rss_bytes()
        if rss is not None:
            peak_rss_bytes = max(peak_rss_bytes, int(rss))

        if is_measured:
            _sync(device)
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1000.0)
            measured_losses.append(float(loss.detach().cpu().item()))

    if device.type == "cuda":
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_cuda_bytes = 0

    opt_state_bytes = optimizer_state_tensor_nbytes(optimizer)
    projector_bytes = galore_projector_nbytes(optimizer) if method == "galore_adamw" else 0

    avg_step_ms = float(sum(step_times_ms) / max(1, len(step_times_ms)))
    tokens_per_sec = float((batches.shape[1] * batches.shape[2]) / (avg_step_ms / 1000.0)) if avg_step_ms > 0 else float("nan")

    result = {
        "method": method,
        "rank": rank,
        "target_modules": ",".join(target_modules) if method == "galore_adamw" else "",
        "steps": steps,
        "warmup_steps": warmup_steps,
        "avg_step_time_ms": avg_step_ms,
        "p50_step_time_ms": _percentile_ms(step_times_ms, 50.0),
        "p90_step_time_ms": _percentile_ms(step_times_ms, 90.0),
        "tokens_per_sec": tokens_per_sec,
        "optimizer_state_bytes": opt_state_bytes,
        "optimizer_state_mb": opt_state_bytes / (1024.0 * 1024.0),
        "galore_projector_bytes": projector_bytes,
        "galore_projector_mb": projector_bytes / (1024.0 * 1024.0),
        "peak_mps_bytes": peak_mps_bytes,
        "peak_mps_mb": peak_mps_bytes / (1024.0 * 1024.0),
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_mb": peak_cuda_bytes / (1024.0 * 1024.0),
        "peak_rss_bytes": peak_rss_bytes if peak_rss_bytes > 0 else None,
        "peak_rss_mb": (peak_rss_bytes / (1024.0 * 1024.0)) if peak_rss_bytes > 0 else None,
        "avg_loss": float(sum(measured_losses) / max(1, len(measured_losses))),
        "final_loss": float(measured_losses[-1]) if measured_losses else float("nan"),
    }

    del optimizer, model, batches
    gc.collect()
    _empty_cache(device)
    return result


def _render_report_md(*, run_config: dict, results: list[dict]) -> str:
    now = run_config["timestamp"]
    device = run_config["device"]
    dtype = run_config["dtype"]
    model_params = run_config["model_total_params"]
    model_param_bytes = run_config["model_param_bytes"]
    cmd = run_config["repro_command"]

    lines: list[str] = []
    lines.append(f"# GaLore Local Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: `{now}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Dtype: `{dtype}`")
    lines.append(f"- Torch: `{run_config['torch_version']}`")
    lines.append(f"- Transformers: `{run_config['transformers_version']}`")
    lines.append(f"- Python: `{run_config['python_version']}`")
    lines.append(f"- Platform: `{run_config['platform']}`")
    lines.append(f"- Model params: `{model_params:,}` ({format_bytes(model_param_bytes)})")
    lines.append("")
    lines.append("## Repro command")
    lines.append("")
    lines.append("```bash")
    lines.append(cmd)
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| method | rank | target_modules | avg_step_ms | tokens/s | opt_state_MB | projector_MB | peak_mps_MB | avg_loss |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")

    def _fmt(x: float | None) -> str:
        if x is None:
            return "-"
        if isinstance(x, float) and (x != x):  # NaN
            return "-"
        return f"{x:.3f}"

    for r in results:
        method = r["method"]
        rank = r["rank"] if r["rank"] is not None else "-"
        targets = r["target_modules"] or "-"
        lines.append(
            f"| {method} | {rank} | {targets} | "
            f"{_fmt(r['avg_step_time_ms'])} | {_fmt(r['tokens_per_sec'])} | "
            f"{_fmt(r['optimizer_state_mb'])} | {_fmt(r['galore_projector_mb'] if method == 'galore_adamw' else None)} | "
            f"{_fmt(r['peak_mps_mb'] if device.startswith('mps') else None)} | "
            f"{_fmt(r['avg_loss'])} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    adamw = next((r for r in results if r["method"] == "adamw"), None)
    galore_runs = [r for r in results if r["method"] == "galore_adamw"]
    galore_runs_sorted = sorted(galore_runs, key=lambda x: int(x["rank"]))

    if adamw is not None and galore_runs_sorted:
        lines.append(
            f"- Optimizer state: AdamW = **{adamw['optimizer_state_mb']:.3f} MB**, "
            f"GaLore(rank={galore_runs_sorted[0]['rank']}) = **{galore_runs_sorted[0]['optimizer_state_mb']:.3f} MB**."
        )
        lines.append("- As rank decreases, GaLore optimizer-state memory should decrease (smaller exp_avg/exp_avg_sq).")
        lines.append("- GaLore may trade some speed for memory due to periodic SVD-based projector updates.")
        if not device.startswith("mps"):
            lines.append("- Note: `peak_mps_MB` is only meaningful on `mps` runs; use JSON for CPU RSS / CUDA peaks.")
    else:
        lines.append("- Compare AdamW vs GaLore ranks; see `results.json` for all metrics.")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _unique_run_dir(Path(args.output_root), timestamp)

    repro_command = " ".join([shlex.quote("python3"), shlex.quote(str(Path(__file__).as_posix()))] + [shlex.quote(a) for a in sys.argv[1:]])

    config = AutoConfig.from_pretrained(args.model_config)

    torch.manual_seed(args.seed)
    base_model = AutoModelForCausalLM.from_config(config)
    base_state_dict = base_model.state_dict()
    model_total_params = sum(p.numel() for p in base_model.parameters())
    model_param_bytes = model_param_nbytes(base_model, trainable_only=True)
    del base_model
    gc.collect()

    total_batches = args.warmup_steps + args.steps
    batches_cpu = _generate_repeat_biased_tokens(
        num_batches=total_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=int(getattr(config, "vocab_size", 32000)),
        seed=args.seed,
    )

    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]

    run_config = {
        "timestamp": timestamp,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "seed": args.seed,
        "model_config": args.model_config,
        "model_total_params": model_total_params,
        "model_param_bytes": model_param_bytes,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ranks": args.ranks,
        "update_proj_gap": args.update_proj_gap,
        "galore_scale": args.galore_scale,
        "proj_type": args.proj_type,
        "target_modules": target_modules,
        "repro_command": repro_command,
    }

    results: list[dict] = []

    results.append(
        _run_training(
            method="adamw",
            rank=None,
            config=config,
            base_state_dict=base_state_dict,
            batches_cpu=batches_cpu,
            device=device,
            dtype=dtype,
            lr=args.lr,
            weight_decay=args.weight_decay,
            update_proj_gap=args.update_proj_gap,
            galore_scale=args.galore_scale,
            proj_type=args.proj_type,
            target_modules=target_modules,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
    )

    for rank in args.ranks:
        results.append(
            _run_training(
                method="galore_adamw",
                rank=int(rank),
                config=config,
                base_state_dict=base_state_dict,
                batches_cpu=batches_cpu,
                device=device,
                dtype=dtype,
                lr=args.lr,
                weight_decay=args.weight_decay,
                update_proj_gap=args.update_proj_gap,
                galore_scale=args.galore_scale,
                proj_type=args.proj_type,
                target_modules=target_modules,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
            )
        )

    payload = {"run_config": run_config, "results": results}
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    (run_dir / "report.md").write_text(_render_report_md(run_config=run_config, results=results))

    print(f"Wrote report: {run_dir / 'report.md'}")
    print(f"Wrote results: {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
