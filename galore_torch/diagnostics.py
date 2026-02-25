from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def model_param_nbytes(model: torch.nn.Module, trainable_only: bool = True) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel() * param.element_size()
    return total


def _tensor_tree_nbytes(obj: Any, *, seen: set[int]) -> int:
    if obj is None:
        return 0

    if isinstance(obj, torch.Tensor):
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        return obj.numel() * obj.element_size()

    if isinstance(obj, Mapping):
        return sum(_tensor_tree_nbytes(v, seen=seen) for v in obj.values())

    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_tensor_tree_nbytes(v, seen=seen) for v in obj)

    return 0


def optimizer_state_tensor_nbytes(optimizer: torch.optim.Optimizer) -> int:
    seen: set[int] = set()
    return _tensor_tree_nbytes(optimizer.state, seen=seen)


def galore_projector_nbytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    seen: set[int] = set()
    for state in optimizer.state.values():
        if not isinstance(state, Mapping):
            continue
        projector = state.get("projector")
        if projector is None:
            continue
        ortho = getattr(projector, "ortho_matrix", None)
        total += _tensor_tree_nbytes(ortho, seen=seen)
    return total


def format_bytes(nbytes: int) -> str:
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(nbytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"

