from __future__ import annotations

from .adamw import AdamW as GaLoreAdamW


def _missing_optional_dependency(name: str, dependency: str):
    class _MissingOptionalDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires the optional dependency '{dependency}'. "
                f"Install it to use this optimizer."
            )

    _MissingOptionalDependency.__name__ = name
    return _MissingOptionalDependency


try:
    from .adafactor import Adafactor as GaLoreAdafactor
except ImportError:
    GaLoreAdafactor = _missing_optional_dependency("GaLoreAdafactor", "tensorly")

try:
    from .adamw8bit import AdamW8bit as GaLoreAdamW8bit
except ImportError:
    GaLoreAdamW8bit = _missing_optional_dependency("GaLoreAdamW8bit", "bitsandbytes")
