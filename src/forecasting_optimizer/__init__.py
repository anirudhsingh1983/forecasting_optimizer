# Copyright 2023 Anirudh Singh
# SPDX-License-Identifier: BSD-3-Clause

"""Public interface for the forecasting optimizer framework."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecasting_optimizer.optimization.optimizer import Optimizer

__all__ = ["Optimizer"]


def __getattr__(name: str):
    """Load the public optimizer lazily to keep package imports lightweight."""
    if name == "Optimizer":
        from forecasting_optimizer.optimization.optimizer import Optimizer

        return Optimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
