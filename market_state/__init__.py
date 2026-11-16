"""Market state engine of the paper (weighted evidence voting over technical-analysis layers)."""
from .engine import UNDETERMINED, EngineConfig, aggregate, candidate, persist, run_engine, switch_count
from .layers import STATES

__all__ = [
    "STATES",
    "UNDETERMINED",
    "EngineConfig",
    "aggregate",
    "candidate",
    "persist",
    "run_engine",
    "switch_count",
]
