"""UI-independent reactor simulation, ready for later platform adapters."""
from .core import Config, SimulationCancelled, simulate

__all__ = ["Config", "SimulationCancelled", "simulate"]
