"""gemmaflash: fast, low-memory Gemma 4 26B-A4B decoding on Apple Silicon."""

from .budget import describe as describe_budget
from .budget import fit as fit_budget
from .runtime import Runtime

__all__ = ["Runtime", "fit_budget", "describe_budget"]
__version__ = "0.1.0"
