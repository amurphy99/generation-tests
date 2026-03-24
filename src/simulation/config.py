"""
Runtime configuration for simulations.
--------------------------------------------------------------------------------
`src.simulation.config`

Pass a SimulationConfig to run_simulation() in v1_loop or v2_loop.

All agent-level architecture params (temperature, window, max_tokens) are
hardcoded inside each loop file.

TODO: Might have to update this to build things in a smarter way...

"""
import os
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    # LLM connection (defaults to env vars)
    LLM_URL: str = field(default_factory=lambda: os.getenv("LLM_URL", "http://localhost:8000/v1"))
    LLM_KEY: str = field(default_factory=lambda: os.getenv("LLM_KEY", "TOKEN"                   ))

    # Model to use for all agents in the simulation
    model  : str   = "qwen2.5-3b"

    # Run parameters
    turns   : int   = 10
    sleep_s : float = 1.0

    # Verbosity (v2 only; ignored by v1_loop)
    verbose_context : bool = False  # Print full context JSON each turn
    verbose_slow    : bool = True   # Print slow controller update each turn

