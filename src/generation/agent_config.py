"""
Standardized configuration for agents.
--------------------------------------------------------------------------------
`src.generation.agent_config`

Holds all of the configuration parameters for the agents.

"""
from dataclasses import dataclass
from typing      import Optional, Type
from pydantic    import BaseModel


# --------------------------------------------------------------------------------
# Agent Config (dataclass)
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    name           : str
    model          : str
    temperature    : float
    max_tokens     : int
    window         : int
    response_model : Optional[Type[BaseModel]] = None  # None => raw text

