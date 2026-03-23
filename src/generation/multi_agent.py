"""
Multi-agent based response generation for the assistant.
--------------------------------------------------------------------------------
`src.generation.multi_agent`

"""
from openai import OpenAI

# From this project
from ..utils.history import get_sliding_context
from  .agent_config  import AgentConfig


# ================================================================================
# Get responses from the multi-agent assistant setup
# ================================================================================
def run_agent(
    client_raw        : OpenAI,
    client_structured,
    agent             : AgentConfig,
    system_prompt     : str,
    history           : list,
):
    """
    Runs the OpenAI web API call for all agent configurations.
    - If agent.response_model is None => raw text
    - Else => instructor/pydantic enforced structured output
    """
    # Prepare a history to provide the model with
    messages = get_sliding_context(system_prompt, history, agent.window)

    # For regular, non-structure generation
    if agent.response_model is None:
        completion = client_raw.chat.completions.create(
            model       = agent.model,
            messages    = messages,
            temperature = agent.temperature,
            max_tokens  = agent.max_tokens,
        )
        return completion.choices[0].message.content.strip()

    # Structure generation query
    return client_structured.chat.completions.create(
        model          = agent.model,
        messages       = messages,
        response_model = agent.response_model,
        temperature    = agent.temperature,
        max_tokens     = agent.max_tokens,
    )

