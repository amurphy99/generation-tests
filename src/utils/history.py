"""
Utility functions for managing conversation histories. 
--------------------------------------------------------------------------------
`src.utils.history`

"""


# History Management
def get_sliding_context(system_prompt_text: str, full_history: list, window_size: int) -> list:
    """ 
    Builds a "history" with a given system prompt + last N messages.
    """
    system_msg = {"role": "system", "content": system_prompt_text}
    recent     = full_history[-window_size:] if window_size > 0 else []
    return [system_msg] + recent


