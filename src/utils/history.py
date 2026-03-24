"""
Utility functions for managing conversation histories. 
--------------------------------------------------------------------------------
`src.utils.history`

Whoever is the "user" or "assistant" changes based on the perspective. 

TODO: This should probably be in a different folder...

"""
from ..conversation_models.simulated_user import UserConversationResponse
from ..conversation_models.buddy.models   import ConversationResponse


# History Management
def get_sliding_context(system_prompt_text: str, full_history: list, window_size: int) -> list:
    """ 
    Builds a "history" with a given system prompt + last N messages.
    """
    system_msg = {"role": "system", "content": system_prompt_text}
    recent     = full_history[-window_size:] if window_size > 0 else []
    return [system_msg] + recent


# --------------------------------------------------------------------------------
# [v1] History Helpers
# --------------------------------------------------------------------------------
def sync_history_robot(history_robot: list, history_user: list, response: ConversationResponse):
    """Robot spoke: robot saves its full JSON; user history gets plain text."""
    history_robot.append({"role": "assistant", "content": response.model_dump_json()    })
    history_user .append({"role": "user",      "content": f"[Buddy]: {response.message}"})


def sync_history_user(history_robot: list, history_user: list, response: UserConversationResponse):
    """User spoke: both histories get plain text (robot doesn't see user's thought)."""
    history_robot.append({"role": "user",      "content": response.message          })
    history_user .append({"role": "assistant", "content": response.model_dump_json()})


# --------------------------------------------------------------------------------
# [v2] History Utilities
# --------------------------------------------------------------------------------
# User simulator "hears" Buddy as the 'user'
def append_buddy_to_user_history(history_user: list, buddy_msg: str):
    history_user.append({"role": "user", "content": f"[Buddy]: {buddy_msg}"})

# User simulator "remembers" what it (Martha) said as the 'assistant'
def append_martha_to_user_history(history_user: list, martha_msg: str):
    history_user.append({"role": "assistant", "content": martha_msg})


