"""
Response model dedicated to maintaining a conversation/user context graph. 
--------------------------------------------------------------------------------
`src.conversation_models.context_manager`

Dedicated response model for a slower, structured generation agent to maintain a
conversation context.

We define the Pydantic response model here as well as the overall prompt to use.

"""
from typing   import List, Literal
from pydantic import BaseModel, Field

# Conversation State
ConversationState = Literal[
    "initiate_smalltalk",
    "explore_user_interests",
    "initiate_memory_activity",
    "discuss_memory_activity_topic",
]


# ================================================================================
# Internal Robot Controller (state update + context deltas + plan for next turn)
# ================================================================================
class ContextManager(BaseModel):
    # Overall message processing
    user_message_interpretation: str = Field(..., description=(
        "1-3 grounded sentences summarizing what the user expressed in the latest message. "
        "Include self-disclosures, emotional tone, and temporal cues when present. "
        "Do not invent facts beyond the user's words."
    ))

    # --------------------------------------------------------------------------------
    # Conversation State Update
    # --------------------------------------------------------------------------------
    # Rationale first
    state_rationale: str = Field(..., description=(
        "Why the conversation state should stay the same or shift, based on the user's latest message "
        "and the current context. Prefer staying in the same state unless there is a clear reason to change."
    ))

    # Update (or don't update) the conversation state accordingly
    conversation_state : ConversationState

    # --------------------------------------------------------------------------------
    # Context Management
    # --------------------------------------------------------------------------------
    context_delta: List[str] = Field(..., description=(
        "A list of compact context update operations to apply to the persistent ConversationContext. "
        "Use 'key=value' to set scalar fields and 'key+=value' to append to list fields. "
        "Include only stable, useful facts; omit if no updates are needed."
    ))

    # --------------------------------------------------------------------------------
    # Final Plan (for the fast response model)
    # --------------------------------------------------------------------------------
    tentative_plan: str = Field(..., description=(
        "A brief suggestion to the assistant about how to advance the conversation depending on what the user's next reply could be."
    ))



