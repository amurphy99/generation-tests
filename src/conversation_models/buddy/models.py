"""
Pydantic response models for the assistant.
--------------------------------------------------------------------------------
`src.conversation_models.buddy.models`

  V1: ConversationResponse       => single-agent (thought + state + message)
  V2: RobotFastReply             => fast track (message only)
  V2: RobotSlowUpdate            => slow controller (state + deltas + plan)

TODO: Seems like we might not need to worry about latency as much anymore...

"""
from typing   import List, Literal
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------
# Conversation State
# --------------------------------------------------------------------------------
ConversationState = Literal[
    "initiate_smalltalk",
    "explore_user_interests",
    "initiate_memory_activity",
    "discuss_memory_activity_topic",
]

# ================================================================================
# Single-agent robot response
# ================================================================================
class ConversationResponse(BaseModel):
    # Analyze the user's message
    thought: str = Field(..., description="Brief internal reasoning about how the conversation is going and how to continue.")

    # Do this before the message, so it decides the state and then drafts the response to say with that in mind
    conversation_state : ConversationState  

    # Draft response
    message: str = Field(..., description="Your spoken response to the user.")


# ================================================================================
# Dual Agent Response Models
# ================================================================================
# Fast robot replies (spoken reply only, lower latency)
class RobotFastReply(BaseModel):
    message: str = Field(..., description="Your spoken response to the user.")

# Slow, "thinking" robot controller (state update + context deltas + plan for next turn)
class RobotSlowUpdate(BaseModel):
    state_reason       : str       = Field(..., description="Justification for state change, citing user message.")
    conversation_state : ConversationState
    evidence           : str       = Field(..., description="Compressed fragments extracted from the USER message only. Use short phrase chunks separated by ...")
    context_delta      : List[str] = Field(..., description=(
        "A list of compact context update operations to apply to the persistent ConversationContext. "
        "Use 'key=value' to set scalar fields and 'key+=value' to append to list fields. "
        "Include only stable, useful facts; omit if no updates are needed."
    ))
    tentative_plan     : str       = Field(..., description=(
        "A brief suggestion to the assistant about how to advance the conversation depending on what the user's next reply could be."
    ))
