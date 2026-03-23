"""
Persistent conversation context for the multi-agent configuration.
--------------------------------------------------------------------------------
`src.conversation_models.buddy.context`

The "slow" thinking model generates small updates to the context that are
converted to a string representation to be given to the "fast" response model.

TODO: Need to see how it holds up in longer conversations

"""
from typing   import List, Optional
from pydantic import BaseModel, Field

# From this project
from .models import ConversationState


# ================================================================================
# Long-term Context Storage (persistent across turns)
# ================================================================================
# Pydantic Conversation Context
class ConversationContext(BaseModel):
    user_name         : Optional[str] = None
    user_profile      : List[str]     = Field(default_factory=list)  # Stable facts (ex: retired librarian, likes gardening)
    key_entities      : List[str]     = Field(default_factory=list)  # People/pets/places mentioned
    open_threads      : List[str]     = Field(default_factory=list)  # Questions to return to later
    last_focus        : Optional[str] = None                         # What we're talking about now
    #memory_topic     : Optional[str] = None
    turns_in_interest : int           = 0

# Default init
def init_context_store() -> ConversationContext:
    return ConversationContext()


# ================================================================================
# Apply generated deltas to the locally stored ConversationContext
# ================================================================================
# Apply "key=val" or "key+=val" delta strings from the slow controller
# TODO: I hate this function 
def apply_context_delta(ctx: ConversationContext, deltas: List[str]) -> ConversationContext:
    """
    Applies a list of delta strings from the slow controller to the context.

    Accepts strict delta formatting:
      "user_name=Martha"
      "user_profile+=likes gardening"
    """
    for raw in deltas:
        s = (raw or "").strip()
        if not s: continue

        # Normalize common prefixes like "key+=" or "key="
        lowered = s.lower()
        if   lowered.startswith("key+="): s = s[5:].lstrip()
        elif lowered.startswith("key=" ): s = s[4:].lstrip()

        # Normalize ":" to "=" if the model used a colon instead of equals
        if (":" in s) and ("=" not in s) and ("+=" not in s): s = s.replace(":", "=", 1)

        # Parse operator
        op = None
        if   "+=" in s: key, val = s.split("+=", 1); op = "+="
        elif  "=" in s: key, val = s.split( "=", 1); op =  "="
        else:  continue

        key = key.strip()
        val = val.strip()
        if (not key) or (not hasattr(ctx, key)): continue

        current = getattr(ctx, key)

        # If "+=" used on a scalar field, treat as "="
        if (op == "+=") and (not isinstance(current, list)): op = "="

        if op == "+=":
            if isinstance(current, list) and val and val not in current:
                current.append(val)

        elif op == "=":
            # Don't allow scalar set on list fields
            if isinstance(current, list): continue

            if isinstance(current, int):
                try:   setattr(ctx, key, int(val))
                except ValueError: continue
            else: setattr(ctx, key, val if val != "" else None)

    return ctx

# --------------------------------------------------------------------------------
# Context Display
# --------------------------------------------------------------------------------
# Dump the context 
def context_to_json(ctx: ConversationContext) -> str:
    return ctx.model_dump_json()

# Convert the JSON context to something for the fast response model to use
def render_context_for_fast(ctx: ConversationContext, n_max: int = 100) -> str:
    """
    Turn persistent context into 1-3 short sentences for the fast prompt.
    Trying to be efficient with tokens and only include relevant information.
    """
    bits: List[str] = []

    # User name
    if ctx.user_name: 
        bits.append(f"User name is {ctx.user_name}.")

    # Keep only the most important/stable facts 
    if ctx.user_profile: 
        top_profile = "; ".join(ctx.user_profile[:n_max])
        bits.append(f"About user: {top_profile}.")

    # People/pets/places etc. 
    if ctx.key_entities:   
        top_entities = ", ".join(ctx.key_entities[:n_max])
        bits.append(f"Key entities: {top_entities}.")

    # Last known topic of conversation
    if ctx.last_focus:
        bits.append(f"Current focus: {ctx.last_focus}.")

    # Conversation topics/questions to revisit later
    if ctx.open_threads:
        top_open_threads = ", ".join(ctx.open_threads[:n_max])
        bits.append(f"Open threads: {top_open_threads}.")

    # Topic of the current memory task (if there is one active)
    #if ctx.memory_topic:
    #    bits.append(f"Memory topic: {ctx.memory_topic}.")

    # If nothing in the context so far
    # TODO: Hard cap to keep prompt short? (would be bits[:3])
    return " ".join(bits) if bits else "Just met the user." 


# ================================================================================
# State Gating
# ================================================================================
# Prevent the conversation state from advancing too quickly
def gate_state(prev_state: ConversationState, proposed_state: ConversationState, ctx: ConversationContext) -> ConversationState:
    """ Only allow initiate_memory_activity after >=2 turns in interest. """
    if (proposed_state == "initiate_memory_activity"     ) and (prev_state != "explore_user_interests"  ) and (ctx.turns_in_interest < 2): return prev_state
    if (proposed_state == "discuss_memory_activity_topic") and (prev_state != "initiate_memory_activity") and (ctx.turns_in_interest < 2): return prev_state
    return proposed_state

# Increment/reset the turn counter per state
def update_turns_in_interest(prev_state: ConversationState, new_state: ConversationState, ctx: ConversationContext) -> None:
    """ Increments turns_in_interest when staying in the same non-smalltalk state, resets it otherwise. """
    same_state   = prev_state == new_state
    sticky_state = new_state in ("explore_user_interests", "initiate_memory_activity", "discuss_memory_activity_topic")

    if (same_state and sticky_state): ctx.turns_in_interest += 1 # Been in the same state for at least 2 in a row
    else:                             ctx.turns_in_interest  = 0 # Reset count

