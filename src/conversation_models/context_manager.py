"""
Response model dedicated to maintaining conversation/user context. 
--------------------------------------------------------------------------------
`src.conversation_models.context_manager`

Dedicated response model for a slower, structured generation agent to maintain a
conversation context. Reads the latest user message, updates the persistent
conversation context, manages conversation state, and gives the fast reply model
a plan for the next turn.

We define the Pydantic response model here as well as the overall prompt to use.

TODO: Try cutting the prompt down to be shorter

"""
import json
from typing   import List, Literal, Optional
from pydantic import BaseModel, Field

# Style formatting
from ..utils.logging.logging import RESET, DIM
from ..utils.logging.logging import CYAN, GREEN, RED, YELLOW, BLUE, BRIGHT_BLUE, BLACK
from ..utils.logging.utils   import hr


# ================================================================================
# Conversation State
# ================================================================================
ConversationState = Literal[
    "initiate_smalltalk",    # Opener: get name, check in on how they are doing
    "learn_about_user",      # Build present_facts, understand current life
    "explore_life_story",    # User sharing past; build past_facts
    "guided_reminiscence",   # Active positive reminiscence around a specific anchor
]


# ================================================================================
# Context Delta Operation
# ================================================================================
class ContextOp(BaseModel):
    """
    A single structured update operation to apply to the ConversationContext.

    "op" semantics:
      set     => set a scalar field (user_name)
      append  => add a new item to a list; no-op if already present
      update  => replace old_value with value in a list field; if old_value not found, append value instead
    """
    op        : Literal["set", "append", "update"]
    field     : str           = Field(...,  description="Target field name in ConversationContext (e.g. 'present_facts', 'user_name').")
    value     : str           = Field(...,  description="New value to set or append.")
    old_value : Optional[str] = Field(None, description="Existing value to replace. Required for 'update'.")


# ================================================================================
# Slow Controller Response Model
# ================================================================================
class ContextManager(BaseModel):
    # --------------------------------------------------------------------------------
    # Analysis of the message before any decisions
    # --------------------------------------------------------------------------------
    # Overall message processing
    user_message_interpretation : str = Field(..., description=(
        "1-3 grounded sentences summarizing exactly what the user expressed in their latest message. "
        "Include self-disclosures, emotional tone, and temporal cues when present. "
        "Do not invent facts beyond the user's words."
    ))

    # Current topic of conversation (auto-applied to context.last_focus in the loop)
    last_topic : str = Field(..., description=(
        "A brief description of what the current main topic of the conversation is. "
    ))

    # --------------------------------------------------------------------------------
    # State Decision
    # --------------------------------------------------------------------------------
    # Reasoning about whether the state should change (before the decision)
    state_rationale : str = Field(..., description=(
        "Why the conversation state should stay the same or change, based on the user's latest message "
        "and the current context. Prefer staying in the current state unless there is a clear reason to advance."
    ))

    # The state decision
    conversation_state : ConversationState

    # --------------------------------------------------------------------------------
    # Context Updates
    # --------------------------------------------------------------------------------
    context_delta : List[ContextOp] = Field(default_factory=list, description=(
        "A list of structured context update operations. "
        "Be SPECIFIC - store 'has a garden with roses', not 'gardening'. "
        "Update existing entries when more detail emerges rather than creating duplicate entries. "
        "Leave empty if nothing new was learned."
    ))

    # --------------------------------------------------------------------------------
    # Conversation Advancement
    # --------------------------------------------------------------------------------


    # High-level plan for the response model to use for its next turn
    tentative_plan : str = Field(..., description=(
        "A brief suggestion to the fast reply model about how to advance the conversation next turn, "
        "depending on what the user's next reply could be."
    ))


# ================================================================================
# Prompt Builder
# ================================================================================
def get_context_manager_prompt(current_state: ConversationState, context_json: str) -> str:
    """
    Slow Controller system prompt.
    Keeps state and context sticky by default, uses structured ContextOp deltas.
    """
    return f"""
You are Buddy's CONTROLLER. You do NOT speak to the user directly.

Your job is to keep the conversation coherent across turns.
Prefer natural rapport first. Explore the user's current interests, routines, or background.
Guide Buddy toward discussing the user's past memories when the user provides a clear opening, 
such as a past reference, nostalgia cue, long-term hobby, life role, place, or family connection.

Go with the flow, if the user doesn't remember something, don't press them; switch to a different topic.
If the user changes the topic suddenly, you can try gently guiding them back to what you were talking about, 
but don't force it. Talk about whatever they want to talk about.

STATE POLICY:
- Default is NO CHANGE. Only advance when the user's message clearly supports it.
- "initiate_smalltalk":  Opening: get the user's name, ask how they are doing.
- "learn_about_user":    Actively learning about their present life and background.
- "explore_life_story":  User is sharing memories or past experiences; go deeper.
- "guided_reminiscence": Focused positive reminiscence around a specific anchor.

CONTEXT POLICY - BE SPECIFIC:
  BAD:  append present_facts "gardening"
  GOOD: append present_facts "has a garden"
  BEST: append present_facts "has a garden with roses in the backyard"

When the user provides more detail about something already in context, UPDATE the existing
entry rather than creating a new one.

CONTEXT FIELDS (allowed delta targets):
  user_name     (string): user's first name
  present_facts (list): things currently true: "has a garden with roses", "lives alone"
  past_facts    (list): things that used to be true: "used to work as a librarian"
  people        (list): named people with context: "grandson Tommy who calls on Sundays"

DELTA OPERATION EXAMPLES:
  {{"op": "set",    "field": "user_name",     "value": "Martha"}}
  {{"op": "append", "field": "present_facts", "value": "has a garden with roses"}}
  {{"op": "append", "field": "people",        "value": "grandson Tommy who calls on Sundays"}}
  {{"op": "update", "field": "present_facts", "old_value": "has a garden", "value": "has a garden with roses"}}

CURRENT conversation_state: {current_state}

CURRENT CONTEXT (JSON):
{context_json}

""".strip()





# ================================================================================
# Print Function
# ================================================================================
def print_context_manager(duration: float, old_state: str, old_focus: str, update: ContextManager) -> None:
    print(f"{BLUE}--- CONTEXT MANAGER ({duration:.2f}s) [Controller Update] -------------- {RESET}")

    # State transition
    new_state = update.conversation_state
    if new_state == old_state: print(f"{BRIGHT_BLUE}State:      {RESET}{new_state} {DIM}(no change){RESET}")
    else:                      print(f"{BRIGHT_BLUE}State:      {GREEN}{new_state} {DIM}({RED}{old_state}{BLACK}){RESET}")
    print(f"{BRIGHT_BLUE}Rationale:  {RESET}{update.state_rationale}")

    # Topic transition (same display pattern as state)
    new_focus = update.last_topic
    if new_focus == old_focus: print(f"{BRIGHT_BLUE}Topic:      {RESET}{new_focus} {DIM}(no change){RESET}")
    else:                      print(f"{BRIGHT_BLUE}Topic:      {GREEN}{new_focus} {DIM}({RED}{old_focus}{BLACK}){RESET}")

    # Context delta operations
    if update.context_delta:
        print(f"{BRIGHT_BLUE}Deltas:{RESET}")
        for op in update.context_delta: _print_op(op)
    else:
        print(f"{BRIGHT_BLUE}Deltas:     {RESET}{DIM}(none){RESET}")

    # Tentative response plan
    print(f"{BRIGHT_BLUE}Plan:       {RESET}{update.tentative_plan}")
    print(f"{BLUE}{hr('-')}{RESET}\n")


# Helper for printing context delta operations
def _print_op(op: ContextOp) -> None:
    """ Print a single ContextOp in color. """
    tag   = f"{op.op:<6}"
    field = op.field

    if   op.op == "set"   : print(f"  {CYAN} [{tag}]{RESET} {field}: {json.dumps(op.value)}")
    elif op.op == "append": print(f"  {GREEN}[{tag}]{RESET} {field}: {json.dumps(op.value)}")
    elif op.op == "update":
        old = json.dumps(op.old_value) if op.old_value else "?"
        print(f"  {YELLOW}[{tag}]{RESET} {field}: {RED}{old}{RESET} -> {GREEN}{json.dumps(op.value)}{RESET}")
