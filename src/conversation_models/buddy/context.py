"""
Persistent conversation context for the multi-agent configuration.
--------------------------------------------------------------------------------
`src.conversation_models.buddy.context`

The "slow" thinking model generates small updates to the context that are
converted to a string representation to be given to the "fast" response model.

TODO: Need to see how it holds up in longer conversations

"""
import json
from typing   import List, Optional
from pydantic import BaseModel, Field

# From this project
from .models import ConversationState

# Style formatting (for print_diff)
from ...utils.logging.logging import RESET, BOLD, GREEN, RED, BLACK
from ...utils.logging.utils   import hr


# ================================================================================
# Long-term Context Storage (persistent across turns)
# ================================================================================
# Inner Pydantic Model (holds the delta-updatable data fields)
class _ContextData(BaseModel):
    user_name         : Optional[str] = None
    user_profile      : List[str]     = Field(default_factory=list)  # Stable facts (ex: retired librarian, likes gardening)
    key_entities      : List[str]     = Field(default_factory=list)  # People/pets/places mentioned
    open_threads      : List[str]     = Field(default_factory=list)  # Questions to return to later
    last_focus        : Optional[str] = None                         # What we're talking about now
    #memory_topic     : Optional[str] = None
    turns_in_interest : int           = 0


# ================================================================================
# ConversationContext — wrapper class with all context-related methods
# ================================================================================
class ConversationContext:

    def __init__(self):
        self.data               : _ContextData      = _ContextData()
        self.conversation_state : ConversationState = "initiate_smalltalk"
        self.plan               : str               = "Get the user's name and ask how they are doing."

    # ================================================================================
    # Apply generated deltas to the locally stored context
    # ================================================================================
    # TODO: I hate this function
    def apply_delta(self, deltas: List[str]) -> None:
        """
        Applies a list of delta strings from the slow controller to self.data.

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
            else: continue

            key = key.strip()
            val = val.strip()
            if (not key) or (not hasattr(self.data, key)): continue

            current = getattr(self.data, key)

            # If "+=" used on a scalar field, treat as "="
            if (op == "+=") and (not isinstance(current, list)): op = "="

            if op == "+=":
                if isinstance(current, list) and val and val not in current:
                    current.append(val)

            elif op == "=":
                # Don't allow scalar set on list fields
                if isinstance(current, list): continue

                if isinstance(current, int):
                    try:   setattr(self.data, key, int(val))
                    except ValueError: continue
                else: setattr(self.data, key, val if val != "" else None)


    # ================================================================================
    # Context Display
    # ================================================================================
    def to_json(self) -> str:
        """ Dump self.data to JSON for the slow model prompt. """
        return self.data.model_dump_json()

    def to_fast_string(self, n_max: int = 100) -> str:
        """
        Turn self.data into 1-3 short sentences for the fast prompt.
        Trying to be efficient with tokens and only include relevant information.
        """
        bits: List[str] = []

        # User name
        if self.data.user_name:
            bits.append(f"User name is {self.data.user_name}.")

        # Keep only the most important/stable facts
        if self.data.user_profile:
            top_profile = "; ".join(self.data.user_profile[:n_max])
            bits.append(f"About user: {top_profile}.")

        # People/pets/places etc.
        if self.data.key_entities:
            top_entities = ", ".join(self.data.key_entities[:n_max])
            bits.append(f"Key entities: {top_entities}.")

        # Last known topic of conversation
        if self.data.last_focus:
            bits.append(f"Current focus: {self.data.last_focus}.")

        # Conversation topics/questions to revisit later
        if self.data.open_threads:
            top_open_threads = ", ".join(self.data.open_threads[:n_max])
            bits.append(f"Open threads: {top_open_threads}.")

        # Topic of the current memory task (if there is one active)
        #if self.data.memory_topic:
        #    bits.append(f"Memory topic: {self.data.memory_topic}.")

        # If nothing in the context so far
        # TODO: Hard cap to keep prompt short? (would be bits[:3])
        return " ".join(bits) if bits else "Just met the user."


    # ================================================================================
    # State Gating
    # ================================================================================
    # Prevent the conversation from advancing too quickly
    def advance_state(self, proposed: ConversationState) -> None:
        """
        Gate the proposed state transition (prevent advancing too quickly),
        update turns_in_interest, and commit to self.conversation_state.
        """
        prev = self.conversation_state
        new  = proposed

        # Prevent the conversation state from advancing too quickly
        if (new == "initiate_memory_activity"     ) and (prev != "explore_user_interests"  ) and (self.data.turns_in_interest < 2): new = prev
        if (new == "discuss_memory_activity_topic") and (prev != "initiate_memory_activity") and (self.data.turns_in_interest < 2): new = prev

        # Increment/reset the turn counter per state
        same_state   = (prev == new)
        sticky_state = new in ("explore_user_interests", "initiate_memory_activity", "discuss_memory_activity_topic")

        if (same_state and sticky_state): self.data.turns_in_interest += 1  # Been in the same state for at least 2 in a row
        else:                             self.data.turns_in_interest  = 0  # Reset count

        self.conversation_state = new


    # --------------------------------------------------------------------------------
    # Context "Snapshot"
    # --------------------------------------------------------------------------------
    # String version of the context for the fast model to use
    def snapshot(self) -> 'ConversationContext':
        """ Return a deep copy of this context (cheap snapshot copy). """
        copy                    = ConversationContext.__new__(ConversationContext)
        copy.data               = _ContextData.model_validate(self.data.model_dump())
        copy.conversation_state = self.conversation_state
        copy.plan               = self.plan
        return copy

    # ================================================================================
    # Display the updated conversation context as well as what the changes were
    # ================================================================================
    def print_diff(self, old: 'ConversationContext') -> None:
        """ Print a color diff of self.data vs old.data. """
        print(f"{BOLD}--- Context Updates -----------------------------{RESET}")

        # Convert Pydantic models to dictionaries
        old_data = old.data.model_dump()
        new_data = self.data.model_dump()

        for key, new_val in new_data.items():
            old_val = old_data.get(key)

            # --------------------------------------------------------------------------------
            # Loop through each list element of the context
            # --------------------------------------------------------------------------------
            # (user_profile, key_entities, open_threads)
            if isinstance(new_val, list) and isinstance(old_val, list):
                # If no change: just print it as is
                if old_val == new_val:  print(f"  \"{key}\": {json.dumps(new_val)},")

                # If there was a change: print the diffs in red/green
                else:
                    print(f"  \"{key}\": [")

                    # Print removed items in RED
                    for item in old_val:
                        if (item not in new_val): print(f"{RED}    - {json.dumps(item)},{BLACK}")

                    # Print added items in GREEN, unchanged in BLACK
                    for item in new_val:
                        if (item not in old_val): print(f"{GREEN}    + {json.dumps(item)},{BLACK}")
                        else:                     print(f"{BLACK}      {json.dumps(item)},{BLACK}")
                    print("  ],")

            # --------------------------------------------------------------------------------
            # Handle primitive values (user_name, last_focus, turns_in_interest)
            # --------------------------------------------------------------------------------
            else:
                # Special handling specifically for turns_in_interest
                if key == "turns_in_interest":
                    if                             (new_val == old_val): print(f"  \"{key}\": {       json.dumps(new_val)              },")
                    elif (old_val is not None) and (new_val  > old_val): print(f"  \"{key}\": {GREEN}{json.dumps(new_val)}{RESET}{BLACK},")
                    elif (old_val is not None) and (new_val  < old_val): print(f"  \"{key}\": {RED  }{json.dumps(new_val)}{RESET}{BLACK},")

                # Standard handling for all other primitive values
                else:
                    # No change
                    if old_val == new_val: print(f"  \"{key}\": {json.dumps(new_val)},")

                    # Print the old value in RED and the new one in GREEN
                    else:
                        print(f"{RED  }  - \"{key}\": {json.dumps(old_val)},{BLACK}")
                        print(f"{GREEN}  + \"{key}\": {json.dumps(new_val)},{BLACK}")

        print(f"{BOLD}{hr('-', 50)}{RESET}")
