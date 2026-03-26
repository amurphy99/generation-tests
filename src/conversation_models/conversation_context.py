"""
Persistent conversation context for the V3 multi-agent configuration.
--------------------------------------------------------------------------------
`src.conversation_models.conversation_context`

The ContextManager generates structured ContextOp operations each turn. These 
are applied here to build up a full context for what we know based on the
conversation with the user so far.

"""
import json
from typing   import List, Optional
from pydantic import BaseModel, Field

# From this project
from .context_manager import ContextOp, ConversationState

# Style formatting (for print_diff)
from ..utils.logging.logging import RESET, BOLD, GREEN, RED, BLACK
from ..utils.logging.utils   import hr


# ================================================================================
# Long-Term Context Storage (persistent across turns)
# ================================================================================
class ContextData(BaseModel):
    # 1. Once the user introduced themselves
    user_name: Optional[str] = None

    # 2. Things currently true about the user's life
    # ("has a garden with roses", "lives alone in an apartment")
    present_facts: List[str] = Field(default_factory=list) 

    # 3. Things that used to be true -- memories, past roles, past possessions
    # ("used to work as a librarian", "had a husband named Frank")
    past_facts: List[str] = Field(default_factory=list)

    # 4. Named people in the user's life with relationship context
    # ("grandson Tommy who calls on Sundays", "daughter Susan in Chicago")
    people: List[str] = Field(default_factory=list)

    # 5. Current topic of conversation
    last_focus: Optional[str] = "Introduction"

    # 6. Turn counter for state gating
    turns_in_interest: int = 0


# ================================================================================
# Wrapper class for all context-related methods
# ================================================================================
class ConversationContext:
    # Initialize with values for starting off the conversation
    def __init__(self):
        self.data               : ContextData      = ContextData()
        self.conversation_state : ConversationState = "initiate_smalltalk"
        self.plan               : str               = "Get the user's name and ask how they are doing."

    # --------------------------------------------------------------------------------
    # Apply structured ContextOp deltas from the ContextManager
    # --------------------------------------------------------------------------------
    def apply_delta(self, ops: List[ContextOp]) -> None:
        """
        Apply a list of ContextOp operations to self.data.

        Op semantics:
          set     =>  set a scalar field (user_name)
          append  =>  add to a list field; no-op if already present
          update  =>  replace old_value with value in a list; if old_value not found, append value instead
        
        """
        for op in ops:
            # 1) Guard for the op being valid for a field
            field = op.field
            if not hasattr(self.data, field): continue
            current = getattr(self.data, field)

            # 2) Set an item to a non-list field
            if op.op == "set":
                if not isinstance(current, list):
                    setattr(self.data, field, op.value if op.value else None)
            
            # 3) Append a new item to a list field
            elif op.op == "append":
                if isinstance(current, list) and (op.value) and (op.value not in current):
                    current.append(op.value)

            # 4) Update an existing list field item (e.g., "has garden" -> "has garden with roses")
            elif op.op == "update":
                if isinstance(current, list) and op.value:
                    # Look for an existing field item with this value
                    if op.old_value and op.old_value in current:
                        idx = current.index(op.old_value)
                        current[idx] = op.value
                    
                    # Just create a new entry (couldn't find something to replace)
                    elif op.value not in current:
                        current.append(op.value)

    # ================================================================================
    # Context Display
    # ================================================================================
    def to_json(self) -> str:
        """ Dump self.data to JSON for the slow controller prompt. """
        return self.data.model_dump_json()

    def to_fast_string(self, n_max: int = 100) -> str:
        """
        Turn self.data into 1-3 short sentences for the fast prompt.
        Trying to be efficient with tokens and only include relevant information.
        """
        bits: List[str] = []

        # User name
        if self.data.user_name:
            bits.append(f"User: {self.data.user_name}.")

        # Facts about their current life
        if self.data.present_facts:
            facts = "; ".join(self.data.present_facts[:n_max])
            bits.append(f"Current life: {facts}.")

        # Memories, past information about them
        if self.data.past_facts:
            past = "; ".join(self.data.past_facts[:n_max])
            bits.append(f"Memories/past: {past}.")
        
        # People they have spoken of
        if self.data.people:
            people = ", ".join(self.data.people[:n_max])
            bits.append(f"People: {people}.")

        # Last known topic of conversation
        if self.data.last_focus:
            bits.append(f"Currently talking about: {self.data.last_focus}.")

        return " ".join(bits) if bits else "Just met the user."


    # ================================================================================
    # State Gating
    # ================================================================================
    def advance_state(self, proposed: ConversationState) -> None:
        """
        Gate the proposed state transition (prevent advancing too quickly),
        update turns_in_interest, and commit to self.conversation_state.
        """
        prev = self.conversation_state
        new  = proposed

        # Prevent jumping states too quickly
        turns = self.data.turns_in_interest
        if (new == "explore_life_story" ) and (prev != "learn_about_user" ) and (turns < 2): new = prev
        if (new == "guided_reminiscence") and (prev != "explore_life_story") and (turns < 2): new = prev

        # Update turn counter per sticky state
        same_state   = (prev == new)
        sticky_state = new in ("learn_about_user", "explore_life_story", "guided_reminiscence")

        if (same_state and sticky_state): self.data.turns_in_interest += 1
        else:                             self.data.turns_in_interest  = 0

        self.conversation_state = new

    # ================================================================================
    # Snapshot (for print_diff comparison)
    # ================================================================================
    def snapshot(self) -> 'ConversationContext':
        """ Return a deep copy of this context. """
        copy                    = ConversationContext.__new__(ConversationContext)
        copy.data               = ContextData.model_validate(self.data.model_dump())
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
        old_data = old .data.model_dump()
        new_data = self.data.model_dump()

        for key, new_val in new_data.items():
            old_val = old_data.get(key)

            # --------------------------------------------------------------------------------
            # List fields (present_facts, past_facts, people)
            # --------------------------------------------------------------------------------
            if isinstance(new_val, list) and isinstance(old_val, list):
                
                # Print in all black as normal
                if old_val == new_val: 
                    if len(new_val) == 0: print(f"  \"{key}\": []")
                    else:
                        print(f"  \"{key}\": [")
                        for item in new_val: 
                            print(f"{BLACK}      {json.dumps(item)},{BLACK}")
                        print("  ],")
                
                # Highlight with different colors to show changes
                else:
                    print(f"  \"{key}\": [")
                    for item in old_val:
                        if item not in new_val: print(f"{RED  }    - {json.dumps(item)},{BLACK}")
                    for item in new_val:
                        if item not in old_val: print(f"{GREEN}    + {json.dumps(item)},{BLACK}")
                        else:                   print(f"{BLACK}      {json.dumps(item)},{BLACK}")
                    print("  ],")

            # --------------------------------------------------------------------------------
            # Scalar fields (user_name, last_focus, turns_in_interest)
            # --------------------------------------------------------------------------------
            else:
                if key == "turns_in_interest":
                    if                             (new_val == old_val): print(f"  \"{key}\": {       json.dumps(new_val)              },")
                    elif (old_val is not None) and (new_val  > old_val): print(f"  \"{key}\": {GREEN}{json.dumps(new_val)}{RESET}{BLACK},")
                    elif (old_val is not None) and (new_val  < old_val): print(f"  \"{key}\": {RED  }{json.dumps(new_val)}{RESET}{BLACK},")
                else:
                    if old_val == new_val:
                        print(f"  \"{key}\": {json.dumps(new_val)},")
                    else:
                        print(f"{RED  }  - \"{key}\": {json.dumps(old_val)},{BLACK}")
                        print(f"{GREEN}  + \"{key}\": {json.dumps(new_val)},{BLACK}")

        print(f"{BOLD}{hr('-', 50)}{RESET}")

