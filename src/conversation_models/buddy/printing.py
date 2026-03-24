"""
Print responses from the robot
--------------------------------------------------------------------------------
`src.conversation_models.buddy.printing`

"""
import json

# From this project
from .models  import ConversationResponse, RobotSlowUpdate, RobotFastReply
from .context import ConversationContext

# Style formatting
from ...utils.logging.logging import RESET, BOLD, DIM, WHITE, BRIGHT_BLUE, BLACK
from ...utils.logging.logging import CYAN, GREEN, RED, BLUE
from ...utils.logging.utils   import hr


# --------------------------------------------------------------------------------
# Single-agent robot response
# --------------------------------------------------------------------------------
def print_robot_turn(duration: float, response: ConversationResponse):
    print(f"{CYAN}--- ROBOT RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{GREEN}Thought:    {RESET} {response.thought           }")
    print(f"{GREEN}State:      {RESET} {response.conversation_state}")
    print(f"{GREEN}Message:    {RESET} {response.message           }")
    print(f"{CYAN}{hr('-')} {RESET}\n")


# ================================================================================
# Dual Agent Response Models
# ================================================================================
# ROBOT FAST
def print_robot_fast(duration: float, response: RobotFastReply):
    print(f"{CYAN}--- ROBOT FAST TRACK ({duration:.2f}s) --------------------------------- {RESET}")
    print(f"{GREEN}Message:    {RESET}{response.message}")
    print(f"{CYAN} {hr('-')} {RESET}\n")

# --------------------------------------------------------------------------------
# ROBOT SLOW
# --------------------------------------------------------------------------------
def print_robot_slow(duration: float, old_state: str, slow_update: RobotSlowUpdate):
    print(f"{BLUE}--- ROBOT SLOW TRACK ({duration:.2f}s) [Controller Update] ------------- {RESET}")

    # State change
    new_state = slow_update.conversation_state
    change    = (new_state == old_state)
    if change: print(f"{BRIGHT_BLUE}Conv State:{RESET} {new_state} {RESET}{DIM}(no change){RESET}")
    else:      print(f"{BRIGHT_BLUE}Conv State:{GREEN} {new_state} {RESET}{DIM}({RED}{old_state}{BLACK}){RESET}")

    # Standard outputs
    print(f"{BRIGHT_BLUE}Reason:     {RESET}{slow_update.state_reason      }")
    print(f"{BRIGHT_BLUE}Evidence:   {RESET}{slow_update.evidence          }")
    print(f"{BRIGHT_BLUE}Deltas:     {RESET}{slow_update.context_delta     }")
    print(f"{BRIGHT_BLUE}Plan:       {RESET}{slow_update.tentative_plan    }")
    print(f"{BLUE}{hr('-')}{RESET}\n")


# String context (formatted for the FAST reply model)
def print_fast_context(context_text: str):
    print(f"{RED}Context: {WHITE} {context_text}{RESET}")

# ================================================================================
# JSON Conversation Context
# ================================================================================
def print_context_diff(old_ctx: ConversationContext, new_ctx: ConversationContext):
    print(f"{BOLD}--- Context Updates -----------------------------{RESET}")
    
    # Convert Pydantic models to dictionaries
    old_data = old_ctx.model_dump()
    new_data = new_ctx.model_dump()
    
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

