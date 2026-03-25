"""
Print responses from the robot
--------------------------------------------------------------------------------
`src.conversation_models.buddy.printing`

"""
# From this project
from .models import ConversationResponse, RobotSlowUpdate, RobotFastReply

# Style formatting
from ...utils.logging.logging import RESET, BOLD, DIM, ITALIC, BRIGHT_BLUE, BLACK
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
    print(f"{CYAN}{hr('-')} {RESET}\n")

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
    print(f"{RED}{BOLD}Context:{RESET} {ITALIC}{context_text}{RESET}")
