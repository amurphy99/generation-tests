"""
Print responses from the robot
--------------------------------------------------------------------------------
`src.conversation_models.buddy.printing`

"""
# From this project
from .models  import ConversationResponse, RobotSlowUpdate, RobotFastReply
from .context import ConversationContext

# Style formatting
from ...utils.logging.logging import RESET, DIM, WHITE, BRIGHT_WHITE
from ...utils.logging.logging import CYAN, GREEN, RED, BLUE, LIME
from ...utils.logging.utils   import hr


# --------------------------------------------------------------------------------
# Single-agent robot response
# --------------------------------------------------------------------------------
def print_robot_turn(duration: float, response: ConversationResponse):
    print(f"{CYAN} --- ROBOT RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{GREEN} Thought:    {RESET} {response.thought           }")
    print(f"{GREEN} State:      {RESET} {response.conversation_state}")
    print(f"{GREEN} Message:    {RESET} {response.message           }")
    print(f"{CYAN} {hr('-')} {RESET}\n")


# --------------------------------------------------------------------------------
# Dual Agent Response Models
# --------------------------------------------------------------------------------
# ROBOT FAST
def print_robot_fast(duration: float, response: RobotFastReply):
    print(f"{CYAN} --- ROBOT FAST TRACK ({duration:.2f}s) --------------------------------- {RESET}")
    print(f"{GREEN} Message:    {RESET}{response.message}")
    print(f"{CYAN} {hr('-')} {RESET}\n")

# ROBOT SLOW
def print_robot_slow(
    duration: float, prev_state: str, slow_update: RobotSlowUpdate, 
    ctx_before: ConversationContext, ctx_after: ConversationContext,
):
    print(f"{BLUE} --- ROBOT SLOW TRACK ({duration:.2f}s) [Controller Update] ------------- {RESET}")
    print(f"{BRIGHT_WHITE} Prev State: {RESET}{prev_state}")
    print(f"{BRIGHT_WHITE} New State:  {RESET}{slow_update.conversation_state}")
    print(f"{BRIGHT_WHITE} Reason:     {RESET}{slow_update.state_reason}")
    print(f"{BRIGHT_WHITE} Evidence:   {RESET}{slow_update.evidence}")
    print(f"{BRIGHT_WHITE} Deltas:     {RESET}{slow_update.context_delta}")
    print(f"{BRIGHT_WHITE} Plan:       {RESET}{slow_update.tentative_plan}")

    # Highlight if context changed
    if ctx_before.model_dump() != ctx_after.model_dump(): print(f"{LIME} Context:    {RESET}updated")
    else:                                                 print(f"{DIM}{LIME} Context:    {RESET}{DIM}no change{RESET}")

    print(f"{BLUE} {hr('-')} {RESET}\n")


# --------------------------------------------------------------------------------
# Conversation Context
# --------------------------------------------------------------------------------
# String context (formatted for the FAST reply model)
def print_fast_context(context_text: str):
    print(f"{RED} Context: {WHITE} {context_text}{RESET}")

# JSON Context
def print_context_store(ctx: ConversationContext):
    print(f"{BLUE} --- New Context ---------------------------------")
    print(f"{WHITE}{ctx.model_dump_json(indent=2)}{RESET}")
    print(f"{BLUE} {hr('-')} {RESET}\n")

