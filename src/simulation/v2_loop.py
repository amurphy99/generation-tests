"""
V2 simulation loop -- dual-track robot (fast reply + slow controller).
--------------------------------------------------------------------------------
`src.simulation.v2_loop`

Setup:
  USER_SIM    => UserConversationResponse  (message only)
  ROBOT_FAST  => RobotFastReply            (spoken reply; uses OLD state/context for low latency)
  ROBOT_SLOW  => RobotSlowUpdate           (state + context deltas + plan; prepares NEXT turn)

The slow controller runs after the fast reply is already spoken. Its output
(updated state, context deltas, tentative plan) is used in the *next* turn's
fast prompt, not the current one.

Usage:
    from src.simulation.config  import SimulationConfig
    from src.simulation.v2_loop import run_simulation

    config = SimulationConfig(model="gpt-4o-mini", turns=10, verbose_context=True)
    run_simulation(config)

"""
import time, instructor
from openai import OpenAI

# From this project
from ..config                import TIMEOUT
from ..utils.logging.logging import RESET, BOLD, UNBOLD, GREEN, CYAN, MAGENTA, BLUE
from ..utils.logging.utils   import print_banner, print_turn_header

# Version-Specific Setup
from ..conversation_models.simulated_user import UserConversationResponse, USER_SYSTEM_PROMPT, print_user_turn
from ..conversation_models.buddy.context  import ConversationContext
from ..conversation_models.buddy.models   import RobotFastReply, RobotSlowUpdate
from ..conversation_models.buddy.prompts  import get_robot_fast_prompt, get_robot_slow_prompt
from ..conversation_models.buddy          import printing as pr

# Generation Utilities
from  .config                  import SimulationConfig
from ..generation.agent_config import AgentConfig
from ..generation.multi_agent  import run_agent
from ..utils.history           import append_buddy_to_user_history, append_martha_to_user_history


# ================================================================================
# Client & Agent Construction
# ================================================================================
def _make_agents(config: SimulationConfig):
    robot_fast = AgentConfig(
        name           = "robot_fast",
        model          = config.model,
        temperature    = 0.7,
        max_tokens     = 128,
        window         =   2,
        response_model = RobotFastReply,
    )
    robot_slow = AgentConfig(
        name           = "robot_slow",
        model          = config.model,
        temperature    = 0.3,
        max_tokens     = 256,
        window         =   2,
        response_model = RobotSlowUpdate,
    )
    user_sim = AgentConfig(
        name           = "user_sim",
        model          = config.model,
        temperature    = 0.7,
        max_tokens     = 256,
        window         =   6,
        response_model = UserConversationResponse,
    )
    return robot_fast, robot_slow, user_sim


# ================================================================================
# Conversation Loop
# ================================================================================
def run_simulation(config: SimulationConfig):
    """
    Run a dual-agent conversation simulation.
      ROBOT_FAST => structured message-only (fast)
      ROBOT_SLOW => structured controller update w/ deltas (sticky context/state)
    """
    print_banner("Starting Simulation: Split Model Architecture")

    # --------------------------------------------------------------------------------
    # 1) Initialize Clients & Histories
    # --------------------------------------------------------------------------------
    # Client in instructor mode
    client = instructor.from_openai(
        OpenAI(base_url=config.LLM_URL, api_key=config.LLM_KEY, timeout=TIMEOUT),
        mode=instructor.Mode.JSON
    )

    # Initialize agents
    robot_fast, robot_slow, user_sim = _make_agents(config)

    # Separate histories (user doesn't know about robots thought processes & vice versa)
    history_robot : list = []  # Buddy view: user/assistant turns (spoken only)
    history_user  : list = []  # Martha simulator view (Buddy as "user", Martha as "assistant")

    # --------------------------------------------------------------------------------
    # 2) Begin the Conversation (robot goes first)
    # --------------------------------------------------------------------------------
    # Initial State for the Robot
    start_message = "Good morning! My name is Buddy, it is nice to meet you. What is your name?"
    context       = ConversationContext()

    # First message
    print(f"{CYAN}BUDDY (Start):{RESET} {start_message}\n")

    # Update histories
    history_robot.append({"role": "assistant", "content": start_message})
    append_buddy_to_user_history(history_user, start_message)

    # ================================================================================
    # 3) Main Loop
    # ================================================================================
    for t in range(1, config.turns + 1):
        print_turn_header(t); print()

        # --------------------------------------------------------------------------------
        # a) USER Speaks (Grandma)
        # --------------------------------------------------------------------------------
        #print(f"{MAGENTA}[USER] Thinking...{RESET}")
        t0 = time.time()

        # API call for response
        user_response: UserConversationResponse = run_agent(
            client        = client,
            agent         = user_sim,
            system_prompt = USER_SYSTEM_PROMPT,
            history       = history_user,
        )

        t1 = time.time()
        user_time = t1 - t0
        print_user_turn(t1 - t0, user_response)

        # Update histories
        martha_msg = user_response.message.strip()
        history_robot.append({"role": "user", "content": martha_msg})
        append_martha_to_user_history(history_user, martha_msg)

        # --------------------------------------------------------------------------------
        # b) Robot FAST reply (robot's spoken reply)
        # --------------------------------------------------------------------------------
        # We inject the OLD state/context into the prompt to generate the reply quickly
        fast_context_text = context.to_fast_string()
        fast_sys_prompt   = get_robot_fast_prompt(state=context.conversation_state, context_text=fast_context_text, plan=context.plan)

        # Print context
        #print(f"{CYAN}[ROBOT] Track 1: Generating reply...{RESET}")
        pr.print_fast_context(fast_context_text)

        # API call for response
        t0 = time.time()
        robot_response: RobotFastReply = run_agent(
            client        = client,
            agent         = robot_fast,
            system_prompt = fast_sys_prompt,
            history       = history_robot,
        )

        # Timing
        t1 = time.time()
        r1_time = t1 - t0

        # Print message
        buddy_msg = robot_response.message.strip()
        pr.print_robot_fast(t1 - t0, robot_response)

        # Update histories
        history_robot.append({"role": "assistant", "content": buddy_msg})
        append_buddy_to_user_history(history_user, buddy_msg)

        # --------------------------------------------------------------------------------
        # c) Robot SLOW controller update (state + deltas + plan)
        # --------------------------------------------------------------------------------
        # Now we think about what just happened to prepare for the NEXT turn
        #print(f"{BLUE} [ROBOT] Track 2: Updating state/context...{RESET}")

        # Handling context
        ctx_before      = context.snapshot()  # cheap snapshot copy
        slow_sys_prompt = get_robot_slow_prompt(current_state=context.conversation_state, context_json=context.to_json())

        # API call for response
        t0 = time.time()
        slow_update: RobotSlowUpdate = run_agent(
            client        = client,
            agent         = robot_slow,
            system_prompt = slow_sys_prompt,
            history       = history_robot,
        )

        # Timing
        t1 = time.time()
        r2_time = t1 = t0

        # Apply deltas first
        context.apply_delta(slow_update.context_delta)

        # Advance state (gate + counter update)
        prev_state = context.conversation_state
        context.advance_state(slow_update.conversation_state)

        # Commit plan for next loop
        context.plan = slow_update.tentative_plan.strip()

        # --------------------------------------------------------------------------------
        # d) End-of-turn steps
        # --------------------------------------------------------------------------------
        if config.verbose_slow: pr.print_robot_slow(t1 - t0, prev_state, slow_update)

        # Print just the replies
        print(
            f"{BOLD}{GREEN  }User: {UNBOLD} { user_response.message}{RESET}\n"
            f"{BOLD}{MAGENTA}Buddy:{UNBOLD} {robot_response.message}{RESET}\n"
        )

        # Print the updates this caused to the context
        if config.verbose_context: context.print_diff(ctx_before)
        print()

        # Wait a little bit between turns
        if config.sleep_s > 0: time.sleep(config.sleep_s)
