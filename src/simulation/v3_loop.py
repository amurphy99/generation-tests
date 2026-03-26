"""
V3 simulation loop -- dual-track robot with better context management.
--------------------------------------------------------------------------------
`src.simulation.v3_loop`

Usage:
    from src.simulation.config  import SimulationConfig
    from src.simulation.v3_loop import run_simulation

    config = SimulationConfig(model="gpt-4o-mini", turns=10, verbose_context=True)
    run_simulation(config)

"""
import time, instructor
from openai import OpenAI

# From this project
from ..config                import TIMEOUT
from ..utils.logging.logging import RESET, BOLD, UNBOLD, GREEN, CYAN, MAGENTA
from ..utils.logging.utils   import print_banner, print_turn_header

# Version-Specific Setup
from ..conversation_models.simulated_user       import UserConversationResponse, USER_SYSTEM_PROMPT, print_user_turn
from ..conversation_models.fast_reply           import FastReply, get_fast_reply_prompt, print_fast_reply
from ..conversation_models.context_manager      import ContextManager, get_context_manager_prompt, print_context_manager
from ..conversation_models.conversation_context import ConversationContext

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
        response_model = FastReply,
    )
    robot_slow = AgentConfig(
        name           = "robot_slow",
        model          = config.model,
        temperature    = 0.3,
        max_tokens     = 512,
        window         =   4,
        response_model = ContextManager,
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
      ROBOT_FAST   => structured message-only (fast)
      ROBOT_SLOW   => structured controller update w/ ContextOp deltas + callback_note
    """
    print_banner("Starting Simulation: V3 Split Model Architecture")

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

    # Separate histories (user doesn't know about robot's thought processes & vice versa)
    history_robot : list = []  # Buddy view: user/assistant turns (spoken only)
    history_user  : list = []  # Martha simulator view (Buddy as "user", Martha as "assistant")

    # --------------------------------------------------------------------------------
    # 2) Begin the Conversation (robot goes first)
    # --------------------------------------------------------------------------------
    # Initial state for the robot
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
        t0 = time.time()

        # API call for response
        user_response: UserConversationResponse = run_agent(
            client        = client,
            agent         = user_sim,
            system_prompt = USER_SYSTEM_PROMPT,
            history       = history_user,
        )

        t1 = time.time()
        print_user_turn(t1 - t0, user_response)

        # Update histories
        martha_msg = user_response.message.strip()
        history_robot.append({"role": "user", "content": martha_msg})
        append_martha_to_user_history(history_user, martha_msg)

        # --------------------------------------------------------------------------------
        # b) Robot FAST reply (uses slow controller output from PREVIOUS turn)
        # --------------------------------------------------------------------------------
        # Inject old state/context/plan into the prompt
        fast_context_text = context.to_fast_string()
        fast_sys_prompt   = get_fast_reply_prompt(
            state        = context.conversation_state,
            context_text = fast_context_text,
            plan         = context.plan,
        )

        # Print context being used
        #print(f"{CYAN}[ROBOT] Track 1: Generating reply...{RESET}")

        # API call for response
        t0 = time.time()
        robot_response: FastReply = run_agent(
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
        print_fast_reply(r1_time, robot_response)

        # Update histories
        history_robot.append({"role": "assistant", "content": buddy_msg})
        append_buddy_to_user_history(history_user, buddy_msg)

        # --------------------------------------------------------------------------------
        # c) Robot SLOW controller update (state + ContextOp deltas + last_topic + plan)
        # --------------------------------------------------------------------------------
        # Now we think about what just happened to prepare for the NEXT turn
        #print(f"{BLUE} [ROBOT] Track 2: Updating state/context...{RESET}")

        # Snapshot context before update (for print_diff)
        ctx_before      = context.snapshot()
        slow_sys_prompt = get_context_manager_prompt(
            current_state = context.conversation_state,
            context_json  = context.to_json(),
        )

        # API call for response
        t0 = time.time()
        slow_update: ContextManager = run_agent(
            client        = client,
            agent         = robot_slow,
            system_prompt = slow_sys_prompt,
            history       = history_robot,
        )

        # Timing
        t1 = time.time()
        r2_time = t1 - t0

        # Apply structured deltas
        context.apply_delta(slow_update.context_delta)

        # Advance state (gate + counter update)
        prev_state = context.conversation_state
        context.advance_state(slow_update.conversation_state)

        # Commit plan and last_topic for next turn
        context.plan              = slow_update.tentative_plan.strip()
        context.data.last_focus   = slow_update.last_topic.strip()

        # --------------------------------------------------------------------------------
        # d) End-of-turn steps
        # --------------------------------------------------------------------------------
        if config.verbose_slow: print_context_manager(r2_time, prev_state, ctx_before.data.last_focus, slow_update)

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
