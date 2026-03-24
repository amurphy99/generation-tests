"""
V1 simulation loop for the single-agent assistant.
--------------------------------------------------------------------------------
`src.simulation.v1_loop`

Both agents share the same model but maintain separate histories so each only
sees its own side of the conversation.

Usage:
    from src.simulation.config  import SimulationConfig
    from src.simulation.v1_loop import run_simulation

    config = SimulationConfig(model="gpt-4o-mini", turns=5)
    run_simulation(config)

"""
import json, time, instructor
from openai import OpenAI

# From this project
from ..config                import TIMEOUT
from ..utils.logging.logging import RESET, BOLD, UNBOLD, GREEN, CYAN, MAGENTA
from ..utils.logging.utils   import print_banner, print_turn_header

# Version-Specific Setup
from ..conversation_models.simulated_user import UserConversationResponse, USER_SYSTEM_PROMPT, print_user_turn
from ..conversation_models.buddy.models   import ConversationResponse
from ..conversation_models.buddy.prompts  import ROBOT_SYSTEM_PROMPT
from ..conversation_models.buddy.printing import print_robot_turn

# Generation Utilities
from  .config                  import SimulationConfig
from ..generation.agent_config import AgentConfig
from ..generation.multi_agent  import run_agent
from ..utils.history           import sync_history_robot, sync_history_user


# ================================================================================
# Client & Agent Construction
# ================================================================================
def _make_agents(config: SimulationConfig):
    robot_agent = AgentConfig(
        name           = "robot",
        model          = config.model,
        temperature    = 0.7,
        max_tokens     = 512,
        window         = 8,
        response_model = ConversationResponse,
    )
    user_agent = AgentConfig(
        name           = "user_sim",
        model          = config.model,
        temperature    = 0.7,
        max_tokens     = 512,
        window         = 7,
        response_model = UserConversationResponse,
    )
    return robot_agent, user_agent


# ================================================================================
# Conversation Loop
# ================================================================================
def run_simulation(config: SimulationConfig):
    """
    Run a single-agent conversation simulation.
    """
    print_banner("Starting Simulation: Buddy vs. Grandma (Structured)")

    # --------------------------------------------------------------------------------
    # 1) Initialize Client & Histories
    # --------------------------------------------------------------------------------
    # Client in instructor mode
    client = instructor.from_openai(
        OpenAI(base_url=config.LLM_URL, api_key=config.LLM_KEY, timeout=TIMEOUT),
        mode=instructor.Mode.JSON
    )

    # Initialize agents
    robot_agent, user_agent = _make_agents(config)

    # Conversation histories (system prompts are handled by run_agent, not stored here)
    # (user doesn't know about robots thought processes & vice versa)
    history_robot : list = []
    history_user  : list = []

    # --------------------------------------------------------------------------------
    # 2) Begin the Conversation (robot goes first)
    # --------------------------------------------------------------------------------
    # Manually send the greeting so the user has something to reply to
    start_message = "Good morning! My name is Buddy. It is nice to meet you. What is your name?"
    
    print(f"{CYAN}BUDDY (Start):{RESET} {start_message}\n")
    
    # Sync Histories (robot history sees JSON)
    history_user .append({"role": "user",      "content": f"[Buddy]: {start_message}"})
    history_robot.append({"role": "assistant", "content": json.dumps({
        "thought"            : "I should start the conversation by introducing myself to the user and asking their name.", 
        "conversation_state" : "start_conversation", 
        "message"            : start_message,
    })})

    # ================================================================================
    # 3) Main Loop
    # ================================================================================
    for i in range(1, config.turns + 1):
        print_turn_header(i)

        # --------------------------------------------------------------------------------
        # a) User speaks
        # --------------------------------------------------------------------------------
        print(f"{MAGENTA}[USER] Thinking...{RESET}")
        t0 = time.time()

        # User thinks based on Robot's last message
        user_response: UserConversationResponse = run_agent(
            client        = client,
            agent         = user_agent,
            system_prompt = USER_SYSTEM_PROMPT,
            history       = history_user,
        )

        t1 = time.time()
        print_user_turn(t1 - t0, user_response)

        # Sync histories
        sync_history_user(history_robot, history_user, user_response)

        # --------------------------------------------------------------------------------
        # b) Robot speaks
        # --------------------------------------------------------------------------------
        print(f"{CYAN}[ROBOT] Thinking...{RESET}")
        t0 = time.time()

        # Robot thinks based on User's message
        robot_response: ConversationResponse = run_agent(
            client        = client,
            agent         = robot_agent,
            system_prompt = ROBOT_SYSTEM_PROMPT,
            history       = history_robot,
        )

        t1 = time.time()
        print_robot_turn(t1 - t0, robot_response)

        # Sync histories
        sync_history_robot(history_robot, history_user, robot_response)

        # --------------------------------------------------------------------------------
        # c) End-of-turn steps
        # --------------------------------------------------------------------------------
        # Print just the replies
        print(
            f"\n"
            f"{BOLD}{GREEN  }User: {UNBOLD} { user_response.message}{RESET}\n"
            f"{BOLD}{MAGENTA}Buddy:{UNBOLD} {robot_response.message}{RESET}\n"
        )

        # Wait a little bit between turns
        if config.sleep_s > 0: time.sleep(config.sleep_s)

